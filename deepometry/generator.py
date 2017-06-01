import os.path

import keras.backend
import keras.preprocessing.image
import numpy


class NumpyArrayIterator(keras.preprocessing.image.Iterator):
    """Iterator yielding data from a Numpy array.
    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
    """
    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix=''):
        if y is not None and len(x) != len(y):
            raise ValueError(
                "x and y should have the same length. Found: x.shape = {}, y.shape = {}".format(
                    numpy.asarray(x).shape, numpy.asarray(y).shape)
            )

        if data_format is None:
            data_format = keras.backend.image_data_format()

        self.x = numpy.asarray(x, dtype=keras.backend.floatx())

        if self.x.ndim != 4:
            raise ValueError(
                "Input data in `NumpyArrayIterator` should have rank 4. You passed an array with shape",
                self.x.shape
            )

        if y is not None:
            self.y = numpy.asarray(y)
        else:
            self.y = None

        self.image_data_generator = image_data_generator

        self.data_format = data_format

        self.save_to_dir = save_to_dir

        self.save_prefix = save_prefix

        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = numpy.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=keras.backend.floatx())

        for i, j in enumerate(index_array):
            x = self.x[j]

            x = self.image_data_generator.random_transform(x.astype(keras.backend.floatx()))

            batch_x[i] = x

        if self.save_to_dir:
            for i in range(current_batch_size):
                fname = "{prefix}_{index}_{hash}.npy".format(
                    prefix=self.save_prefix,
                    index=current_index + i,
                    hash=numpy.random.randint(int(1e4))
                )

                numpy.save(os.path.join(self.save_to_dir, fname), batch_x[i])

        if self.y is None:
            return batch_x

        batch_y = self.y[index_array]

        return batch_x, batch_y


class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.
    # Arguments
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """
    def __init__(self,
                 rotation_range=0.0,
                 width_shift_range=0.0,
                 height_shift_range=0.0,
                 shear_range=0.0,
                 zoom_range=0.0,
                 fill_mode="nearest",
                 cval=0.0,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        if data_format is None:
            data_format = keras.backend.image_data_format()

        self.rotation_range = rotation_range

        self.width_shift_range = width_shift_range

        self.height_shift_range = height_shift_range

        self.shear_range = shear_range

        self.zoom_range = zoom_range

        self.fill_mode = fill_mode

        self.cval = cval

        self.horizontal_flip = horizontal_flip

        self.vertical_flip = vertical_flip

        self.rescale = rescale

        self.preprocessing_function = preprocessing_function

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError(
                "data_format should be \"channels_last\" (channel after row and column) or \"channels_first\" "
                "(channel before row and column). Received arg: ",
                data_format
            )

        self.data_format = data_format

        if data_format == "channels_first":
            self.channel_axis = 0
            self.row_axis = 1
            self.col_axis = 2

        if data_format == "channels_last":
            self.channel_axis = 2
            self.row_axis = 0
            self.col_axis = 1

        if numpy.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError(
                "zoom_range should be a float or a tuple or list of two floats. Received arg: ",
                zoom_range
            )

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix=""):
        return NumpyArrayIterator(
            x, y, self,
            batch_size=batch_size,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            shuffle=shuffle
        )

    def random_transform(self, x):
        """Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = numpy.pi / 180 * numpy.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = numpy.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[self.row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = numpy.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[self.col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = numpy.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = numpy.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None

        if theta != 0:
            rotation_matrix = numpy.array(
                [[numpy.cos(theta), -numpy.sin(theta), 0],
                 [numpy.sin(theta), numpy.cos(theta), 0],
                 [0, 0, 1]]
            )

            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = numpy.array(
                [[1, 0, tx],
                 [0, 1, ty],
                 [0, 0, 1]]
            )

            transform_matrix = shift_matrix if transform_matrix is None else numpy.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = numpy.array(
                [[1, -numpy.sin(shear), 0],
                 [0, numpy.cos(shear), 0],
                 [0, 0, 1]]
            )

            transform_matrix = shear_matrix if transform_matrix is None else numpy.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = numpy.array(
                [[zx, 0, 0],
                 [0, zy, 0],
                 [0, 0, 1]]
            )

            transform_matrix = zoom_matrix if transform_matrix is None else numpy.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[self.row_axis], x.shape[self.col_axis]

            transform_matrix = keras.preprocessing.image.transform_matrix_offset_center(transform_matrix, h, w)

            x = keras.preprocessing.image.apply_transform(
                x,
                transform_matrix,
                self.channel_axis,
                fill_mode=self.fill_mode,
                cval=self.cval
            )

        if self.horizontal_flip:
            if numpy.random.random() < 0.5:
                x = keras.preprocessing.image.flip_axis(x, self.col_axis)

        if self.vertical_flip:
            if numpy.random.random() < 0.5:
                x = keras.preprocessing.image.flip_axis(x, self.row_axis)

        return x
