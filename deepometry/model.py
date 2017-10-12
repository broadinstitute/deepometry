import csv
import os.path

import keras
import keras.preprocessing.image
import keras_resnet.models
import numpy
import pkg_resources

import deepometry.image.generator


class Model(object):
    def __init__(self, shape, units, directory=None, name=None):
        """
        Create a model for single-cell image classification.

        :param shape: Input image shape, including channels. Grayscale data should specify channels as 1. Check your
                      keras configuration for channel order (e.g., "image_data_format": "channels_last"). Usually,
                      this configuration is defined at `$HOME/.keras/keras.json`, or `%USERPROFILE%\.keras\keras.json`
                      on Windows.
        :param units: Number of predictable classes.
        :param directory: (Optional) Output directory for model checkpoints, metrics, and metadata. Otherwise, the
                          package's data directory is used.
        :param name: (Optional) A unique identifier for referencing this model.
        """
        self.directory = directory

        self.name = name

        self.units = units

        x = keras.layers.Input(shape)

        self.model = keras_resnet.models.ResNet50(x, classes=units)

    def compile(self):
        """
        Configure the model.
        """
        self.model.compile(
            loss="categorical_crossentropy",
            metrics=[
                "accuracy"
            ],
            optimizer="adam"
        )

    def evaluate(self, x, y, batch_size=32, verbose=0):
        """
        Compute the loss value & metrics values for the model in test mode.

        Computation is done in batches.

        :param x: NumPy array of test data.
        :param y: NumPy array of target data.
        :param batch_size: Number of samples evaluated per batch.
        :param verbose: Verbosity mode, 0 = silent, or 1 = verbose.
        :return: Tuple of scalars: (loss, accuracy).
        """
        self.model.load_weights(self._resource("checkpoint.hdf5"))

        return self.model.evaluate(
            x=self._center(x),
            y=keras.utils.to_categorical(y, num_classes=self.units),
            batch_size=batch_size,
            verbose=verbose
        )

    def fit(self, x, y, batch_size=32, epochs=512, validation_split=0.2, verbose=0):
        """
        Train the model for a fixed number of epochs (iterations on a dataset). Training will automatically stop
        if the validation loss fails to improve for 20 epochs.

        :param x: NumPy array of training data.
        :param y: NumPy array of target data.
        :param batch_size: Number of samples per gradient update.
        :param epochs: Number of times to iterate over the training data arrays.
        :param validation_split: Fraction of the training data to be used as validation data.
        :param verbose: Verbosity mode. 0 = silent, 1 = verbose, 2 = one log line per epoch.
        """
        x_train, y_train, x_valid, y_valid = _split(x, y, validation_split)

        self._calculate_means(x_train)

        train_generator = self._create_generator()

        valid_generator = self._create_generator()

        options = {
            "callbacks": [
                keras.callbacks.CSVLogger(
                    self._resource("training.csv")
                ),
                keras.callbacks.EarlyStopping(patience=20),
                keras.callbacks.ModelCheckpoint(
                    self._resource("checkpoint.hdf5")
                ),
                keras.callbacks.ReduceLROnPlateau()
            ],
            "epochs": epochs,
            "steps_per_epoch": len(x_train) // batch_size,
            "validation_steps": len(x_valid) // batch_size,
            "verbose": verbose
        }

        self.model.fit_generator(
            generator=train_generator.flow(
                x=x_train,
                y=keras.utils.to_categorical(y_train, num_classes=self.units),
                batch_size=batch_size
            ),
            validation_data=valid_generator.flow(
                x=x_valid,
                y=keras.utils.to_categorical(y_valid, num_classes=self.units),
                batch_size=batch_size
            ),
            **options
        )

    def predict(self, x, batch_size=32, verbose=0):
        """
        Make predictions for the input samples.

        Computation is done in batches.

        :param x: NumPy array of input data.
        :param batch_size: Number of samples predicted per batch.
        :param verbose: Verbosity mode, 0 = silent, or 1 = verbose.
        :return: NumPy array of predictions.
        """
        self.model.load_weights(self._resource("checkpoint.hdf5"))

        return self.model.predict(self._center(x), batch_size=batch_size, verbose=verbose)

    def _calculate_means(self, x):
        reshaped = x.reshape(-1, x.shape[-1])

        means = numpy.mean(reshaped, axis=0)

        with open(self._resource("means.csv"), "w") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(means)

        return means

    def _center(self, x):
        xc = x.reshape(-1, x.shape[-1])

        xc = xc - self._means()

        return xc.reshape(x.shape)

    def _create_generator(self):
        means = self._means()

        generator_options = {
            "height_shift_range": 0.5,
            "horizontal_flip": True,
            "preprocessing_function": lambda data: data - means,
            "rotation_range": 180,
            "vertical_flip": True,
            "width_shift_range": 0.5
        }

        return deepometry.image.generator.ImageDataGenerator(
            **generator_options
        )

    def _means(self):
        means = None

        with open(self._resource("means.csv"), "r") as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                means = [float(mean) for mean in row]

                break

        return means

    def _resource(self, filename):
        if self.name is None:
            resource_filename = filename
        else:
            resource_filename = "{:s}_{:s}".format(self.name, filename)

        if self.directory is None:
            return pkg_resources.resource_filename(
                "deepometry",
                os.path.join("data", resource_filename)
            )

        return os.path.join(self.directory, resource_filename)


def _split(x, y, validation_split=0.2):
    split_index = int(len(x) * (1.0 - validation_split))

    indexes = numpy.random.permutation(len(x))

    x_train = numpy.asarray([x[index] for index in indexes[:split_index]])
    x_valid = numpy.asarray([x[index] for index in indexes[split_index:]])

    y_train = numpy.asarray([y[index] for index in indexes[:split_index]])
    y_valid = numpy.asarray([y[index] for index in indexes[split_index:]])

    return x_train, y_train, x_valid, y_valid
