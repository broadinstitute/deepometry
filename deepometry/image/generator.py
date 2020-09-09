import sys
sys.path.insert(0, '/home/paul/.conda/envs/tensorflow/lib/python3.6/site-packages')

import keras.preprocessing.image

import deepometry.image.iterator_balanced, deepometry.image.iterator


class ImageDataGenerator(keras.preprocessing.image.ImageDataGenerator):
    def __init__(self,
                 height_shift_range=0.0,
                 horizontal_flip=False,
                 preprocessing_function=None,
                 rotation_range=0.0,
                 vertical_flip=False,
                 width_shift_range=0.0):
        super(ImageDataGenerator, self).__init__(
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip,
            preprocessing_function=preprocessing_function,
            rotation_range=rotation_range,
            vertical_flip=vertical_flip,
            width_shift_range=width_shift_range
        )

    def flow(self, x,
             y=None,
             batch_size=32,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix="",
             save_format="tif",
             balance=True,
             mixup_alpha=0.0):
        
        if balance:
            return deepometry.image.iterator_balanced.NumpyArrayIterator(
                x, y, self,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format,
                mixup_alpha=mixup_alpha
            )
        else:
            return deepometry.image.iterator.NumpyArrayIterator(
                x, y, self,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format,
                mixup_alpha=mixup_alpha
            )            

    def flow_from_directory(self, directory,
                            target_size=(48, 48),
                            color_mode="rgb",
                            classes=None,
                            class_mode="categorical",
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix="",
                            save_format="tif",
                            follow_links=False):
        raise NotImplementedError()
