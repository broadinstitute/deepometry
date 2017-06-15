import pkg_resources

import keras.callbacks
import keras.layers
import keras.models
import keras_resnet.models


class Model(keras.models.Model):
    def __init__(self, shape=(48, 48, 3), classes=4):
        """
        Image classification model for single-cell images.

        :param shape: Image shape (width, height, channels).
        :param classes: Number of predicted classes.
        """
        x = keras.layers.Input(shape)

        y = keras_resnet.ResNet200(x)

        y = keras.layers.Flatten()(y.output)

        y = keras.layers.Dense(classes, activation="softmax")(y)

        self.batch_size = 256

        self.callbacks = [
            keras.callbacks.CSVLogger(pkg_resources.resource_filename("deepometry", "data/training.csv")),
            keras.callbacks.EarlyStopping(patience=20),
            keras.callbacks.ModelCheckpoint(pkg_resources.resource_filename("deepometry", "data/checkpoint.hdf5")),
            keras.callbacks.ReduceLROnPlateau()
        ]

        super(Model, self).__init__(x, y)

    def compile(self, optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
                loss_weights=None,
                sample_weight_mode=None,
                **kwargs):
        super(Model, self).compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            sample_weight_mode=sample_weight_mode,
            **kwargs
        )
