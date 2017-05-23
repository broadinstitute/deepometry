import keras.layers
import keras.models


def _block(x, filters, downsample=True):
    y = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    y = keras.layers.Activation("relu")(y)
    y = keras.layers.normalization.BatchNormalization()(y)

    y = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(y)
    y = keras.layers.Activation("relu")(y)
    y = keras.layers.normalization.BatchNormalization()(y)

    if downsample:
        y = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding="same")(y)

    return y


class Model(keras.models.Model):
    def __init__(self, shape, classes):
        x = keras.layers.Input(shape)

        y = _block(x, 32, downsample=False)
        y = _block(y, 64)
        y = _block(y, 128)
        y = _block(y, 256)

        y = keras.layers.Flatten()(y)
        y = keras.layers.Dense(1024, activation="relu")(y)
        y = keras.layers.Dropout(0.5)(y)
        y = keras.layers.Dense(classes)(y)
        y = keras.layers.Activation("softmax")(y)

        super(Model, self).__init__(inputs=x, outputs=y)
