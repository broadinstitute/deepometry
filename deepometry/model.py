import keras.layers
import keras.models


def _block(filters, input_shape=None):
    layers = []

    if input_shape:
        layers.append(keras.layers.Conv2D(filters, (3, 3), padding="same", input_shape=input_shape))
    else:
        layers.append(keras.layers.Conv2D(filters, (3, 3), padding="same"))
    layers.append(keras.layers.Activation("relu"))
    layers.append(keras.layers.normalization.BatchNormalization())

    layers.append(keras.layers.Conv2D(filters, (3, 3), padding="same"))
    layers.append(keras.layers.Activation("relu"))
    layers.append(keras.layers.normalization.BatchNormalization())

    layers.append(keras.layers.MaxPooling2D(pool_size=2, strides=None, padding="same"))

    return layers


class Model(keras.models.Sequential):
    def __init__(self, shape, classes):
        layers = _block(32, input_shape=shape)
        layers += _block(64)
        layers += _block(128)
        layers += _block(256)

        layers += [
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(classes),
            keras.layers.Activation("softmax")
        ]

        super(Model, self).__init__(layers)
