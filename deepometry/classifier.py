import os.path

import keras.callbacks
import keras.layers
import keras.losses
import keras.models
import keras.optimizers
import keras.wrappers.scikit_learn


def _add_block(model, filters, input_shape=None):
    if input_shape:
        model.add(keras.layers.Conv2D(filters, (3, 3), padding="same", input_shape=input_shape))
    else:
        model.add(keras.layers.Conv2D(filters, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.normalization.BatchNormalization())

    model.add(keras.layers.Conv2D(filters, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.normalization.BatchNormalization())

    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=None, padding="same"))

    return model


def create_model(input_shape=(32, 32, 1), classes=2):
    model = keras.models.Sequential()

    model = _add_block(model, 32, input_shape=input_shape)
    model = _add_block(model, 64)
    model = _add_block(model, 128)
    model = _add_block(model, 256)

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(classes))
    model.add(keras.layers.Activation("softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(0.00001),
        loss=keras.losses.categorical_crossentropy,
        metrics=[
            "accuracy"
        ]
    )

    return model


class Classifier(keras.wrappers.scikit_learn.KerasClassifier):
    def __init__(self, input_shape, classes, model_dir=None):
        callbacks = []

        if model_dir:
            callbacks = [
                keras.callbacks.CSVLogger(os.path.join(model_dir, "training.csv")),
                keras.callbacks.ModelCheckpoint(os.path.join(model_dir, "checkpoint.hdf5"))
            ]

        options = {
            "batch_size": 32,
            "callbacks": callbacks,
            "epochs": 8,
            "shuffle": True,
            "validation_split": 0.2,
            "verbose": 1
        }

        super(Classifier, self).__init__(create_model, input_shape=input_shape, classes=classes, **options)
