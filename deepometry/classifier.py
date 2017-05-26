import os.path

import keras.callbacks
import keras.losses
import keras.optimizers
import keras.wrappers.scikit_learn
import numpy
import sklearn.utils

import deepometry.model


def create_model(input_shape=(32, 32, 1), classes=2):
    model = deepometry.model.Model(input_shape, classes)

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

        self._classes = classes

        super(Classifier, self).__init__(create_model, input_shape=input_shape, classes=classes, **options)

    def fit(self, x, y, **kwargs):
        if "class_weight" not in kwargs:
            kwargs["class_weight"] = sklearn.utils.compute_class_weight(
                "balanced",
                numpy.arange(self._classes),
                numpy.nonzero(y)[1]
            )

        super(Classifier, self).fit(x, y, **kwargs)
