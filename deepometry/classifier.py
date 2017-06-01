import pkg_resources

import keras.callbacks
import keras.losses
import keras.optimizers
import keras.utils
import keras.wrappers.scikit_learn
import numpy
import numpy.random
import sklearn.utils

import deepometry.model


def create_model(input_shape=(32, 32, 1), classes=2):
    """
    Define and compile a deepometry model.

    :param input_shape: Input data shape (rows, columns, channels).
    :param classes: Number of classes.
    :return: A compiled deepometry model.
    """
    model = deepometry.model.Model(input_shape, classes)

    model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss=keras.losses.categorical_crossentropy,
        metrics=[
            "accuracy"
        ]
    )

    return model


class Classifier(keras.wrappers.scikit_learn.KerasClassifier):
    def __init__(self, input_shape, classes):
        """
        A scikit-learn compatible classifier using a deepometry model.

        :param input_shape: Input data shape (rows, columns, channels).
        :param classes: Number of classes.
        :param model_dir: (Optional) Directory to save training logs and model checkpoints.
        """
        options = {
            "batch_size": 32,
            "callbacks": [
                keras.callbacks.EarlyStopping(patience=8),
                keras.callbacks.ModelCheckpoint(pkg_resources.resource_filename("deepometry", "data/checkpoint.hdf5")),
                keras.callbacks.ReduceLROnPlateau()
            ],
            "epochs": 32,
            "shuffle": True,
            "validation_split": 0.2,
            "verbose": 1
        }

        self._classes = classes

        super(Classifier, self).__init__(create_model, input_shape=input_shape, classes=classes, **options)

    def fit(self, x, y, **kwargs):
        """
        Fit classifier.

        :param x: NumPy array of training data (N_samples, rows, columns channels).
        :param y: NumPy array target values (N_samples,).
        :param kwargs: (Optional) Arguments passed to keras.models.Sequential fit.
        """
        if "class_weight" not in kwargs:
            kwargs["class_weight"] = sklearn.utils.compute_class_weight(
                "balanced",
                numpy.arange(self._classes),
                y
            )

        if "shuffle" not in kwargs or kwargs["shuffle"]:
            indexes = numpy.random.permutation(len(x))

            x = numpy.asarray([x[index] for index in indexes])

            y = numpy.asarray([y[index] for index in indexes])

        y_cat = keras.utils.to_categorical(y, self._classes)

        super(Classifier, self).fit(x, y_cat, **kwargs)
