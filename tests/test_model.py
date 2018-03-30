import csv
import os.path

import keras
import keras_resnet.models
import numpy
import pkg_resources
import pytest

import deepometry.image.iterator
import deepometry.model


@pytest.fixture()
def data_dir(tmpdir):
    return tmpdir.mkdir("data")


def test_init():
    model = deepometry.model.Model(shape=(48, 48, 3), units=4, directory="/home/mugatu/models", name="zoolander")

    assert model.model.input.shape.as_list() == [None, 48, 48, 3]

    assert model.model.output.shape.as_list() == [None, 4]

    assert model.directory == "/home/mugatu/models"

    assert model.name == "zoolander"


def test_compile():
    model = deepometry.model.Model(shape=(48, 48, 3), units=4)
    model.compile()

    assert model.model.loss == "categorical_crossentropy"

    assert model.model.metrics == ["accuracy"]

    assert isinstance(model.model.optimizer, keras.optimizers.Adam)


def test_fit_defaults(data_dir, mocker):
    numpy.random.seed(53)

    x = numpy.random.randint(256, size=(100, 48, 48, 3))
    y = numpy.random.randint(4, size=(100,))

    with mocker.patch("keras_resnet.models.ResNet50") as model_mock:
        keras_resnet.models.ResNet50.return_value = model_mock

        resources = mocker.patch("pkg_resources.resource_filename")
        resources.side_effect = lambda _, filename: str(data_dir.join(os.path.basename(filename)))

        model = deepometry.model.Model(shape=(48, 48, 3), units=4)
        model.compile()
        model.fit(
            x,
            y,
            batch_size=10,
            epochs=1,
            validation_split=0.1,
            verbose=0
        )

        model_mock.fit_generator.assert_called_once_with(
            callbacks=mocker.ANY,
            class_weight=mocker.ANY,
            epochs=1,
            generator=mocker.ANY,
            steps_per_epoch=9,
            validation_data=mocker.ANY,
            validation_steps=1,
            verbose=0
        )

        _, kwargs = model_mock.fit_generator.call_args

        # callbacks
        callbacks = kwargs["callbacks"]
        assert len(callbacks) == 4

        assert isinstance(callbacks[0], keras.callbacks.CSVLogger)
        assert callbacks[0].filename == pkg_resources.resource_filename(
            "deepometry",
            os.path.join("data", "training.csv")
        )

        assert isinstance(callbacks[1], keras.callbacks.EarlyStopping)

        assert isinstance(callbacks[2], keras.callbacks.ModelCheckpoint)
        assert callbacks[2].filepath == pkg_resources.resource_filename(
            "deepometry",
            os.path.join("data", "checkpoint.hdf5")
        )

        assert isinstance(callbacks[3], keras.callbacks.ReduceLROnPlateau)

        # generator
        generator = kwargs["generator"]
        assert isinstance(generator, deepometry.image.iterator.NumpyArrayIterator)
        assert generator.batch_size == 10
        assert generator.x.shape == (90, 48, 48, 3)
        assert generator.image_data_generator.height_shift_range == 0.5
        assert generator.image_data_generator.horizontal_flip == True
        assert generator.image_data_generator.rotation_range == 180
        assert generator.image_data_generator.vertical_flip == True
        assert generator.image_data_generator.width_shift_range == 0.5

        x_train = generator.x
        sample = x_train[0]

        expected = numpy.empty((48, 48, 3))
        expected[:, :, 0] = (sample[:, :, 0] - numpy.mean(x_train[:, :, :, 0]) + 255.0) / (2.0 * 255.0)
        expected[:, :, 1] = (sample[:, :, 1] - numpy.mean(x_train[:, :, :, 1]) + 255.0) / (2.0 * 255.0)
        expected[:, :, 2] = (sample[:, :, 2] - numpy.mean(x_train[:, :, :, 2]) + 255.0) / (2.0 * 255.0)

        actual = generator.image_data_generator.preprocessing_function(sample)

        numpy.testing.assert_array_almost_equal(actual, expected, decimal=5)

        # validation_data
        validation_data = kwargs["validation_data"]
        assert isinstance(validation_data, deepometry.image.iterator.NumpyArrayIterator)
        assert validation_data.batch_size == 10
        assert validation_data.x.shape == (10, 48, 48, 3)
        assert validation_data.image_data_generator.height_shift_range == 0.5
        assert validation_data.image_data_generator.horizontal_flip == True
        assert validation_data.image_data_generator.rotation_range == 180
        assert validation_data.image_data_generator.vertical_flip == True
        assert validation_data.image_data_generator.width_shift_range == 0.5

        x_valid = validation_data.x
        sample = x_valid[0]

        expected = numpy.empty((48, 48, 3))
        expected[:, :, 0] = (sample[:, :, 0] - numpy.mean(x_train[:, :, :, 0]) + 255.0) / (2.0 * 255.0)
        expected[:, :, 1] = (sample[:, :, 1] - numpy.mean(x_train[:, :, :, 1]) + 255.0) / (2.0 * 255.0)
        expected[:, :, 2] = (sample[:, :, 2] - numpy.mean(x_train[:, :, :, 2]) + 255.0) / (2.0 * 255.0)

        actual = generator.image_data_generator.preprocessing_function(sample)

        numpy.testing.assert_array_almost_equal(actual, expected, decimal=5)


def test_fit_named_model(data_dir, mocker):
    numpy.random.seed(53)

    x = numpy.random.randint(256, size=(100, 48, 48, 3))
    y = numpy.random.randint(4, size=(100,))

    with mocker.patch("keras_resnet.models.ResNet50") as model_mock:
        keras_resnet.models.ResNet50.return_value = model_mock

        resources = mocker.patch("pkg_resources.resource_filename")
        resources.side_effect = lambda _, filename: str(data_dir.join(os.path.basename(filename)))

        model = deepometry.model.Model(shape=(48, 48, 3), units=4, name="zoolander")
        model.compile()
        model.fit(
            x,
            y,
            batch_size=10,
            epochs=1,
            validation_split=0.1,
            verbose=0
        )

        _, kwargs = model_mock.fit_generator.call_args

        # callbacks
        callbacks = kwargs["callbacks"]

        assert callbacks[0].filename == pkg_resources.resource_filename(
            "deepometry",
            os.path.join("data", "zoolander_training.csv")
        )

        assert callbacks[2].filepath == pkg_resources.resource_filename(
            "deepometry",
            os.path.join("data", "zoolander_checkpoint.hdf5")
        )

        assert os.path.exists(
            pkg_resources.resource_filename(
                "deepometry",
                os.path.join("data", "zoolander_means.csv")
            )
        )


def test_fit_named_directory(data_dir, mocker):
    numpy.random.seed(53)

    x = numpy.random.randint(256, size=(100, 48, 48, 3))
    y = numpy.random.randint(4, size=(100,))

    with mocker.patch("keras_resnet.models.ResNet50") as model_mock:
        keras_resnet.models.ResNet50.return_value = model_mock

        model_directory = str(data_dir.mkdir("models"))

        model = deepometry.model.Model(shape=(48, 48, 3), units=4, directory=model_directory)
        model.compile()
        model.fit(
            x,
            y,
            batch_size=10,
            epochs=1,
            validation_split=0.1,
            verbose=0
        )

        _, kwargs = model_mock.fit_generator.call_args

        # callbacks
        callbacks = kwargs["callbacks"]

        assert callbacks[0].filename == os.path.join(model_directory, "training.csv")

        assert callbacks[2].filepath == os.path.join(model_directory, "checkpoint.hdf5")

        assert os.path.exists(os.path.join(model_directory, "means.csv"))


def test_evaluate_defaults(data_dir, mocker):
    x = numpy.random.randint(256, size=(100, 48, 48, 3)).astype(numpy.float64)
    y = numpy.random.randint(4, size=(100,))

    meanscsv = str(data_dir.join("means.csv"))
    with open(meanscsv, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([125.3, 127.12, 121.9])

    expected_samples = x.copy()
    expected_samples[:, :, :, 0] = (expected_samples[:, :, :, 0] - 125.3 + 255.0) / (2.0 * 255.0)
    expected_samples[:, :, :, 1] = (expected_samples[:, :, :, 1] - 127.12 + 255.0) / (2.0 * 255.0)
    expected_samples[:, :, :, 2] = (expected_samples[:, :, :, 2] - 121.9 + 255.0) / (2.0 * 255.0)

    expected_targets = keras.utils.to_categorical(y, 4)

    with mocker.patch("keras_resnet.models.ResNet50") as model_mock:
        keras_resnet.models.ResNet50.return_value = model_mock

        resources = mocker.patch("pkg_resources.resource_filename")
        resources.side_effect = lambda _, filename: str(data_dir.join(os.path.basename(filename)))

        model = deepometry.model.Model(shape=(48, 48, 3), units=4)
        model.compile()
        model.evaluate(
            x,
            y,
            batch_size=10,
            verbose=0
        )

        model_mock.load_weights.assert_called_once_with(
            pkg_resources.resource_filename("deepometry", os.path.join("data", "checkpoint.hdf5"))
        )

        model_mock.evaluate.assert_called_once_with(
            x=mocker.ANY,
            y=mocker.ANY,
            batch_size=10,
            verbose=0
        )

        _, kwargs = model_mock.evaluate.call_args

        samples = kwargs["x"]
        assert samples.shape == expected_samples.shape
        numpy.testing.assert_array_equal(samples, expected_samples)

        targets = kwargs["y"]
        assert targets.shape == expected_targets.shape
        numpy.testing.assert_array_equal(targets, expected_targets)


def test_evaluate_named_model(data_dir, mocker):
    x = numpy.random.randint(256, size=(100, 48, 48, 3)).astype(numpy.float64)
    y = numpy.random.randint(4, size=(100,))

    meanscsv = str(data_dir.join("zoolander_means.csv"))
    with open(meanscsv, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([125.3, 127.12, 121.9])

    expected_samples = x.copy()
    expected_samples[:, :, :, 0] = (expected_samples[:, :, :, 0] - 125.3 + 255.0) / (2.0 * 255.0)
    expected_samples[:, :, :, 1] = (expected_samples[:, :, :, 1] - 127.12 + 255.0) / (2.0 * 255.0)
    expected_samples[:, :, :, 2] = (expected_samples[:, :, :, 2] - 121.9 + 255.0) / (2.0 * 255.0)

    expected_targets = keras.utils.to_categorical(y, 4)

    with mocker.patch("keras_resnet.models.ResNet50") as model_mock:
        keras_resnet.models.ResNet50.return_value = model_mock

        resources = mocker.patch("pkg_resources.resource_filename")
        resources.side_effect = lambda _, filename: str(data_dir.join(os.path.basename(filename)))

        model = deepometry.model.Model(shape=(48, 48, 3), units=4, name="zoolander")
        model.compile()
        model.evaluate(
            x,
            y,
            batch_size=10,
            verbose=0
        )

        model_mock.load_weights.assert_called_once_with(
            pkg_resources.resource_filename(
                "deepometry",
                os.path.join("data", "zoolander_checkpoint.hdf5")
            )
        )

        model_mock.evaluate.assert_called_once_with(
            x=mocker.ANY,
            y=mocker.ANY,
            batch_size=10,
            verbose=0
        )

        _, kwargs = model_mock.evaluate.call_args

        samples = kwargs["x"]
        assert samples.shape == expected_samples.shape
        numpy.testing.assert_array_equal(samples, expected_samples)

        targets = kwargs["y"]
        assert targets.shape == expected_targets.shape
        numpy.testing.assert_array_equal(targets, expected_targets)


def test_evaluate_named_directory(data_dir, mocker):
    x = numpy.random.randint(256, size=(100, 48, 48, 3)).astype(numpy.float64)
    y = numpy.random.randint(4, size=(100,))

    model_directory = data_dir.mkdir("models")

    meanscsv = str(model_directory.join("means.csv"))
    with open(meanscsv, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([125.3, 127.12, 121.9])

    expected_samples = x.copy()
    expected_samples[:, :, :, 0] = (expected_samples[:, :, :, 0] - 125.3 + 255.0) / (2.0 * 255.0)
    expected_samples[:, :, :, 1] = (expected_samples[:, :, :, 1] - 127.12 + 255.0) / (2.0 * 255.0)
    expected_samples[:, :, :, 2] = (expected_samples[:, :, :, 2] - 121.9 + 255.0) / (2.0 * 255.0)

    expected_targets = keras.utils.to_categorical(y, 4)

    with mocker.patch("keras_resnet.models.ResNet50") as model_mock:
        keras_resnet.models.ResNet50.return_value = model_mock

        model = deepometry.model.Model(shape=(48, 48, 3), units=4, directory=str(model_directory))
        model.compile()
        model.evaluate(
            x,
            y,
            batch_size=10,
            verbose=0
        )

        model_mock.load_weights.assert_called_once_with(
            os.path.join(str(model_directory), "checkpoint.hdf5")
        )

        model_mock.evaluate.assert_called_once_with(
            x=mocker.ANY,
            y=mocker.ANY,
            batch_size=10,
            verbose=0
        )

        _, kwargs = model_mock.evaluate.call_args

        samples = kwargs["x"]
        assert samples.shape == expected_samples.shape
        numpy.testing.assert_array_equal(samples, expected_samples)

        targets = kwargs["y"]
        assert targets.shape == expected_targets.shape
        numpy.testing.assert_array_equal(targets, expected_targets)
