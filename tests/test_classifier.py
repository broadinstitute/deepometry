import keras.callbacks

import deepometry.classifier


def test_init():
    classifier = deepometry.classifier.Classifier(input_shape=(32, 32, 5), classes=3, model_dir="tmp")

    model = classifier.build_fn()

    assert len(model.layers) == 33

    params = classifier.get_params()

    assert params["batch_size"] == 32

    assert len(params["callbacks"]) == 2

    assert isinstance(params["callbacks"][0], keras.callbacks.CSVLogger)

    assert isinstance(params["callbacks"][1], keras.callbacks.ModelCheckpoint)

    assert params["epochs"] == 8

    assert params["shuffle"]

    assert params["validation_split"] == 0.2
