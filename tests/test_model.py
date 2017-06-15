import deepometry.model


def test_init():
    model = deepometry.model.Model((32, 32, 5), 3)

    assert len(model.layers) == 33
