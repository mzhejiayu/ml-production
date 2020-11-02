import pytest
from tensorflow.keras import Sequential
from ..model import create_model


def test_create_model(training_xy):
    x, y = training_xy
    model: Sequential = create_model([x.shape[1]])
    assert model.layers[0].count_params() == 77

    model.fit(x, y)
    model.save("saved_model")
