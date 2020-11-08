import os
import requests
from src.encoder import Encoder
from joblib import load
from src.dataproc import train_sk_pipe
import pytest
from tensorflow.keras import Sequential
from ..model import create_model
from pathlib import Path
import json

import shutil

curdir = Path(__file__).parent


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_create_model(training_xy):
    x, y = training_xy
    model: Sequential = create_model([x.shape[1]])
    assert model.layers[0].count_params() == 77

    model.fit(x, y)
    model.save(curdir / "saved_model")


def test_integration(model_service, xy):
    p = Path(curdir / "saved_model")
    assert "http://127.0.0.1:8501" == model_service

    fname = curdir / ".tmp.joblib"
    train_sk_pipe(fname, xy[0])
    assert os.path.exists(fname)
    pipe = load(fname)

    encoder = Encoder(pipe)
    matrixs = encoder.encode(xy[0][:100]).tolist()

    res = requests.post(
        model_service + "/v1/models/tp_pred:predict",
        data=json.dumps({"instances": matrixs}),
    )

    assert len(res.json()["predictions"]) == 100
