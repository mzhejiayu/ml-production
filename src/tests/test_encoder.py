from pathlib import Path

import pandas as pd
import pytest

from jsonschema import validate

from ..encoder import Encoder, create_encoder
from ..server import request_schema


def test_encoder(ppath, df: pd.DataFrame):
    encoder = create_encoder(ppath)
    assert isinstance(encoder, Encoder)

    X = df.iloc[:, 1:-1].values
    json_format_X = [[str(j) for j in i] for i in X]

    validate(json_format_X, request_schema)

    result = encoder.encode(json_format_X)

    assert result.shape[0] == len(X)
    assert result.shape[1] > 0
