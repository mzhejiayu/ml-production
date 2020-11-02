from dataclasses import dataclass
import os
from pathlib import Path

from joblib import load
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ..dataproc import InfoCompressor, create_sk_pipe, train_sk_pipe


def test_info_compressor(df2):
    ic = InfoCompressor("city1", 2)
    with pytest.raises(ValueError):
        ic.transform(df2)

    X = df2[["city1"]]

    ic.fit(X)
    assert len(ic.levels) == 2
    assert ic.levels == ["c1", "c2"]

    transformed_x = ic.transform(X)
    assert len(transformed_x) == len(df2)
    assert list(transformed_x.city1.values) == [
        "c1",
        "c2",
        "others",
        "others",
        "c2",
        "c1",
        "c1",
    ]


def test_create_sk_pipe(df):
    pipe = create_sk_pipe()
    assert isinstance(pipe, Pipeline)

    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    X_transformed = pipe.fit_transform(X, y)
    assert X_transformed.shape[0] == len(X)
    assert 11 * 6 + 7 * 2 > X_transformed.shape[1] > len(X.columns)


def test_train_sk_pipe(df):
    fname = "example.joblib"
    train_sk_pipe(fname, df)
    assert os.path.exists(fname)

    pipe = load(fname)
    assert isinstance(pipe, Pipeline)
    assert len(pipe.steps) > 0
    os.remove(fname)
