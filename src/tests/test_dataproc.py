from dataclasses import dataclass
from ..dataproc import create_sk_pipe, InfoCompressor
from sklearn.pipeline import Pipeline
import os
import pandas as pd
import pytest
from pathlib import Path

curdir = Path(__file__).parent


@pytest.fixture
def df():
    return pd.read_csv(curdir / "test_training_data.csv")


@pytest.fixture
def df2():
    return pd.DataFrame(
        {
            "city1": ["c1", "c2", "c3", "c4", "c2", "c1", "c1"],
            "target": [1, 2, 3, 4, 5, 6, 7],
        }
    )


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
