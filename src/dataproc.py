from typing import Any, List

from attr import dataclass
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__doc__ = """
This module contains function to train the pipeline and save the artifact
"""


@dataclass
class InfoCompressor(BaseEstimator, TransformerMixin):
    """InfoCompressor compresses the extra informations

    Args:
        BaseEstimator ([type]): [description]
        TransformerMixin ([type]): [description]
    """

    col_name: str
    top_n: int
    levels: List[Any] = []
    name_others: str = "others"

    def fit(self, X: pd.DataFrame, y=None):
        self.levels = list(
            X[self.col_name].value_counts(ascending=False)[: self.top_n].index
        )

    def transform(self, X: pd.DataFrame):

        if len(self.levels) == 0:
            raise ValueError("levels don't contain any level")

        hidden = X[self.col_name].where(
            X[self.col_name].isin(self.levels), self.name_others
        )

        X[self.col_name] = hidden
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def create_sk_pipe(config: List[int] = [10, 10, 10, 10, 6, 6, 10, 10]) -> Pipeline:
    assert len(config) == 8
    l_info_compressors = [
        ("city1", InfoCompressor("city1", config[0], name_others="ci1o")),
        ("city2", InfoCompressor("city2", config[1], name_others="ci2o")),
        ("country1", InfoCompressor("country1", config[2], name_others="cou1o")),
        ("country2", InfoCompressor("country2", config[3], name_others="cou2o")),
        ("continent1", InfoCompressor("continent1", config[4], name_others="con1o")),
        ("continent2", InfoCompressor("continent2", config[5], name_others="con2o")),
        ("isp1", InfoCompressor("isp1", config[6], name_others="i1o")),
        ("isp2", InfoCompressor("isp2", config[7], name_others="i2o")),
    ]
    p = Pipeline(
        l_info_compressors
        + [
            ("onehot", OneHotEncoder(sparse=False)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    return p


def train_sk_pipe(dest_path, X: pd.DataFrame, **kwargs):
    pipe = create_sk_pipe(**kwargs)
    pipe.fit(X)
    joblib.dump(pipe, dest_path)


def train_tf_pipe(dest_path, df: pd.DataFrame):
    ...  # pragma: no cover
