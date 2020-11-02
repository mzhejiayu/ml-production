from typing import Any, List
from attr import dataclass
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

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
        if self.levels is None:
            raise ValueError("needs to fit before transform")

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


def create_sk_pipe() -> Pipeline:
    l_info_compressors = [
        ("city1", InfoCompressor("city1", 10, name_others="ci1o")),
        ("city2", InfoCompressor("city2", 10, name_others="ci2o")),
        ("isp1", InfoCompressor("isp1", 10, name_others="i1o")),
        ("isp2", InfoCompressor("isp2", 10, name_others="i2o")),
        ("country1", InfoCompressor("country1", 10, name_others="cou1o")),
        ("country2", InfoCompressor("country2", 10, name_others="cou2o")),
        ("continent1", InfoCompressor("continent1", 6, name_others="con1o")),
        ("continent2", InfoCompressor("continent2", 6, name_others="con2o")),
    ]
    p = Pipeline(
        l_info_compressors
        + [
            ("onehot", OneHotEncoder(sparse=False)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    return p


def train_tf_pipe(dest_path, df: pd.DataFrame):
    ...
