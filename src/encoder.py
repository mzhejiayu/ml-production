from dataclasses import dataclass
from typing import List

import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline


@dataclass
class Encoder:
    """Encoder will encode the raw data. json, """

    pipe: Pipeline

    def encode(self, raw_data: List[List[str]]):
        """
        Args:
            raw_data ([type]): [description]
        """
        # reconstruct with raw_data
        df = pd.DataFrame(raw_data)
        df.columns = [
            "city1",
            "city2",
            "isp1",
            "isp2",
            "country1",
            "country2",
            "continent1",
            "continent2",
        ]

        return self.pipe.transform(df)


def create_encoder(ppath: str) -> Encoder:
    pipe = load(ppath)
    return Encoder(pipe)
