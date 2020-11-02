from dataclasses import dataclass
from sklearn.pipeline import Pipeline


@dataclass
class RowVectorRawFormat:
    # pid is important because this will encsure that the
    # model doesn't return the same peer id as the request peer
    pid: str
    city: str
    country: str
    continent: str
    isp: str


class Encoder:
    """Encoder will encode the raw data. json, """

    def __init__(self, pipeline: Pipeline):

        ...

    def encode(self, raw_data):
        """
        Args:
            raw_data ([type]): [description]
        """
        ...
