import pytest
from pathlib import Path
import pandas as pd
from ..encoder import Encoder, create_encoder
import requests

curdir = Path(__file__).parent


def is_responsive(url):
    try:
        response = requests.get(url)
        if response.status_code == 200 or response.status_code == 404:
            return True
    except ConnectionError:
        return False


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return curdir / "docker-compose.yml"


@pytest.fixture
def ppath():
    return Path(__file__).parent / "test_pipe.joblib"


@pytest.fixture
def schema():
    return {
        "type": "object",
        "properties": {
            "city1": {"type": "string"},
            "city2": {"type": "string"},
            "country1": {"type": "string"},
            "country2": {"type": "string"},
            "isp1": {"type": "string"},
            "isp2": {"type": "string"},
            "continent1": {"type": "string"},
            "continent2": {"type": "string"},
            "target": {"type": "number"},
        },
    }


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


@pytest.fixture
def xy(df):
    y = df["target"]
    x = df.iloc[:, 1:-1]
    return x, y


@pytest.fixture
def training_xy(xy, ppath):
    x, y = xy
    encoder = create_encoder(ppath)
    encoded_x = encoder.encode(x)
    return encoded_x, y


@pytest.fixture()
def model_service(docker_ip, docker_services):
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("tfs", 8501)
    url = "http://{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive(url)
    )
    return url


@pytest.fixture
def info_compress_config():
    return [10, 10, 10, 10, 6, 6, 10, 10]