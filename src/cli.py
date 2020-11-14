from pathlib import Path
from src.model import create_model

import click
import joblib
import pandas as pd
from flask.cli import AppGroup

from src.encoder import Encoder

from .datagen import gen_csv
from .dataproc import create_sk_pipe, train_sk_pipe

data_cli = AppGroup("data")

DEFAULT_TRAINING_DATA = Path(__file__).parent / "tests" / "test_training_data.csv"
DEFAULT_PIPE = Path(__file__).parent / "tests" / "test_pipe.joblib"


@data_cli.command("gen-test")
@click.argument("path", default=DEFAULT_TRAINING_DATA)
@click.argument("nrows", type=int, default=1000)
def gen_test_data(path, nrows):
    click.echo(f"saving data to {path}")
    gen_csv(path, nrows)


@data_cli.command("gen-pipe")
@click.argument("dpath", default=DEFAULT_TRAINING_DATA)
@click.argument("path", default=DEFAULT_PIPE)
def gen_pipe(dpath, path):
    df = pd.read_csv(dpath)
    X = df.iloc[:, 1:-1]

    train_sk_pipe(path, X)


@data_cli.command("train-pipe")
@click.option("--dpath", default="data.csv")
@click.option("--path", default="pipeline/pipe.joblib")
def train_pipe(dpath, path):

    if dpath.endswith(".csv"):
        d = pd.read_csv(dpath)
    else:
        raise ValueError("data format is not supported")

    x = d.iloc[:, 1:-1]
    print("We have a dataframe of len=", len(x))
    train_sk_pipe(path, x)


@data_cli.command("train-model")
@click.option("--dpath", default="data.csv")
@click.option("--ppath", default="pipeline/pipe.joblib")
@click.option("--epoch", default=10)
@click.option("--version", prompt=True)
def train_model(dpath, ppath, epoch, version):
    if dpath.endswith(".csv"):
        d = pd.read_csv(dpath)
    else:
        raise ValueError("data format is not supported")

    pipe = joblib.load(ppath)
    encoder = Encoder(pipe)
    x = encoder.encode(d.iloc[:, 1:-1])

    m = create_model(
        [
            x.shape[1],
        ]
    )
    m.fit(x, d.iloc[:, -1], batch_size=1000, epochs=epoch)
    m.save(f"model/{version}")
