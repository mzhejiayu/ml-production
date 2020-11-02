from flask.cli import AppGroup
from .dataproc import train_sk_pipe
import pandas as pd
import click
from .datagen import gen_csv
from pathlib import Path

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