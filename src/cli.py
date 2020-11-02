from flask.cli import AppGroup
import click
from .datagen import gen_csv
from pathlib import Path

# add the command line tools
data_cli = AppGroup("data")


@data_cli.command("gen-test")
@click.argument(
    "path", default=Path(__file__).parent / "tests" / "test_training_data.csv"
)
@click.argument("nrows", type=int, default=1000)
def gen_test_data(path, nrows):
    click.echo(f"saving data to {path}")
    gen_csv(path, nrows)
