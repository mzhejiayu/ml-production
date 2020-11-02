import pytest
import os
from jsonschema import validate
from ..datagen import extract_geo_info, gen_data, gen_csv


def test_extract_geo_info():
    c, c1, c2 = extract_geo_info(
        ("34.95303", "-120.43572", "Santa Maria", "US", "America/Los_Angeles")
    )

    assert c == "Santa Maria"
    assert c1 == "US"
    assert c2 == "America"


def test_gen_data(schema):
    data = [i for i in gen_data(1, 1, 1)]
    assert len(data) == 1
    d = data[0]
    validate(d, schema)

    data = [i for i in gen_data(10, 10, 10)]
    assert len(data) == 10
    for d in data:
        validate(d, schema)


def test_gen_csv():
    gen_csv("test.csv")
    assert os.path.exists("test.csv")
    os.remove("test.csv")
