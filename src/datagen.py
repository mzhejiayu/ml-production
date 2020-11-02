from typing import Tuple
import pandas as pd
from pathlib import Path
from faker import Faker
import random


def extract_geo_info(tuple_record) -> Tuple[str, str, str]:
    """This function extracts the information from the faker generated data
    Args:
        tuple_record ([type]): [description]

    Returns:
        [type]: [description]
    """
    city = tuple_record[2]
    continent = tuple_record[4].split("/")[0]
    country = tuple_record[3]
    return city, country, continent


def gen_data(nrows: int, num_location=100, num_isps=60):
    """Generator function

    Args:
        nrows (int): [description]
        num_location (int, optional): [description]. Defaults to 100.
        num_isps (int, optional): [description]. Defaults to 60.

    Yields:
        [type]: [description]
    """
    fake = Faker()

    # ListOfLocations
    lofl = [fake.location_on_land() for i in range(num_location)]
    # ListOfIsps
    lofi = [fake.company() for i in range(num_isps)]

    for _ in range(nrows):
        l1 = lofl[random.randint(0, num_location - 1)]
        l2 = lofl[random.randint(0, num_location - 1)]

        i1 = lofi[random.randint(0, num_isps - 1)]
        i2 = lofi[random.randint(0, num_isps - 1)]
        perf = random.random() * 100

        city1, country1, continent1 = extract_geo_info(l1)
        city2, country2, continent2 = extract_geo_info(l2)

        yield dict(
            {},
            city1=city1,
            city2=city2,
            isp1=i1,
            isp2=i2,
            country1=country1,
            country2=country2,
            continent1=continent1,
            continent2=continent2,
            target=perf,
        )


def gen_csv(file_path: str, nrows=100, **kwargs):
    df = pd.DataFrame([i for i in gen_data(nrows, **kwargs)])
    df.to_csv(file_path)
