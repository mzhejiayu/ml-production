import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

__doc__ = """
This file contains the model training
"""


def create_model(input_shape, **kwargs):
    model = Sequential()
    model.add(Dense(1, input_shape=input_shape))
    model.compile(loss="mse", metrics=["mse"])
    return model


def train_model(x_train: pd.DataFrame, y_train: pd.Series, **kwargs):
    ...


def save_model(mpath: str, **kwargs):
    ...
