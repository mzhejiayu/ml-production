import tensorflow as tf
from tensorflow.keras import layers

__default_config = [
    ("c1", "city1", 20),
    ("c2", "city2", 20),
    ("co1", "country1", 20),
    ("co2", "country2", 20),
    ("t1", "continent1", 6),
    ("t2", "continent2", 6),
    ("i1", "isp1", 20),
    ("i2", "isp2", 20),
]


def build_model(uniq_geo_token):
    # unknown level or empty level
    max_token = len(uniq_geo_token) + 2
    lookup_layer = layers.experimental.preprocessing.StringLookup(
        max_token, vocabulary=uniq_geo_token
    )
    one_hot_layer = layers.experimental.preprocessing.CategoryEncoding(max_token)
    dense_layer = layers.Dense(1, activation="linear")

    geo_input = tf.keras.Input(shape=(8,), dtype=tf.string, name="g")
    sames_input = tf.keras.Input(shape=(4,), name="s")

    geo_output = one_hot_layer(lookup_layer(geo_input))

    x = layers.concatenate([geo_output, sames_input])
    prediction = dense_layer(x)

    model = tf.keras.Model(inputs=[geo_input, sames_input], outputs=[prediction])
    model.compile(loss="mse")

    return model
