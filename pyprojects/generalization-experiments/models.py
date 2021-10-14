from tensorflow import keras
from tensorflow.keras import layers


INITIALIZERS = {
    "he_normal": keras.initializers.HeNormal()
}


def mlp(outputs, depth, hidden, initializer):
    model = keras.Sequential(
        [
            layers.Dense(
                hidden,
                activation="relu",
                name=f"layer{l_ind}",
                kernel_initializer=INITIALIZERS[initializer],
                bias_initializer="zeros"
            ) for l_ind in range(depth - 1)
        ] +
        [
            layers.Dense(
                outputs,
                activation="linear",
                name='output_layer',
                kernel_initializer=INITIALIZERS[initializer],
                bias_initializer="zeros"
            )
        ]
    )

    return model
