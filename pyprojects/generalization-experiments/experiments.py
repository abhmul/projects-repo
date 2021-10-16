import argparse
import numpy as np
import tensorflow as tf

from models import mlp
from data import load_mnist, flatten_data, mix_datasets, sample_train, mnist_mlp_connector, retrieve_split
from utils import model_extra_summary

EXPERIMENTS = {}


def experiment(exp_number, description):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Running experiment {exp_number}")
            print(description)
            return func(*args, **kwargs)

        EXPERIMENTS[exp_number] = wrapper

        return wrapper
    return decorator


@experiment(
   0, "TEST"
)
def experiment0(rng):
    return mlp_mnist_sgd_experiment(rng, 1000, 128, 3, "he_normal", 1e-3, 0, False, 20, 64)


def mlp_mnist_sgd_experiment(rng, sample_size, hidden_size, depth, initializer, learning_rate, momentum, nesterov, epochs, batch_size):
    X, Y = mix_datasets(*mnist_mlp_connector(load_mnist()))
    x_train, y_train, inds = sample_train((X, Y), sample_size, rng)
    assert len(x_train) == len(y_train)
    print(f"Sampled {len(x_train)} datapoints iid")

    model = mlp(Y.shape[1], depth=depth, hidden=hidden_size, initializer=initializer)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy", dtype=None)],
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None
    )

    model_extra_summary(model)

    print(f"Training model for {epochs} epochs and with {batch_size} batch size.")
    model.fit(
        x=x_train, y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )
    # DEBUG
    model.summary()


    # measure generalization error
    train_results = model.evaluate(
        x=x_train,
        y=y_train,
        batch_size=None,
        verbose=0,
        sample_weight=None,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        return_dict=True
    )
    expected_results = model.evaluate(
        x=X,
        y=Y,
        batch_size=None,
        verbose=0,
        sample_weight=None,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        return_dict=True
    )
    
    (Xtr_uniq, Ytr_uniq), (Xtest, Ytest) = retrieve_split((X, Y), inds)
    train_unique_results = model.evaluate(
        x=Xtr_uniq,
        y=Ytr_uniq,
        batch_size=None,
        verbose=0,
        sample_weight=None,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        return_dict=True
    )


    train_risk = 1 - train_results["accuracy"]
    expected_risk = 1 - expected_results["accuracy"]
    train_unique_risk = 1 - train_unique_results["accuracy"]
    test_risk = 1. / len(Ytest) * (len(Y) * expected_risk - len(Ytr_uniq) * train_unique_risk)
    generalization = expected_risk - train_risk

    return {
        "train_risk": train_risk,
        "expected_risk": expected_risk,
        "generalization": generalization,
        "test_risk": test_risk,
        "train_unique_risk": train_unique_risk
    }


@experiment(
    1,
    """
    MNIST 1k
    MLP 1 32 HeNormal
    Vanilla SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment1(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=32, depth=1, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    2,
    """
    MNIST 1k
    MLP 2 32 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment2(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=32, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    3,
    """
    MNIST 1k
    MLP 3 32 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment3(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=32, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    4,
    """
    MNIST 1k
    MLP 4 32 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment4(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=32, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    5,
    """
    MNIST 1k
    MLP 5 32 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment5(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=32, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# Same as experiment 1
# @experiment(
#     6,
#     """
#     MNIST 1k
#     MLP 1 64 HeNormal
#     Vanilla SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )


@experiment(
    7,
    """
    MNIST 1k
    MLP 2 64 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment7(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=64, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    8,
    """
    MNIST 1k
    MLP 3 64 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment8(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=64, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    9,
    """
    MNIST 1k
    MLP 4 64 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment9(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=64, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    10,
    """
    MNIST 1k
    MLP 5 64 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment10(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=64, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# Same as experiment 1
# @experiment(
#     11,
#     """
#     MNIST 1k
#     MLP 1 128 HeNormal
#     Vanilla SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )

@experiment(
    12,
    """
    MNIST 1k
    MLP 2 128 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment12(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=128, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    13,
    """
    MNIST 1k
    MLP 3 128 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment13(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=128, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    14,
    """
    MNIST 1k
    MLP 4 128 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment14(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=128, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    15,
    """
    MNIST 1k
    MLP 5 128 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment15(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=128, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    16,
    """
    MNIST 1k
    MLP 1 256 HeNormal
    Vanilla SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment16(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=256, depth=1, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    17,
    """
    MNIST 1k
    MLP 2 256 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment17(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=256, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    18,
    """
    MNIST 1k
    MLP 3 256 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment18(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=256, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    19,
    """
    MNIST 1k
    MLP 4 256 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment19(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=256, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    20,
    """
    MNIST 1k
    MLP 5 256 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment20(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=256, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# Same as experiment 1
# @experiment(
#     21,
#     """
#     MNIST 1k
#     MLP 1 512 HeNormal
#     Vanilla SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )

@experiment(
    22,
    """
    MNIST 1k
    MLP 2 512 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment22(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=512, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    23,
    """
    MNIST 1k
    MLP 3 512 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment23(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=512, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    24,
    """
    MNIST 1k
    MLP 4 512 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment24(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=512, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    25,
    """
    MNIST 1k
    MLP 5 512 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment25(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=512, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# Same as experiment 1
# @experiment(
#     26,
#     """
#     MNIST 1k
#     MLP 1 1024 HeNormal
#     Vanilla SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )


@experiment(
    27,
    """
    MNIST 1k
    MLP 2 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment27(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=1024, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    28,
    """
    MNIST 1k
    MLP 3 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment28(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=1024, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    29,
    """
    MNIST 1k
    MLP 4 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment29(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=1024, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    30,
    """
    MNIST 1k
    MLP 5 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment30(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=1024, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    31,
    """
    MNIST 5k
    MLP 1 N/A HeNormal
    Vanilla SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment31(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=5000, hidden_size=1, depth=1, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    32,
    """
    MNIST 5k
    MLP 2 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment32(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=5000, hidden_size=1024, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    33,
    """
    MNIST 5k
    MLP 3 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment33(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=5000, hidden_size=1024, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    34,
    """
    MNIST 5k
    MLP 4 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment34(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=5000, hidden_size=1024, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    35,
    """
    MNIST 5k
    MLP 5 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment35(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=5000, hidden_size=1024, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    36,
    """
    MNIST 10k
    MLP 1 NA HeNormal
    Vanilla SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment36(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=10000, hidden_size=1, depth=1, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    37,
    """
    MNIST 10k
    MLP 2 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment37(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=10000, hidden_size=1024, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    38,
    """
    MNIST 10k
    MLP 3 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment38(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=10000, hidden_size=1024, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    39,
    """
    MNIST 10k
    MLP 4 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment39(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=10000, hidden_size=1024, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    40,
    """
    MNIST 10k
    MLP 5 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment40(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=10000, hidden_size=1024, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    41,
    """
    MNIST 5k
    MLP 1 N/A HeNormal
    Vanilla SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment41(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=15000, hidden_size=1, depth=1, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    42,
    """
    MNIST 5k
    MLP 2 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment42(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=15000, hidden_size=1024, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    43,
    """
    MNIST 5k
    MLP 3 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment43(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=15000, hidden_size=1024, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    44,
    """
    MNIST 5k
    MLP 4 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment44(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=15000, hidden_size=1024, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    45,
    """
    MNIST 5k
    MLP 5 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment45(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=15000, hidden_size=1024, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    46,
    """
    MNIST 20k
    MLP 1 NA HeNormal
    Vanilla SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment46(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=20000, hidden_size=1, depth=1, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    47,
    """
    MNIST 20k
    MLP 2 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment47(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=20000, hidden_size=1024, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


@experiment(
    48,
    """
    MNIST 20k
    MLP 3 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment48(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=20000, hidden_size=1024, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    49,
    """
    MNIST 20k
    MLP 4 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment49(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=20000, hidden_size=1024, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

@experiment(
    50,
    """
    MNIST 20k
    MLP 5 1024 HeNormal
    SGD 1e3
    CCE
    ep 30
    batch_size 32
    """
)
def experiment50(rng):
    return mlp_mnist_sgd_experiment(rng, sample_size=20000, hidden_size=1024, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)