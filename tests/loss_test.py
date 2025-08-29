# test_masked_loss.py

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import serialize_keras_object, deserialize_keras_object

# Import masked_loss from your module
from src.loss import masked_loss  # adjust filename accordingly


@pytest.fixture
def sample_data():
    # Batch of 3 samples, 4 classes
    y_true = np.array([1, 0, 2], dtype=np.int32)  # second entry masked
    y_pred = np.array([
        [2.0, 1.0, 0.1, -1.0],
        [0.0, 0.5, 1.5, -0.5],
        [1.0, -0.5, 2.5, 0.0],
    ], dtype=np.float32)
    return y_true, y_pred


def test_masked_loss_computation(sample_data):
    y_true, y_pred = sample_data

    # Compute loss
    loss = masked_loss(y_true, y_pred).numpy()

    # Manual expected loss (only entries with y_true != 0 contribute)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    raw_loss = loss_fn(y_true, y_pred).numpy()

    mask = (y_true != 0).astype(np.float32)
    expected = np.sum(raw_loss * mask) / np.sum(mask)

    assert np.isclose(loss, expected, atol=1e-6)


def test_all_masked_out():
    # All y_true == 0 → denominator = 0
    y_true = np.zeros((3,), dtype=np.int32)
    y_pred = np.random.randn(3, 4).astype(np.float32)

    loss = masked_loss(y_true, y_pred)

    # Should result in NaN or inf (division by zero)
    assert tf.math.is_nan(loss) or tf.math.is_inf(loss)


def test_no_masking():
    # No zeros in y_true → identical to normal loss
    y_true = np.array([1, 2, 3], dtype=np.int32)
    y_pred = np.random.randn(3, 4).astype(np.float32)

    masked_val = masked_loss(y_true, y_pred).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum")
    normal_val = loss_fn(y_true, y_pred).numpy() / len(y_true)

    assert np.isclose(masked_val, normal_val, atol=1e-6)



def test_serialization_roundtrip():
    # Ensure @tf.keras.utils.register_keras_serializable works
    cfg = serialize_keras_object(masked_loss)
    loaded = deserialize_keras_object(cfg)

    # Check callable equivalence
    y_true = np.array([1], dtype=np.int32)
    y_pred = np.array([[0.1, 0.9]], dtype=np.float32)

    assert np.isclose(
        masked_loss(y_true, y_pred).numpy(),
        loaded(y_true, y_pred).numpy(),
        atol=1e-6
    )


def test_in_keras_model(tmp_path):
    # Verify masked_loss can be used inside a compiled model
    model = keras.Sequential([
        keras.layers.Dense(4, activation=None)
    ])
    model.compile(optimizer="adam", loss=masked_loss)

    x = np.random.randn(5, 3).astype(np.float32)
    y = np.array([1, 2, 0, 3, 1], dtype=np.int32)  # some masked

    history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)

    assert "loss" in history.history
