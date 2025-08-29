# test_positionwise_feedforward.py

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Import the class from your module
from src.pos_ff import PositionwiseFeedforward  # adjust filename


@pytest.fixture
def layer():
    return PositionwiseFeedforward(embedding_dim=16, dff=64)


def test_output_shape(layer):
    # Input: batch=2, seq_len=5, embed_dim=16
    x = tf.random.uniform((2, 5, 16))
    y = layer(x)
    # Output should have same embedding_dim as input
    assert y.shape == (2, 5, 16)


def test_values_change(layer):
    x = tf.ones((1, 3, 16))
    y1 = layer(x)
    y2 = layer(x)
    # Forward pass should produce deterministic outputs (same weights, same input)
    np.testing.assert_allclose(y1.numpy(), y2.numpy(), rtol=1e-6)


def test_trainable_variables(layer):
    x = tf.random.uniform((2, 5, 16))
    y = layer(x)
    assert layer.trainable_variables  # layer should have trainable params
    assert all(v.shape[0] > 0 for v in layer.trainable_variables)


def test_serialization_roundtrip(layer):
    config = layer.get_config()
    cloned = PositionwiseFeedforward.from_config(config)

    x = tf.random.uniform((1, 4, 16))
    y1 = layer(x)
    y2 = cloned(x)

    # Outputs differ since weights not cloned, but shape must be identical
    assert y1.shape == y2.shape
    assert cloned.embedding_dim == layer.embedding_dim
    assert cloned.dff == layer.dff


def test_in_keras_model(tmp_path):
    seq_len, embed_dim = 5, 16
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_len, embed_dim)),
        PositionwiseFeedforward(embedding_dim=embed_dim, dff=32),
    ])
    model.compile(optimizer="adam", loss="mse")

    x = np.random.randn(10, seq_len, embed_dim).astype(np.float32)
    y = np.random.randn(10, seq_len, embed_dim).astype(np.float32)

    history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)
    assert "loss" in history.history
