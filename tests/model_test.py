# test_titans.py

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import the class
from src.model import Titans  # adjust to your actual module path


@pytest.fixture
def tiny_model():
    """Small Titans instance for quick tests."""
    return Titans(
        embedding_dim=8,
        sequence_length=6,
        num_heads=2,
        dff=16,
        vocab_size=20,
        rate=0.1
    )


def test_build_and_call_shapes(tiny_model):
    batch_size = 4
    x = np.random.randint(0, 20, size=(batch_size, 6))

    y = tiny_model(x)

    # Should produce logits over vocab
    assert isinstance(y, tf.Tensor)
    assert y.shape == (batch_size, 6, 20)


def test_causal_mask_shape(tiny_model):
    seq_len = tiny_model.sequence_length
    mask = tiny_model.create_causal_mask(seq_len)

    assert mask.shape == (seq_len, seq_len)
    # Ensure it's lower-triangular
    assert np.allclose(mask.numpy(), np.tril(np.ones((seq_len, seq_len))))


def test_combine_masks(tiny_model):
    batch_size, seq_len = 2, tiny_model.sequence_length
    pad_mask = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0]], dtype=np.int32)

    combined = tiny_model.combine_masks(tf.constant(pad_mask), seq_len)

    assert combined.shape == (batch_size, 1, seq_len, seq_len)
    assert set(np.unique(combined.numpy())).issubset({0.0, 1.0})




def test_training_step(tiny_model):
    model = tiny_model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    x = np.random.randint(0, 20, size=(8, 6))
    y = np.random.randint(0, 20, size=(8, 6))

    history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    assert "loss" in history.history
    assert len(history.history["loss"]) == 1


def test_get_config_and_from_config(tiny_model):
    config = tiny_model.get_config()
    clone = Titans.from_config(config)

    assert isinstance(config, dict)
    assert isinstance(clone, Titans)
    assert clone.embedding_dim == tiny_model.embedding_dim
    assert clone.vocab_size == tiny_model.vocab_size
