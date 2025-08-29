# tests/test_selfies_utils.py

import pytest
import pandas as pd
import numpy as np
import tensorflow as tf

from src.utils import (
    tokenize_selfies,
    build_vocab_from_dataframe,
    tokenizer_initialize_from_dataframe,
    sequence_generator,
    create_selfies_dataset,
    dataset_gen,
)


def test_tokenize_selfies_valid_and_invalid():
    # Valid SMILES
    tokens = tokenize_selfies("CCO")  # ethanol
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    assert len(tokens) > 0

    # Invalid SMILES returns empty list
    tokens_invalid = tokenize_selfies("INVALID!!")
    assert tokens_invalid == []


def test_build_vocab_from_dataframe():
    df = pd.DataFrame({"smiles": ["CCO", "CCN"]})
    vocab = build_vocab_from_dataframe(df)

    assert "<start>" in vocab
    assert "<end>" in vocab
    assert isinstance(vocab, list)
    assert len(vocab) > 2
    assert all(isinstance(tok, str) for tok in vocab)


def test_tokenizer_initialize_from_dataframe():
    df = pd.DataFrame({"smiles": ["CCO", "CCN"]})
    tokenizer, vocab_size = tokenizer_initialize_from_dataframe(df)

    assert isinstance(tokenizer, tf.keras.preprocessing.text.Tokenizer)
    assert isinstance(vocab_size, int)
    assert vocab_size > 0

    # tokenizer must recognize <start> and <end>
    assert "<start>" in tokenizer.word_index
    assert "<end>" in tokenizer.word_index


def test_sequence_generator_shapes():
    df = pd.DataFrame({"smiles": ["CCO"]})
    tokenizer, vocab_size = tokenizer_initialize_from_dataframe(df)

    max_len = 5
    gen = sequence_generator(df, tokenizer, max_seq_length=max_len, seq_padding=1)
    x, y = next(gen)

    # Should be length max_len
    assert len(x) == max_len
    assert len(y) == max_len
    # Check shifting property
    assert y[:-1] == x[1:]


def test_create_selfies_dataset_and_dataset_gen():
    df = pd.DataFrame({"smiles": ["CCO", "CCN"]})
    max_len = 6
    batch_size = 2

    dataset, tokenizer, vocab_size, max_seq_length = create_selfies_dataset(
        df, max_seq_length=max_len, batch_size=batch_size, buffer_size=10
    )

    assert isinstance(dataset, tf.data.Dataset)
    assert isinstance(tokenizer, tf.keras.preprocessing.text.Tokenizer)
    assert vocab_size > 0
    assert max_seq_length == max_len

    # Get one batch
    x_batch, y_batch = next(iter(dataset))
    assert x_batch.shape[1] == max_len
    assert y_batch.shape[1] == max_len

    # Dataset_gen with external tokenizer
    dataset2 = dataset_gen(df, tokenizer, max_seq_length=max_len, vocab_size=vocab_size, batch_size=batch_size)
    x2, y2 = next(iter(dataset2))
    assert x2.shape[1] == max_len
    assert y2.shape[1] == max_len
