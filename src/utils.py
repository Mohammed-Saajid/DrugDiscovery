import tensorflow as tf
import selfies as sf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from typing import Generator,Tuple, List
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize_selfies(smiles: str) -> list:
    """Tokenizes a SMILES string into SELFIES tokens.
    Args:
        smiles (str): A SMILES string.
    Returns:
        list: A list of SELFIES tokens.
    """

    try:
        selfies = sf.encoder(smiles)
    except Exception:
        selfies = ""   # skip invalid ones
    return list(sf.split_selfies(selfies))


def build_vocab_from_dataframe(df: pd.DataFrame) -> list:
    """
    Builds a vocabulary of SELFIES tokens directly from a DataFrame of SMILES.
    Adds <start>, <end>, and <UNK> special tokens.
    Args:
        df (pd.DataFrame): A DataFrame containing SMILES strings.
    Returns:
        list: A list of unique SELFIES tokens.
    """
    vocab = {"<start>", "<end>"}  # start with special tokens

    for smiles in df['smiles']:
        tokens = tokenize_selfies(smiles)
        vocab.update(tokens)

    return sorted(list(vocab))


def tokenizer_initialize_from_dataframe(df: pd.DataFrame) -> Tuple[tf.keras.preprocessing.text.Tokenizer, int]:
    """
    Initializes a Keras tokenizer using vocab built from DataFrame.
    Args:
        df (pd.DataFrame): A DataFrame containing SMILES strings.
    Returns:
        Tuple: A tuple containing the tokenizer and vocabulary size.
    """
    all_tokens = build_vocab_from_dataframe(df)

    tokenizer = Tokenizer(oov_token="<UNK>", filters='', lower=False)
    tokenizer.fit_on_texts(all_tokens)

    vocab_size = len(tokenizer.word_index) + 1 # +1 for padding (index 0)
    return tokenizer, vocab_size


def sequence_generator(df: pd.DataFrame, tokenizer, max_seq_length: int, seq_padding: int = 1) -> Generator[Tuple[List[int], List[int]], None, None]:
    """Generates sequences of token IDs from SELFIES with post-padding.
    Args:
        df (pd.DataFrame): A DataFrame containing SMILES strings.
        tokenizer: A Keras tokenizer fitted on SELFIES tokens.
        max_seq_length (int): The maximum sequence length.
        seq_padding (int): The amount of padding to apply.
    Yields:
        tuple: A tuple containing the input and target sequences.
    """
    for smiles in df['smiles']:
        tokens = ["<start>"]
        selfies = tokenize_selfies(smiles)
        tokens.extend(selfies)
        tokens.append("<end>")

        token_ids = tokenizer.texts_to_sequences([tokens])[0]

        target_len = max_seq_length + seq_padding

        # Post-pad sequences
        if len(token_ids) < target_len:
            token_ids = token_ids + [0] * (target_len - len(token_ids))
        elif len(token_ids) > target_len:
            token_ids = token_ids[-target_len:]  # truncate from the left if too long

        x = token_ids[:-1]
        y = token_ids[1:]
        yield x, y


def create_selfies_dataset(sample_df: pd.DataFrame, max_seq_length: int, batch_size: int = 256, buffer_size: int = 10000, seq_padding: int = 1) -> Tuple[tf.data.Dataset, tf.keras.preprocessing.text.Tokenizer, int, int]:
    """
    Creates a TensorFlow dataset from SMILES strings using SELFIES encoding.
    Args:
        sample_df (pd.DataFrame): A DataFrame containing SMILES strings.
        max_seq_length (int): The maximum sequence length.
        batch_size (int): The batch size for the dataset.
        buffer_size (int): The buffer size for shuffling.
        seq_padding (int): The amount of padding to apply.
    Returns:
        tuple: A tuple containing the dataset, tokenizer, vocabulary size, and maximum sequence length.
    """
    tokenizer, vocab_size = tokenizer_initialize_from_dataframe(sample_df)

    output_signature = (
        tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32),  # Input sequence
        tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32)   # Target sequence
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(sample_df, tokenizer, max_seq_length),
        output_signature=output_signature
    )

    dataset = dataset.shuffle(buffer_size).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, tokenizer, vocab_size, max_seq_length


def dataset_gen(dataframe: pd.DataFrame, tokenizer, max_seq_length: int, vocab_size: int, batch_size: int = 128, buffer_size: int = 10000, seq_padding: int = 1) -> tf.data.Dataset:
    """Generates a TensorFlow dataset from a DataFrame using a given tokenizer.
    Args:
        dataframe (pd.DataFrame): A DataFrame containing SMILES strings.
        tokenizer: A Keras tokenizer fitted on SELFIES tokens.
        max_seq_length (int): The maximum sequence length.
        vocab_size (int): The vocabulary size.
        batch_size (int): The batch size for the dataset.
        buffer_size (int): The buffer size for shuffling.
        seq_padding (int): The amount of padding to apply.
    Returns:
        tf.data.Dataset: A TensorFlow dataset.
    """
    output_signature = (
        tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(dataframe, tokenizer, max_seq_length),
        output_signature=output_signature
    )

    dataset = dataset.shuffle(buffer_size).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def top_p_logits_batch(logits: tf.Tensor, top_p: float = 1.0) -> tf.Tensor:
    """
    Apply top-p (nucleus) filtering to a batch of logits.
    Args:
        logits: [batch, vocab]
        top_p: cumulative probability threshold
    Returns:
        [batch, vocab]: Filtered logits.
    """
    if top_p >= 1.0:
        return logits

    new_logits = []
    for logit in logits:  # loop over batch
        sorted_indices = tf.argsort(logit, direction='DESCENDING')
        sorted_logits = tf.gather(logit, sorted_indices)
        sorted_probs = tf.nn.softmax(sorted_logits)
        cumulative_probs = tf.cumsum(sorted_probs)

        # Mask tokens outside top_p
        mask = cumulative_probs > top_p
        mask = tf.concat([[False], mask[:-1]], axis=0)  # keep first above top_p
        # Set masked logits to -inf
        sorted_logits = tf.where(mask, tf.fill(tf.shape(sorted_logits), float('-inf')), sorted_logits)
        # Scatter back to original order
        new_logit = tf.scatter_nd(
            indices=tf.expand_dims(sorted_indices, axis=-1),
            updates=sorted_logits,
            shape=tf.shape(logit, out_type=tf.int32)
        )
        new_logits.append(new_logit)

    return tf.stack(new_logits, axis=0)


def sample_from_logits(logits: tf.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> tf.Tensor:
    """
    Sample token IDs from logits with temperature scaling, top-k, and top-p filtering.
    Args:
        logits: [batch, vocab]
        temperature: scaling factor
        top_k: number of top tokens to consider
        top_p: cumulative probability threshold
    Returns:
        [batch]: sampled token IDs
    """


    # Scale by temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        values, _ = tf.math.top_k(logits, k=top_k)
        min_values = values[:, -1, tf.newaxis]
        logits = tf.where(logits < min_values, tf.constant(-np.inf, dtype=logits.dtype), logits)
                          

    # Top-p
    if top_p < 1.0:
       logits = top_p_logits_batch(logits, top_p)


    # Convert to probabilities safely and sample
    sampled_ids = tf.random.categorical(logits, num_samples=1)[:, 0]
    return sampled_ids


def generate_drug_batch(
    seed_texts,
    model,
    tokenizer,
    max_length,
    next_words=30,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    end_token_id=None,
    min_length_before_eos=20,
):
    """
    Generate sequences from a model with autoregressive decoding.

    Args:
        seed_texts (list[str]): Starting strings for each sequence in the batch.
        model (tf.keras.Model): Trained model for generation.
        tokenizer: Tokenizer with `texts_to_sequences` & `index_word`.
        max_length (int): Maximum sequence length.
        next_words (int): Maximum number of tokens to generate.
        temperature (float): Softmax temperature scaling.
        top_k (int): Top-k sampling cutoff.
        top_p (float): Nucleus (top-p) sampling cutoff.
        end_token_id (int): ID of EOS token.
        min_length_before_eos (int): Minimum tokens before EOS allowed.

    Returns:
        list[str]: Generated sequences.
    """
    assert next_words <= max_length, "next_words must be <= max_length"

    batch_size = len(seed_texts)

    # Convert seeds to padded token sequences
    token_lists = tokenizer.texts_to_sequences(seed_texts)
    token_lists = pad_sequences(token_lists, maxlen=max_length, padding="post")

    finished = [False] * batch_size
    step = 0

    while step < next_words:
        # Forward pass -> logits for next token
        predicted_logits = model.predict(token_lists, verbose=0)  # [B, T, V]
        logits = predicted_logits[:, -1, :]  # take last step logits [B, V]

        # Mask out banned tokens
        banned_tokens = [
            tokenizer.word_index.get("<UNK>"),
            tokenizer.word_index.get("<start>"),
        ]
        if end_token_id is not None and step < min_length_before_eos:
            banned_tokens.append(end_token_id)

        banned_tokens = [t for t in banned_tokens if t is not None]

        if banned_tokens:
            vocab_size = logits.shape[-1]
            mask = tf.zeros(vocab_size, dtype=logits.dtype)
            updates = tf.constant([-float("inf")] * len(banned_tokens), dtype=logits.dtype)
            mask = tf.tensor_scatter_nd_update(
                mask,
                indices=[[tid] for tid in banned_tokens],
                updates=updates,
            )
            logits = logits + mask  # broadcast to [B, V]

      
        sampled_ids = sample_from_logits(logits, temperature, top_k, top_p)

        for i, token_id in enumerate(sampled_ids.numpy()):
            if finished[i]:
                continue

            if end_token_id is not None and token_id == end_token_id:
                finished[i] = True
                continue

            word = tokenizer.index_word.get(token_id, None)
            if word and word not in ("<UNK>", "<start>"):
                seed_texts[i] += " " + word

        # Roll sequence window forward
        token_lists = tf.concat(
            [token_lists[:, 1:], tf.expand_dims(sampled_ids, axis=-1)], axis=1
        )

        if all(finished):
            break

        step += 1

    return seed_texts
