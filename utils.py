import selfies as sf
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer,pad_sequences


def tokenize_selfies(smiles:str):
        """Tokenizes a SMILES string into SELFIES tokens.
        Args:
            smiles (str): SMILES string to tokenize.
        Returns:
            list: List of SELFIES tokens.
        """
        selfies = sf.encoder(smiles)
        return list(sf.split_selfies(selfies))

def get_vocab(vocab_file:str):
    """Reads a vocabulary file and returns a list of tokens.
    Args:
        vocab_file (str): Path to the vocabulary file.
    Returns:
        list: List of tokens from the vocabulary file.
    """
     
    all_tokens = ["<start>","<end>"]
    with open(vocab_file,"r") as fp:
        file = fp.read()
        file = file.split('\n')
        all_tokens.extend(file)
    return all_tokens


def tokenizer_initialize(all_tokens:list):
        """Initializes a Keras Tokenizer and returns it along with the vocabulary size.
        Args:
            all_tokens (list): List of tokens to initialize the tokenizer with.
        Returns:
            Tokenizer: Keras Tokenizer instance.
            int: Vocabulary size.
        """
        tokenizer = Tokenizer(oov_token="<UNK>", filters='')
        tokenizer.fit_on_texts(all_tokens)
        vocab_size = len(tokenizer.word_index) + 1
        return tokenizer,vocab_size

def sequence_generator(df, tokenizer,max_seq_length,seq_padding=1):
        """Generates sequences of token IDs from SELFIES.
        Args:
            df (pd.DataFrame): DataFrame containing a 'smiles' column with SMILES strings.
            tokenizer (Tokenizer): Keras Tokenizer instance.
            max_seq_length (int): Maximum sequence length for padding.
            seq_padding (int): Padding length to include the target.
        Yields:
            tuple: Tuple of input and target sequences (x, y).
        """
        for smiles in df['smiles']:
            tokens = ["<start>"]
            selfies = tokenize_selfies(smiles)
            tokens.extend(selfies)
            tokens.append("<end>")
            token_ids = tokenizer.texts_to_sequences([tokens])[0]
            if len(token_ids) < max_seq_length + seq_padding:  # +1 to have enough for x and y
                token_ids = [0] * (max_seq_length + seq_padding - len(token_ids)) + token_ids
            elif len(token_ids) > max_seq_length + seq_padding:
                token_ids = token_ids[-(max_seq_length + seq_padding):]
            x = token_ids[:-1]
            y = token_ids[1:]
            yield x, y


def create_selfies_dataset(df,max_seq_length,all_tokens, batch_size=256, buffer_size=10000, seq_padding=1):
    """
    Creates a TensorFlow dataset from SMILES strings using SELFIES encoding.

    Args:
        df (pd.DataFrame): DataFrame containing a 'smiles' column with SMILES strings.
        max_seq_length (int): Maximum sequence length for padding.
        all_tokens (list): List of tokens to initialize the tokenizer with.
        batch_size (int): Batch size for the dataset.
        buffer_size (int): Buffer size for shuffling the dataset.
        seq_padding (int): Padding length to include the target.
    Returns:
        tf.data.Dataset: TensorFlow dataset containing input and target sequences.
        Tokenizer: Keras Tokenizer instance.
        int: Vocabulary size.
        int: Maximum sequence length.
    """

    tokenizer,vocab_size = tokenizer_initialize(all_tokens)


     
    output_signature = (
        tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32),  # Input sequence
        tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32)   # Target sequence
    )


    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(df, tokenizer,max_seq_length),
        output_signature=output_signature
    )

    # Shuffle, batch, and prefetch the dataset
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, tokenizer, vocab_size, max_seq_length


def dataset_gen(dataframe,tokenizer,max_seq_length,batch_size=128,buffer_size=10000):
    """
    Generates a TensorFlow dataset from SMILES strings using SELFIES encoding.
    Args:
        dataframe (pd.DataFrame): DataFrame containing a 'smiles' column with SMILES strings.
        tokenizer (Tokenizer): Keras Tokenizer instance.
        max_seq_length (int): Maximum sequence length for padding.
        batch_size (int): Batch size for the dataset.
        buffer_size (int): Buffer size for shuffling the dataset.
    returns:
        tf.data.Dataset: TensorFlow dataset containing input and target sequences.
    """

    output_signature = (
        tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32),  # Input sequence
        tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32)   # Target sequence
    )


    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(dataframe, tokenizer,max_seq_length),
        output_signature=output_signature
    )

    # Shuffle, batch, and prefetch the dataset
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


    
def sample_from_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """"
    Samples from the logits using temperature scaling and top-k/top-p filtering.
    Args:
        logits (tf.Tensor): Logits from the model.
        temperature (float): Temperature for scaling logits.
        top_k (int): Number of top logits to keep (0 for no filtering).
        top_p (float): Cumulative probability threshold for nucleus sampling.
    Returns:
        tf.Tensor: Sampled indices from the logits.
    """
    # Temperature scaling
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        values, _ = tf.math.top_k(logits, k=top_k)
        min_values = values[:, -1, tf.newaxis]
        logits = tf.where(logits < min_values, tf.fill(tf.shape(logits), float('-inf')), logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
        sorted_indices = tf.argsort(logits, direction='DESCENDING', axis=-1)
        mask = cumulative_probs > top_p

        mask_shifted = tf.concat([tf.zeros_like(mask[:, :1]), mask[:, :-1]], axis=-1)
        indices_to_remove = tf.gather(mask_shifted, sorted_indices, batch_dims=1)
        logits = tf.where(indices_to_remove, float('-inf'), logits)

    # Sample from the filtered distribution
    probs = tf.nn.softmax(logits)
    return tf.random.categorical(tf.math.log(probs), num_samples=1)

# Generate next word predictions
def generate_drug(seed_text, model, tokenizer, max_length,end_token_id, next_words=30):
    """
    Generates a sequence of words based on the input seed text using the trained model.
    Args:
        seed_text (str): Input seed text to start the generation.
        model (tf.keras.Model): Trained model for text generation.
        tokenizer (Tokenizer): Keras Tokenizer instance.
        max_length (int): Maximum sequence length for padding.
        end_token_id (int): Token ID for the end token.
        next_words (int): Number of words to generate.
    Returns:
        str: Generated text sequence.
    """
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length, padding='pre')
        predicted_logits = model.predict(token_list, verbose=0)  # Shape: (1, max_length, total_words)
        logits = predicted_logits[:, -1, :]  # Extract last time step probabilities
        predicted_index = sample_from_logits(logits, temperature=0.9, top_k=50, top_p=0.95)
        predicted_index = predicted_index.numpy()[0][0]  # Convert tensor to integer
        predicted_word = tokenizer.index_word.get(predicted_index, "<UNK>")        
        seed_text += " " + predicted_word  # Append predicted word to input text
        if predicted_index==end_token_id:
            break
    return seed_text
