import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
from model import DrugDiscoveryModel
import argparse

if __name__=="__main__":

    # Get user input
    parser = argparse.ArgumentParser(description="Automate input parameters.")
    parser.add_argument("--embedding_dim", required=False, default=128,help="JEmbedding Dimension")
    parser.add_argument("--num_heads", required=False, default=8,help="Number of heads for Multi Head Attention")
    parser.add_argument("--latent_dim", required=False, default=512,help="Latent Dimension")
    parser.add_argument("--max_seq_length", required=False, default=1600, help="Maximum Sequence Length")
    parser.add_argument("--df_path",required=False,default="data.csv",help="Dataset Containing Molecules")
    parser.add_argument("--vocab_path",required=False,default="vocab.txt",help="Vocabulary text")
    parser.add_argument("--batch_size",required=True)

    args = parser.parse_args()

    embedding_dim = int(args.embedding_dim)
    num_heads = int(args.num_heads)
    latent_dim = int(args.latent_dim)
    max_seq_length = int(args.max_seq_length)
    df_path = args.df_path
    vocab_file = args.vocab_file
    batch_size = args.batch_size

    # Define Loss, Metrics and Optimizer
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    optimizer = tf.keras.optimizers.Adam()

    # Load the dataset
    df = pd.read_csv(df_path)

    # First split into train+val and test
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # Then split train+val into train and val
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)

    # Load the vocabulary, tokenizer, and create the dataset
    dataset,tokenizer,vocab_size,max_seq_length = create_selfies_dataset(train_df,max_seq_length = max_seq_length,batch_size=batch_size)
    # Create validation and test datasets
    val_dataset = dataset_gen(val_df,max_seq_length=max_seq_length,batch_size=batch_size,tokenizer=tokenizer,vocab_size=vocab_size)
    test_dataset = dataset_gen(test_df,max_seq_length=max_seq_length,batch_size=batch_size,tokenizer=tokenizer,vocab_size=vocab_size)

    # Create the model
    model = DrugDiscoveryModel(embedding_dim=embedding_dim, sequence_length=max_seq_length, num_heads=num_heads, dff=latent_dim, total_words=vocab_size)
    model.build(input_shape=(None, max_seq_length))
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

    # Train the model
    csv_logger = tf.keras.callbacks.CSVLogger('training.log')
    history = model.fit(dataset,epochs=10,callbacks=[csv_logger],validation_data=val_dataset)

    # Save the model
    model.save("model.keras")