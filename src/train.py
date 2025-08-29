import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils import *
from .model import Titans
import argparse
import math
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from .loss import masked_loss

if __name__=="__main__":

    # Get user input
    parser = argparse.ArgumentParser(description="Automate input parameters.")
    parser.add_argument("--embedding_dim", required=False, default=128,help="Embedding Dimension", type=int)
    parser.add_argument("--num_heads", required=False, default=8,help="Number of heads for Multi Head Attention", type=int)
    parser.add_argument("--latent_dim", required=False, default=512,help="Latent Dimension", type=int)
    parser.add_argument("--max_seq_length", required=False, default=1600, help="Maximum Sequence Length", type=int)
    parser.add_argument("--df_path",required=False,default="data.csv",help="Dataset Containing Molecules")
    parser.add_argument("--batch_size",required=True,type=int,help="Batch Size")

    args = parser.parse_args()

    embedding_dim = int(args.embedding_dim)
    num_heads = int(args.num_heads)
    latent_dim = int(args.latent_dim)
    max_seq_length = int(args.max_seq_length)
    df_path = args.df_path
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
    model = Titans(embedding_dim=embedding_dim, sequence_length=max_seq_length, num_heads=num_heads, dff=latent_dim, vocab_size=vocab_size)
    # Build by calling with input shape
    inputs = tf.keras.Input(shape=(max_seq_length,))
    outputs = model(inputs)

    drug_discovery_model = tf.keras.Model(inputs, outputs, name="DrugDiscoveryModel")
    drug_discovery_model.compile(loss=masked_loss, optimizer=optimizer, metrics=metrics)
    print(drug_discovery_model.summary())

    

    STEPS_PER_EPOCH = math.ceil(len(train_df) / batch_size) # returns int 
    VAL_STEPS = math.ceil(len(val_df) / batch_size)

    checkpoint_path = "best_model.weights.h5"

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="sparse_categorical_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="sparse_categorical_accuracy",
            mode="max",
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        # Define LR scheduler
        ReduceLROnPlateau(
            monitor="val_loss",     # quantity to be monitored
            factor=0.5,             # new_lr = lr * factor
            patience=3,             # wait for 3 epochs before reducing LR
            min_lr=1e-6,            # lower bound on LR
            verbose=1               # print LR updates
        )

    ]

    history = drug_discovery_model.fit(
        dataset,
        epochs=1000,
        validation_data=val_dataset,
        steps_per_epoch=int(STEPS_PER_EPOCH),
        validation_steps=int(VAL_STEPS),
        callbacks=callbacks
    )

    drug_discovery_model.load_weights(checkpoint_path)

    # Save the model
    drug_discovery_model.save("drug_discovery_model.keras")