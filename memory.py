import tensorflow as tf
from tensorflow.keras.layers import Dense



# Complete Memory Implementation
class Memory(tf.keras.layers.Layer):
    """
    Implements a memory mechanism combining a long-term memory layer with persistent memory.
    
    Parameters:
        embedding_dim (int): Dimensionality of embeddings.
        sequence_length (int): Length of the input sequence.
        activation (function): Activation function to use (default: SiLU).
    """
    def __init__(self, embedding_dim, sequence_length, activation=tf.keras.activations.silu):
        super(Memory, self).__init__()
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.activation = activation

    def build(self):
        """
        Initializes layers for memory processing.
        """
        self.fc1 = Dense(self.embedding_dim, activation=self.activation)
        self.fc2 = Dense(self.embedding_dim * 2, activation=self.activation)
        self.fc3 = Dense(self.embedding_dim, activation=self.activation)
        
        # Query transformation layer
        self.LMWq = Dense(units=self.embedding_dim, activation=self.activation, use_bias=False)
                
        # Persistent memory layer
        self.persistent_memory = Dense(self.embedding_dim)

    def call(self, inputs):
        """
        Forward pass through the Memory module.
        
        Parameters:
            inputs (Tensor): Input tensor.
        
        Returns:
            Tensor: Concatenated output of transformed inputs, long-term memory, and persistent memory.
        """
        x = self.LMWq(inputs)  # Apply query transformation
        x = self.fc1(x)  # Pass through long-term memory layer
        x = self.fc2(x)
        x = self.fc3(x)
        persistent_memory = self.persistent_memory(inputs)  # persistent memory
        
        # Concatenate input, long-term memory output, and persistent memory along the last axis
        return tf.concat([inputs, x, persistent_memory], axis=-1)