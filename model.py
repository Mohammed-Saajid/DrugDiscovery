import tensorflow as tf
from tensorflow.keras.layers import Embedding
from memory import Memory
from multi_head_attention import *

# Titans Layer (Transformer-Based Memory-Augmented Model)
class DrugDiscoveryModel(tf.keras.layers.Layer):
    """
    Implements a attention architecture with memory augmentation.
    
    This layer integrates an embedding layer, memory module, multi-head attention, 
    feed-forward network, and gating mechanisms to enhance contextual learning.
    
    Parameters:
        embedding_dim (int): Dimensionality of embeddings.
        sequence_length (int): Length of input sequences.
        num_heads (int): Number of attention heads.
        dff (int): Hidden layer size in the feed-forward network.
        total_words (int): Vocabulary size for final classification.
        rate (float): Dropout rate (default: 0.1).
    """
    def __init__(self, embedding_dim, sequence_length, num_heads, dff, total_words, rate = 0.1,final_layer=True,embedding_layer=True,position_embedding=True,mask=None):
        super(DrugDiscoveryModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dff = dff
        self.sequence_length = sequence_length
        self.total_words = total_words
        self.rate = rate
        self.final_layer_ex = final_layer
        self.mask = mask

    def build(self,input_shape):
        """
        Initializes memory, attention, feed-forward, normalization, gating, and final projection layers.
        """
        
        self.embedding_layer = Embedding(input_dim=self.total_words, output_dim=self.embedding_dim,mask_zero=True)
        self.position_embedding = Embedding(input_dim=self.total_words, output_dim=self.embedding_dim)
        self.memory = Memory(self.embedding_dim, self.sequence_length)
        self.mha = MultiHeadAttention(self.embedding_dim * 3, self.num_heads)
        self.ffn = PositionwiseFeedforward(self.embedding_dim * 3, self.dff)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(self.rate)
        self.gate = tf.keras.layers.Dense(units=self.embedding_dim * 3, activation='sigmoid')
        self.modulation_layer = tf.keras.layers.Dense(units=self.embedding_dim * 3)
        self.memory_projection = tf.keras.layers.Dense(units = self.embedding_dim, activation="tanh")
        # Softmax layer for final classification
        if self.final_layer_ex:
            self.final_layer = tf.keras.layers.Dense(units=self.total_words,activation="softmax")
        else:
            self.final_layer = tf.keras.layers.Dense(units=self.embedding_dim, activation='relu')    
        
        super().build(input_shape)  # Register trainable weights

    def call(self, x):
        """
        Forward pass through the Titans layer.
        
        Parameters:
            x (Tensor): Input tensor.        
        Returns:
            Tensor: Final output with softmax probabilities over vocabulary.
        """
        # Embedding and positional encoding
        x = self.embedding_layer(x)
        
        
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embeddings = self.position_embedding(positions)
        x = tf.add(x,position_embeddings)

        # Memory augmentation
        memory_output = self.memory(x)
        
        # Multi-head self-attention
        attn_output = self.mha(memory_output, memory_output, memory_output, self.mask)
        
        # Position-wise feedforward network
        ffn_output = self.ffn(attn_output)
        
        # Layer normalization and dropout
        layer_normalization = self.layernorm(ffn_output)
        dropout = self.dropout(layer_normalization)
        
        # Skip connection
        skip = tf.add(memory_output, dropout)
        
        # Gating mechanism
        linear_gating = self.gate(skip)
        modulated_output = self.modulation_layer(linear_gating)
        output = tf.multiply(linear_gating, modulated_output)
        

        # Final classification layer
        final_output = self.final_layer(output)
        
        return final_output