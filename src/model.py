import tensorflow as tf
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dropout
from .memory import Memory
from .pos_ff import *
import keras


# Define a custom model using Titans Transformer-based architecture
@keras.saving.register_keras_serializable()
class Titans(tf.keras.Model):
    """
    Transformer-based memory-augmented architecture with masking support.
    """

    def __init__(self, embedding_dim, sequence_length, num_heads, dff, vocab_size, 
                 rate=0.4, mask_zero=True, **kwargs):     
        """
        Initializes the Titans layer.
        Args:
            embedding_dim (int): Dimensionality of the embedding space.
            sequence_length (int): Length of the input sequences.
            num_heads (int): Number of attention heads.
            dff (int): Dimensionality of the feedforward network.
            vocab_size (int): Total number of words in the vocabulary.
            rate (float): Dropout rate.
            mask_zero (bool): Whether to mask padding tokens.
        Returns:
            None
        """

        super(Titans, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dff = dff
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.supports_masking = mask_zero
        self.mask_zero = mask_zero
        self.rate = rate
        self.name = "Titans"

    
    def build(self, input_shape):
        #Initializes layers (embedding, memory, attention, FFN, normalization, gating).
        
        # Embedding + positional encoding
        self.embedding_layer = Embedding(
            input_dim=self.vocab_size,   
            output_dim=self.embedding_dim,
            mask_zero=self.mask_zero
        )

        self.position_embedding = Embedding(
            input_dim=self.sequence_length,
            output_dim=self.embedding_dim
        )

        # Memory + Transformer components
        self.memory = Memory(self.embedding_dim, self.sequence_length)
        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embedding_dim)
        self.ffn = PositionwiseFeedforward(self.embedding_dim * 3, self.dff)

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(self.rate)

        # Gating
        self.gate = Dense(units=self.embedding_dim * 3, activation='sigmoid')
        self.modulation_layer = Dense(units=self.embedding_dim * 3)

        # Final linear layer
        self.final_layer = Dense(units=self.vocab_size)

    def create_causal_mask(self,seq_len):
        """
        Creates a causal mask for the given sequence length.
        Args:
            seq_len (int): Length of the sequence.
        Returns:
            tf.Tensor: Causal mask tensor.
        """
        return tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    def combine_masks(self,pad_mask, seq_len):
        """
        Combines padding and causal masks.
        Args:
            pad_mask (tf.Tensor): Padding mask tensor.
            seq_len (int): Length of the sequence.
        Returns:
            tf.Tensor: Combined mask tensor.
        """
        causal_mask = self.create_causal_mask(seq_len)  # (seq_len, seq_len)
        causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]  # (1,1,seq,seq)
    
        # pad mask → (batch, 1, 1, seq)
        pad_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]
    
        # Combine (broadcast AND)
        return tf.cast(tf.logical_and(tf.cast(pad_mask, tf.bool),
                                  tf.cast(causal_mask, tf.bool)), tf.float32)

    def call(self, inputs, mask=None, training=False):

        # Embedding
        x = self.embedding_layer(inputs)
    
        # Padding mask
        if mask is None:
            mask = self.embedding_layer.compute_mask(inputs)   # (batch, seq_len)


        # Attention mask
        attn_mask = self.combine_masks(mask, self.sequence_length)  # (batch,1,seq,seq)
        pad_mask = tf.cast(mask[:, :, tf.newaxis], x.dtype)  # (batch, seq_len, 1) for element-wise masking
    
        # Positional encoding
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        pos_emb = self.position_embedding(positions)
        x = tf.add(x, pos_emb)

        # Memory augmentation
        memory_output = self.memory(x, mask=mask)
        memory_output *= pad_mask   # ensure padding stays zero

        # Multi-head attention
        attn_output = self.mha(
        memory_output,   # query
        memory_output,   # value
        memory_output,   # key
        attention_mask=attn_mask,
        training=training
        )

        attn_output *= pad_mask   # re-mask after MHA

        # Feedforward
        ffn_output = self.ffn(attn_output)
        ffn_output = self.layernorm(ffn_output)
        ffn_output = self.dropout(ffn_output, training=training)
        ffn_output *= pad_mask   # re-mask after FFN + norm

        # Skip connection
        skip = tf.add(memory_output, ffn_output)
        skip *= pad_mask   # ensure skip preserves masking

        # Gating
        linear_gating = self.gate(skip)
        modulated_output = self.modulation_layer(linear_gating)
        output = tf.multiply(linear_gating, modulated_output)
        output *= pad_mask   # final mask application

        # Final projection

        return self.final_layer(output)   # logits over vocab
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "sequence_length": self.sequence_length,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "vocab_size": self.vocab_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        


    



