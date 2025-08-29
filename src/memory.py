import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization


class LongTermMemory(tf.keras.layers.Layer):
    """
    Long-term memory layer that compresses and refines input representations.
    """
    def __init__(self, units, activation=tf.keras.activations.silu, **kwargs):
        super(LongTermMemory, self).__init__(**kwargs)
        """
        A long-term memory layer that compresses and refines input representations.
        Args:
            units (int): The number of output units.
            activation (callable): The activation function to use.
        Returns:
            None
        """
        super(LongTermMemory, self).__init__()
        self.units = units
        self.activation = activation
        self.name = "Long_Term_Memory"

        # Define layers
        self.fc1 = Dense(self.units, activation=self.activation)
        self.fc2 = Dense(self.units * 2, activation=self.activation)
        self.fc3 = Dense(self.units, activation=self.activation)
    
    def call(self, inputs, mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Memory(tf.keras.layers.Layer):
    """
    Memory module with long-term + persistent memory integration.
    """
    def __init__(self, embedding_dim, sequence_length, activation=tf.keras.activations.silu, **kwargs):
        super(Memory, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.activation = activation
        self.name = "Memory"

    
        # Query transformation
        self.LMWq = Dense(units=self.embedding_dim, activation=self.activation, use_bias=False, name="Query_Transformation_Layer")
        # Long-term memory
        self.LM = LongTermMemory(self.embedding_dim, activation=self.activation)

        # Persistent memory vector (trainable, sequence-independent)
        self.persistent_memory = self.add_weight(
            shape=(1, 1, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="Persistent_Memory"
        )

        # Normalization after concatenation
        self.norm = LayerNormalization(epsilon=1e-6, name="Memory_Normalization_Layer")

    def call(self, inputs, mask=None):
        q = self.LMWq(inputs)
        ltm_out = self.LM(q)

        batch_size = tf.shape(inputs)[0]
        persistent = tf.tile(self.persistent_memory, [batch_size, self.sequence_length, 1])

        concat = tf.concat([inputs, ltm_out, persistent], axis=-1)
        norm = self.norm(concat)

        return norm

