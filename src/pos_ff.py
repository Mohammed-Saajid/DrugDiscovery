import tensorflow as tf
from tensorflow.keras.layers import Dense



class PositionwiseFeedforward(tf.keras.layers.Layer):
    """
    Standard FFN (expansion + projection) used in Transformers.
    """
    def __init__(self, embedding_dim, dff, **kwargs):
        super(PositionwiseFeedforward, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.dff = dff
        self.name = "Positionwise_Feedforward"
        self.dense1 = Dense(self.dff, activation='relu')
        self.dense2 = Dense(self.embedding_dim)

    def call(self, x):
        return self.dense2(self.dense1(x))
