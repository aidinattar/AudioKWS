'''
Class to define the Low Rank Dense layer, according to:
https://stats.stackexchange.com/questions/365179/what-is-low-rank-linear-layer-in-neural-networks
'''
import tensorflow as tf

class LowRankDense(tf.keras.layers.Layer):
    '''Low Rank Dense Layer.'''
    def __init__(self, units, rank, **kwargs):
        '''Initialize the class.'''
        super().__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.rank =  int(rank) if not isinstance(rank, int) else rank
        if self.rank < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )

    def build(self, input_shape):
        '''Build the layer.'''
        input_dim = input_shape[-1]
        self.U = self.add_weight(
            name = 'U',
            shape=(self.units, self.rank),
            initializer='glorot_uniform',
            trainable=True
        )
        self.V = self.add_weight(
            name = 'V',
            shape=(self.rank, input_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name = 'bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super(LowRankDense, self).build(input_shape)

    def call(self, inputs):
        '''Call the layer.'''
        UV = tf.matmul(self.U, self.V)
        outputs = tf.matmul(inputs, UV, transpose_b=True) + self.bias
        return outputs

    def get_config(self):
        '''Get the config.'''
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'rank': self.rank
        })
        return config