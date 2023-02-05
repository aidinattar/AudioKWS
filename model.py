'''A module to manage the model.'''
import tensorflow as tf

class model(object):
    '''A class to manage the model.'''

    def __init__(self, inputs, loss, optimizer, metrics):
        '''Initialize the class.'''

        self.inputs = inputs
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        for spectrogram, _ in inputs.spectrogram_ds.take(1):
            self.input_shape = spectrogram.shape
        self.num_labels = len(self.inputs.commands)
        #self.normalization()

    def normalization(self):
        self.norm_layer = tf.keras.layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        self.norm_layer.adapt(data=self.inputs.spectrogram_ds.map(map_func=lambda spec, label: spec))

    def print_input_shape(self):
        '''Print the input shape.'''
        print('Input shape:', self.input_shape)
        print('Output shape:', self.num_labels)

    def summary(self):
        '''Print a summary of the model.'''
        print(self.model.summary())

    def compile(self):
        '''Compile the model.'''
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def fit(self, epochs, callbacks, validation_data):
        '''Fit the model.'''
        self.model.fit(self.inputs, epochs=epochs, callbacks=callbacks, validation_data=validation_data)

    def evaluate(self):
        '''Evaluate the model.'''
        self.model.evaluate()