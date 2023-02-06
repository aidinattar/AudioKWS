'''A module to manage the model.'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc

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
        self.normalization()

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

    def fit(self, epochs, callbacks):
        '''Fit the model.'''
        self.history = self.model.fit(self.inputs.train_ds, epochs=epochs, callbacks=callbacks, validation_data=self.inputs.val_ds)

    def evaluate_train(self):
        '''Evaluate the model on the train set.'''
        self.results_train = self.model.evaluate(self.inputs.train_ds)
        print(self.results_train)

    def predict_train(self):
        '''Predict the model on the train set.'''
        self.predictions_train = self.model.predict(self.inputs.train_ds)

    def evaluate_val(self):
        '''Evaluate the model on the validation set.'''
        self.results_val = self.model.evaluate(self.inputs.val_ds)
        print(self.results_val)

    def predict_val(self):
        '''Predict the model on the validation set.'''
        self.predictions_val = self.model.predict(self.inputs.val_ds)

    def evaluate_test(self):
        '''Evaluate the model on the test set.'''
        y_pred, y_true = self.predict_test()
        test_acc = sum(y_pred == y_true) / len(y_true)
        print(f'Test set accuracy: {test_acc:.0%}')

    def predict_test(self):
        '''Predict the model on the test set.'''
        test_audio = []
        test_labels = []

        for audio, label in self.inputs.test_ds:
            test_audio.append(audio.numpy())
            test_labels.append(label.numpy())

        test_audio = np.array(test_audio)
        test_labels = np.array(test_labels)

        self.predictions_test = np.argmax(model.predict(test_audio), axis=1)
        return self.predictions_test, test_labels

    def save(self, path):
        '''Save the model.'''
        self.model.save(path)

    def load(self, path):
        '''Load the model.'''
        self.model = tf.keras.models.load_model(path)

    def plot_model(self, path):
        '''Plot the model.'''
        tf.keras.utils.plot_model(self.model, path, show_shapes=True, show_layer_names=True, expand_nested=True, dpi=96)

    def plot_training(self, path=None):
        '''Plot the training.'''
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot accuracy
        train_acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        epochs = range(1, len(train_acc) + 1)
        ax1.plot(epochs, train_acc, '-o', label='Training Accuracy')
        ax1.plot(epochs, val_acc, '-o', label='Validation Accuracy')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        y1 = np.array(train_acc)
        y2 = np.array(val_acc)
        ax1.fill_between(epochs, y1, y2, where=(y2 > y1), interpolate=True, color='gray', alpha=0.5)

        # Plot loss
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        ax2.plot(epochs, train_loss, '-o', label='Training Loss')
        ax2.plot(epochs, val_loss, '-o', label='Validation Loss')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()

        y3 = np.array(train_loss)
        y4 = np.array(val_loss)
        ax2.fill_between(epochs, y3, y4, where=(y4 > y3), interpolate=True, color='gray', alpha=0.5)

        plt.tight_layout()
        plt.show()

        if path != None:
            fig.savefig(path)

    def plot_confusion_matrix(self, path=None):
        '''Plot the confusion matrix.'''
        y_true = []
        y_pred = []
        for spectrogram, label in self.inputs.val_ds:
            y_true.append(label.numpy().argmax())
            y_pred.append(np.argmax(self.model.predict(spectrogram)))

        cm = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, xticklabels=self.inputs.commands, yticklabels=self.inputs.commands, annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()

        if path != None:
            plt.savefig(path)

    def plot_roc(self, path=None):
        '''Plot the ROC curve.'''
        y_true = []
        y_pred = []
        for spectrogram, label in self.inputs.val_ds:
            y_true.append(label.numpy().argmax())
            y_pred.append(np.argmax(self.model.predict(spectrogram)))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_labels):
            fpr[i], tpr[i], _ = roc_curve(y_true, y_pred)
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        for i in range(self.num_labels):
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

        if path != None:
            plt.savefig(path)