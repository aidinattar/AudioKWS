'''A module to manage the model.'''
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.custom_layers import flatten
from utils.metric_eval import plot_roc_curve, get_all_roc_coordinates, \
                              roc_auc_score, calculate_tpr_fpr
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

class model(object):
    '''A class to manage the model.'''

    def __init__(self,
                 inputs:,
                 loss,
                 optimizer,
                 metrics):
        '''Initialize the class.'''

        self.inputs = inputs
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        for spectrogram, _ in inputs.spectrogram_ds.take(1):
            self.input_shape = spectrogram.shape
        self.num_classes = len(self.inputs.commands)
        self.normalization()
    

    def normalization(self):
        self.norm_layer = tf.keras.layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        self.norm_layer.adapt(data=self.inputs.spectrogram_ds.map(map_func=lambda spec, label: spec))

    def print_input_shape(self):
        '''Print the input shape.'''
        print('Input shape:', self.input_shape)
        print('Output shape:', self.num_classes)

    def summary(self):
        '''Print a summary of the model.'''
        print(self.model.summary())

    def compile(self):
        '''Compile the model.'''
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def fit(self, epochs, callbacks):
        '''Fit the model.'''
        self.history = self.model.fit(self.inputs.train_ds, epochs=epochs, callbacks=callbacks, validation_data=self.inputs.val_ds)

    def save_fit(self, path):
        '''Save the fit history.'''
        with open(path, 'wb') as file:
            pickle.dump(self.history.history, file)

    def load_fit(self, path):
        '''Load the fit history.'''
        with open(path, 'rb') as file:
            self.history = pickle.load(file)

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
        batch_size = self.inputs.batch_size
        len_data = len(list(self.inputs.val_ds))*batch_size

        for i, (spectrogram, label) in enumerate(self.inputs.val_ds.take(1)):
            shape_x = spectrogram.shape

        X_val  = np.zeros(shape=flatten([len_data, list(shape_x[1:])]))
        y_true = np.zeros(shape=(len_data), dtype=np.int32)
        y_pred = np.zeros(shape=(len_data), dtype=np.int32)
        y_prob = np.zeros(shape=(len_data, self.num_classes), dtype=np.float32)

        for i, (spectrogram, label) in enumerate(self.inputs.val_ds):
            start = i*batch_size
            end = (i+1)*batch_size
            try:
                X_val[start:end]  = spectrogram.numpy()
                y_true[start:end] = label.numpy()
                y_prob[start:end] = self.model.predict(spectrogram)
                y_pred[start:end] = np.argmax(y_prob[start:end], axis=1)

            except ValueError:
                last_batch_size = label.numpy().shape[0]
                X_val[start:start+last_batch_size]  = spectrogram.numpy()
                y_true[start:start+last_batch_size] = label.numpy()
                y_prob[start:start+last_batch_size] = self.model.predict(spectrogram)
                y_pred[start:start+last_batch_size] = np.argmax(y_prob[start:start+last_batch_size], axis=1)
                break

        self.X_val  = X_val[:-(batch_size-last_batch_size)]
        self.y_true = y_true[:-(batch_size-last_batch_size)]
        self.y_pred = y_pred[:-(batch_size-last_batch_size)]
        self.y_prob = y_pred[:-(batch_size-last_batch_size)]

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
        self.predict_val()

        cm = tf.math.confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(10, 12))
        sns.heatmap(cm, xticklabels=self.inputs.commands, yticklabels=self.inputs.commands, annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()

        if path != None:
            plt.savefig(path)


    def plot_roc_OvR(self, path=None):
        '''Plots the Probability Distributions and the ROC Curves One vs Rest'''
        self.predict_val()

        plt.figure(figsize = (25, 8))
        bins = [i/20 for i in range(20)] + [1]
        classes = np.unique(self.y_true)
        roc_auc_ovr = {}
        for i in range(len(classes)):
            # Gets the class
            c = classes[i]

            # Prepares an auxiliary array to help with the plots
            class_array = np.array([1 if y == c else 0 for y in self.y_true])
            prob_array = self.y_prob

            # Plots the probability distribution for the class and the rest
            ax = plt.subplot(2, len(classes), i+1)
            plt.hist(prob_array[class_array == 0], bins=bins, color='blue', alpha=0.5, label='Rest')
            plt.hist(prob_array[class_array == 1], bins=bins, color='red', alpha=0.5, label=f'Class: {c}')
            ax.set_title(c)
            ax.legend()
            ax.set_xlabel(f"P(x = {c})")

            # Calculates the ROC Coordinates and plots the ROC Curves
            ax_bottom = plt.subplot(2, len(classes), i+len(classes)+1)
            tpr, fpr = get_all_roc_coordinates(class_array, prob_array)
            plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
            ax_bottom.set_title("ROC Curve OvR")

            # Calculates the ROC AUC OvR
            roc_auc_ovr[c] = roc_auc_score(class_array, prob_array)
        plt.tight_layout()
        plt.show()

        if path != None:
            plt.savefig(path)





    def plot_roc_OvO(self, path=None):
        '''Plots the Probability Distributions and the ROC Curves One vs One'''
        self.predict_val()

        classes_combinations = []
        class_list = list(self.inputs.commands)
        for i in range(len(class_list)):
            for j in range(i+1, len(class_list)):
                classes_combinations.append([class_list[i], class_list[j]])
                classes_combinations.append([class_list[j], class_list[i]])


        plt.figure(figsize = (20, 7))
        bins = [i/20 for i in range(20)] + [1]
        roc_auc_ovo = {}
        for i in range(len(classes_combinations)):
            # Gets the class
            comb = classes_combinations[i]
            c1 = comb[0]
            c2 = comb[1]
            c1_index = class_list.index(c1)
            title = c1 + " vs " +c2

            # Prepares an auxiliar dataframe to help with the plots
            df_aux = pd.DataFrame(self.X_val.copy())
            df_aux['class'] = self.y_true
            df_aux['prob'] = self.y_prob[:, c1_index]

            # Slices only the subset with both classes
            df_aux = df_aux[(df_aux['class'] == c1) | (df_aux['class'] == c2)]
            df_aux['class'] = [1 if y == c1 else 0 for y in df_aux['class']]
            df_aux = df_aux.reset_index(drop = True)

            # Plots the probability distribution for the class and the rest
            ax = plt.subplot(2, 6, i+1)
            sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
            ax.set_title(title)
            ax.legend([f"Class 1: {c1}", f"Class 0: {c2}"])
            ax.set_xlabel(f"P(x = {c1})")

            # Calculates the ROC Coordinates and plots the ROC Curves
            ax_bottom = plt.subplot(2, 6, i+7)
            tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
            plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
            ax_bottom.set_title("ROC Curve OvO")

            # Calculates the ROC AUC OvO
            roc_auc_ovo[title] = roc_auc_score(df_aux['class'], df_aux['prob'])
        plt.tight_layout()

        if path != None:
            plt.savefig(path)