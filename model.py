"""A module to manage the model."""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from utils.custom_layers import flatten
from utils.metric_eval import plot_roc_curve, get_all_roc_coordinates, \
                              roc_auc_score, calculate_tpr_fpr
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from DataSource import DataSource
from tensorflow.keras.metrics import Accuracy, Precision, Recall, F1Score, AUC

class Model(object):
    """A class to manage the model."""

    created = False
    compiled = False
    trained = False

    def __init__(self,
                 inputs:DataSource):
        """
        Initialize the class.
        
        Parameters
        ----------
        inputs : DataSource
            The data to use for the training.
        loss : str
            The loss function.
        optimizer : str
            The optimizer.
        metrics : list
            The metrics to use.
        """

        # Set the attributes.
        self.inputs = inputs
        
        # Get the shape of the input as the dimensions of the spectrogram.
        for spectrogram, _ in inputs.spectrogram_ds.take(1):
            self.input_shape = spectrogram.shape
        # Get the number of classes.
        self.num_classes = len(self.inputs.commands)
    

    def _norm_layer(self):
        """
        Normalization layer.
        
        Returns
        -------
        tf.keras.layers.Normalization
            The normalization layer.
        """
        # Create a normalization layer.
        norm_layer = tf.keras.layers.Normalization()
        # Fit the state of the layer to the spectrograms with `Normalization.adapt`.
        return norm_layer.adapt(
            data = self.inputs.spectrogram_ds.map(
                map_func = lambda spec,
                label: spec
            )
        )


    def print_input_shape(self):
        """
        Print the input shape.
        """
        print('Input shape:', self.input_shape)
        print('Output shape:', self.num_classes)


    def create_model(self):
        """
        Create the model.
        """
        self.define_model()
        self.created = True
        
        
    def define_model(self):
        """
        Parent method to define the model.
        """
        pass


    def summary(self):
        """
        Print a summary of the model.
        """
        print(self.model.summary())


    def compile(self,
                loss:str,
                optimizer:str,
                metrics:list):
        """
        Compile the model.
        
        Parameters
        ----------
        loss : str
            The loss function.
        optimizer : str
            The optimizer.
        metrics : list
            The metrics to use.
        """
        
        if not self.created:
            raise Exception('The model has not been created yet.')
        
        # Set the attributes.
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = self.metrics
        )
        
        self.compiled = True


    def fit(self,
            epochs:int,
            callbacks:list,
            verbose:int = 1,
            return_history:bool = False):
        """
        Train the model.
        
        Parameters
        ----------
        epochs : int
            The number of epochs.
        callbacks : list
            The callbacks to use.
        """
        # Check if the model has been compiled.
        if not self.compiled:
            raise Exception('The model has not been compiled yet.')

        self.history = self.model.fit(self.inputs.train_ds,
                                      epochs = epochs,
                                      callbacks = callbacks,
                                      validation_data = self.inputs.val_ds
                                      verbose = verbose)
        
        self.trained = True
        
        if return_history:
            return self.history


    def save_fit(self,
                 path:str):
        """
        Save the fit history.
        
        Parameters
        ----------
        path : str
            The path to save the history.
        """
        with open(path, 'wb') as file:
            pickle.dump(self.history.history, file)


    def load_fit(self,
                 path:str,
                 return_history:bool=False):
        """
        Load the fit history.
        
        Parameters
        ----------
        path : str
            The path to load the history.
        """
        with open(path, 'rb') as file:
            self.history = pickle.load(file)
            
        if return_history:
            return self.history


    def _predict(self,
                 ds:tf.data.Dataset):
        """
        Compute the predictions on a dataset.
        
        Parameters
        ----------
        ds : tf.data.Dataset
            The dataset to predict.
        """
        if not self.trained:
            raise Exception('The model has not been trained yet.')
        
        preds = self.model.predict(ds)
    
        batch_size = self.inputs.batch_size
        len_data = len(list(ds))*batch_size

        for i, (spectrogram, label) in enumerate(ds.take(1)):
            shape_x = spectrogram.shape

        X = np.zeros(shape = flatten([len_data, list(shape_x[1:])]))
        y_true = np.zeros(shape = (len_data), dtype = np.int32)
        y_pred = np.zeros(shape = (len_data), dtype = np.int32)
        y_prob = np.zeros(shape = (len_data, self.num_classes), dtype = np.float32)

        for i, (spectrogram, label) in enumerate(ds):
            start = i * batch_size
            end = (i + 1)*batch_size
            try:
                X[start : end] = spectrogram.numpy()
                y_true[start : end] = label.numpy()
                y_prob[start : end] = self.model.predict(spectrogram)
                y_pred[start : end] = np.argmax(y_prob[start : end], axis = 1)

            except ValueError:
                last_batch_size = label.numpy().shape[0]
                X[start : start + last_batch_size] = spectrogram.numpy()
                y_true[start : start + last_batch_size] = label.numpy()
                y_prob[start : start + last_batch_size] = self.model.predict(spectrogram)
                y_pred[start : start + last_batch_size] = np.argmax(y_prob[start : start + last_batch_size], axis = 1)
                break

        return X[:-(batch_size - last_batch_size)],\
               y_true[:-(batch_size - last_batch_size)],\
               y_pred[:-(batch_size - last_batch_size)]

    
    def predict_train(self):
        """
        Predict the model on the train set.
        """
        self.x_train, self.predictions_train, self.probabilities_train=self._predict(self.inputs.train_ds)


    def predict_val(self):
        """
        Predict the model on the validation set.
        """
        self.x_val, self.predictions_val, self.probabilities_val=self._predict(self.inputs.val_ds)
        
        
    def predict_test(self):
        """
        Predict the model on the test set.
        """
        self.x_test, self.predictions_test, self.probabilities_test=self._predict(self.inputs.test_ds)


    def evaluate_train(self):
        """Evaluate the model on the train set."""
        self.results_train=self.model.evaluate(self.inputs.train_ds)
        print(self.results_train)
    

    def evaluate_val(self):
        """Evaluate the model on the validation set."""
        self.results_val=self.model.evaluate(self.inputs.val_ds)
        print(self.results_val)


    def evaluate_test(self):
        """Evaluate the model on the test set."""
        #y_pred, y_true=self.predict_test()
        #test_acc=sum(y_pred == y_true) / len(y_true)
        #print(f'Test set accuracy: {test_acc:.0%}')
        self.results_test=self.model.evaluate(self.inputs.test_ds)


    def _accuracy(self,
                  y_true:np.ndarray,
                  y_pred:np.ndarray,
                  **kwargs) -> float:
        """
        Compute the accuracy.
        
        Parameters
        ----------
        y_true : np.ndarray
            The true labels.
        y_pred : np.ndarray
            The predicted labels.
        **kwargs : dict
            The arguments to pass to the accuracy object.
            
        Returns
        -------
        accuracy : float
            The accuracy.
        """
        accuracy=Accuracy(**kwargs)
        accuracy.update_state(y_true, y_pred)
        
        return accuracy.result().numpy()


    def _precision(self,
                   y_true:np.ndarray,
                   y_pred:np.ndarray,
                   **kwargs) -> float:
        """
        Compute the precision.
        
        Parameters
        ----------
        y_true : np.ndarray
            The true labels.
        y_pred : np.ndarray
            The predicted labels.
        **kwargs : dict
            The arguments to pass to the precision object.
            
        Returns
        -------
        precision : float
            The precision.
        """
        precision=Precision(**kwargs)
        precision.update_state(y_true, y_pred)

        return precision.result().numpy()


    def _recall(self,
                y_true:np.ndarray,
                y_pred:np.ndarray,
                **kwargs) -> float:
        """
        Compute the recall.
        
        Parameters
        ----------
        y_true : np.ndarray
            The true labels.
        y_pred : np.ndarray
            The predicted labels.
        **kwargs : dict
            The arguments to pass to the recall object.
            
        Returns
        -------
        recall : float
            The recall.
        """
        recall=Recall(**kwargs)
        recall.update_state(y_true, y_pred)
        
        return recall.result().numpy()


    def _f1(self,
            y_true:np.ndarray,
            y_pred:np.ndarray,
            **kwargs) -> float:
        """
        Compute the F1 score.
        
        Parameters
        y_true : np.ndarray
            The true labels.
        y_pred : np.ndarray
            The predicted labels.
        **kwargs : dict
            The arguments to pass to the F1 score object.
            
        Returns
        -------
        f1 : float
            The F1 score.
        """
        f1=F1Score(**kwargs)
        f1.update_state(y_true, y_pred)
        
        return f1.result().numpy()
    
    
    def _confusion_matrix(self,
                          y_true:np.ndarray,
                          y_pred:np.ndarray,
                          return_cm:bool=True,
                          display:bool=True,
                          save:bool=False,
                          dir:str=None,
                          name:str=None,
                          **kwargs) -> np.ndarray:
        """
        Compute the confusion matrix.
        
        Parameters
        ----------
        y_true : np.ndarray
            The true labels.
        y_pred : np.ndarray
            The predicted labels.
        **kwargs : dict
            The arguments to pass to the confusion matrix object.
        
        Returns
        -------
        confusion_matrix : np.ndarray
            The confusion matrix.
        """
        confusion_matrix=confusion_matrix(
            y_true,
            y_pred,
            **kwargs
        )
        
        if display or save:
            fig, ax = plt.subplots(figsize=(10, 10))
            disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
            plt.tight_layout()
            
            if display:
                plt.show()
    
            if save:
                if dir is None:
                    dir='figures'
                if name is None:
                    name='confusion_matrix.png'

                fig.savefig(
                    os.path.join(
                        dir,
                        name
                    )
                )
                
        if return_cm:
            return confusion_matrix


    def _roc_curve(self,
                   y_pred_prob:np.ndarray,
                   y_true:np.ndarray,
                   return_roc:bool=True,
                   display:bool=True,
                   save:bool=True,
                   dir:str=None,
                   name:str=None,
                   **kwargs):
        """
        Compute the ROC curve of the model.

        Parameters
        ----------
        y_pred_prob: np.array
            Predicted label probabilities.
            Default: None
        y_true: np.array
            True labels.
            Default: None
        return_roc: bool
            Whether to return the ROC curve or not.
            Default: True
        display: bool
            Whether to display the figure or not.
            Default: True
        save: bool
            Whether to save the figure or not.
            Default: True
        dir: str
            Directory to save the figure.
            Default: None
        name: str
            Name of the figure.
            Default: None
        **kwargs:
            Parameters for the ROC curve method.
        """
        if not self.trained:
            raise ValueError('Model not trained')

        # Binarize the labels
        y_binarized = label_binarize(y_true, classes=range(self.num_classes))

        # Compute the ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and AUC
        fpr_micro, tpr_micro, _ = roc_curve(y_binarized.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        if display or save:
            # Plot the ROC curves for each class
            plt.figure()
            for i in range(self.num_classes):
                plt.plot(fpr[i], tpr[i], label='Class {0} (AUC = {1:.2f})'.format(i, roc_auc[i]))
            plt.plot(fpr_micro, tpr_micro, label='Micro-average (AUC = {0:.2f})'.format(roc_auc_micro))

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Multiclass Classification')
            plt.legend(loc='lower right')
            
            if display:
                plt.show()
            
            if save:
                if dir is None:
                    dir='figures'
                if name is None:
                    name='roc_curve.png'
                plt.savefig(
                    os.path.join(
                        dir,
                        name
                    )
                )


    def save(self,
             path:str):
        """
        Save the model.
        
        Parameters
        ----------
        path : str
            The path to save the model.
        """
        self.model.save(path)


    def load(self,
             path:str):
        """
        Load the model.
        
        Parameters
        ----------
        path : str
            The path to load the model.
        """
        self.model=tf.keras.models.load_model(path)


    def plot_model(self,
                   path:str,
                   **kwargs):
        """
        Plot the model.
        
        Parameters
        ----------
        path : str
            The path to save the model.
        **kwargs : dict
            The arguments to pass to the plot_model method.
        """
        tf.keras.utils.plot_model(
            model=self.model,
            to_file=path,
            #show_shapes=True,
            #show_layer_names=True,
            #expand_nested=True,
            #dpi=96,
            **kwargs)


    def plot_training(self, path=None):
        """Plot the training."""
        fig, (ax1, ax2)=plt.subplots(2, 1, figsize=(10, 8))

        # Plot accuracy
        train_acc=self.history.history['accuracy']
        val_acc=self.history.history['val_accuracy']
        epochs=range(1, len(train_acc) + 1)
        ax1.plot(epochs, train_acc, '-o', label='Training Accuracy')
        ax1.plot(epochs, val_acc, '-o', label='Validation Accuracy')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        y1=np.array(train_acc)
        y2=np.array(val_acc)
        ax1.fill_between(epochs, y1, y2, where=(y2 > y1), interpolate=True, color='gray', alpha=0.5)

        # Plot loss
        train_loss=self.history.history['loss']
        val_loss=self.history.history['val_loss']
        ax2.plot(epochs, train_loss, '-o', label='Training Loss')
        ax2.plot(epochs, val_loss, '-o', label='Validation Loss')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()

        y3=np.array(train_loss)
        y4=np.array(val_loss)
        ax2.fill_between(epochs, y3, y4, where=(y4 > y3), interpolate=True, color='gray', alpha=0.5)

        plt.tight_layout()
        plt.show()

        if path != None:
            fig.savefig(path)


    def plot_confusion_matrix(self, path=None):
        """Plot the confusion matrix."""
        self.predict_val()

        cm=tf.math.confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(10, 12))
        sns.heatmap(cm, xticklabels=self.inputs.commands, yticklabels=self.inputs.commands, annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()

        if path != None:
            plt.savefig(path)


    def plot_roc_OvR(self, path=None):
        """Plots the Probability Distributions and the ROC Curves One vs Rest"""
        self.predict_val()

        plt.figure(figsize=(25, 8))
        bins=[i/20 for i in range(20)] + [1]
        classes=np.unique(self.y_true)
        roc_auc_ovr={}
        for i in range(len(classes)):
            # Gets the class
            c=classes[i]

            # Prepares an auxiliary array to help with the plots
            class_array=np.array([1 if y == c else 0 for y in self.y_true])
            prob_array=self.y_prob

            # Plots the probability distribution for the class and the rest
            ax=plt.subplot(2, len(classes), i + 1)
            plt.hist(prob_array[class_array == 0], bins=bins, color='blue', alpha=0.5, label='Rest')
            plt.hist(prob_array[class_array == 1], bins=bins, color='red', alpha=0.5, label=f'Class: {c}')
            ax.set_title(c)
            ax.legend()
            ax.set_xlabel(f"P(x={c})")

            # Calculates the ROC Coordinates and plots the ROC Curves
            ax_bottom=plt.subplot(2, len(classes), i + len(classes) + 1)
            tpr, fpr=get_all_roc_coordinates(class_array, prob_array)
            plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
            ax_bottom.set_title("ROC Curve OvR")

            # Calculates the ROC AUC OvR
            roc_auc_ovr[c]=roc_auc_score(class_array, prob_array)
        plt.tight_layout()
        plt.show()

        if path != None:
            plt.savefig(path)



    def plot_roc_OvO(self, path=None):
        """Plots the Probability Distributions and the ROC Curves One vs One"""
        self.predict_val()

        classes_combinations=[]
        class_list=list(self.inputs.commands)
        for i in range(len(class_list)):
            for j in range(i+1, len(class_list)):
                classes_combinations.append([class_list[i], class_list[j]])
                classes_combinations.append([class_list[j], class_list[i]])


        plt.figure(figsize=(20, 7))
        bins=[i/20 for i in range(20)] + [1]
        roc_auc_ovo={}
        for i in range(len(classes_combinations)):
            # Gets the class
            comb=classes_combinations[i]
            c1=comb[0]
            c2=comb[1]
            c1_index=class_list.index(c1)
            title=c1 + " vs " +c2

            # Prepares an auxiliar dataframe to help with the plots
            df_aux=pd.DataFrame(self.X_val.copy())
            df_aux['class']=self.y_true
            df_aux['prob']=self.y_prob[:, c1_index]

            # Slices only the subset with both classes
            df_aux=df_aux[(df_aux['class'] == c1) | (df_aux['class'] == c2)]
            df_aux['class']=[1 if y == c1 else 0 for y in df_aux['class']]
            df_aux=df_aux.reset_index(drop=True)

            # Plots the probability distribution for the class and the rest
            ax=plt.subplot(2, 6, i+1)
            sns.histplot(x="prob", data=df_aux, hue='class', color='b', ax=ax, bins=bins)
            ax.set_title(title)
            ax.legend([f"Class 1: {c1}", f"Class 0: {c2}"])
            ax.set_xlabel(f"P(x={c1})")

            # Calculates the ROC Coordinates and plots the ROC Curves
            ax_bottom=plt.subplot(2, 6, i+7)
            tpr, fpr=get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
            plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
            ax_bottom.set_title("ROC Curve OvO")

            # Calculates the ROC AUC OvO
            roc_auc_ovo[title]=roc_auc_score(df_aux['class'], df_aux['prob'])
        plt.tight_layout()

        if path != None:
            plt.savefig(path)