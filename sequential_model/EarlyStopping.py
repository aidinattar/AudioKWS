import numpy as np
import torch

class EarlyStopping:
    """
    Stop training when a monitored quantity has stopped improving.
    """
    def __init__(self, patience=4, tolerance=0 ):
        """
        Parameters
        ----------
        
        patience: int
            How long to wait after last time validation loss improved in number of epochs.
        tolerance: float
            Tolerance for the validation loss to be considered as improved.


        """
        self.patience = patience
        self.patience_counter = 0

        self.stopped = False
        self.val_loss_min = np.Inf
        self.tolerance = tolerance


    def __call__(self, val_loss):

        if val_loss < self.val_loss_min: # if we observe an improvement
            self.val_loss_min = val_loss
            self.patience_counter = 0 # reset the patience counter
        
        elif val_loss > (self.val_loss_min + self.tolerance): # if we observe a degradation with a certain tolerance
            self.patience_counter += 1
            print (f"EarlyStopping: patience counter = {self.patience_counter} out of {self.patience}")
            if self.patience_counter >= self.patience: # if the patience counter is greater than the patience
                self.stopped = True


    def to_stop(self):
        return self.stopped