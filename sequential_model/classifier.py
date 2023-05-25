import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import copy
from torch import nn

from tqdm import tqdm
import pandas as pd
from sklearn.metrics import  recall_score, precision_score, f1_score, accuracy_score

class classifier ():
    def __init__(self, model = None, device = 'cpu',):
        torch.manual_seed(0)
        self.device = device
        self.Net = model
        self.Net.to(device)

    def train(self,
            train_dataloader, 
            val_dataloader, 
            loss_fn, 
            epochs, 
            optimizer,
            save_dir = 'checkpoints', 
            start_epoch = 0,
            save_every = 10,
            early_stopping = None,
            ):
        """
        Train the model

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            Dataloader for the training set
        val_dataloader : torch.utils.data.DataLoader
            Dataloader for the validation set
        loss_fn : torch.nn.modules.loss
            Loss function
        epochs : int
            Number of epochs to train
        optimizer : torch.optim.Optimizer
            Optimizer
        save_dir : str, optional
            Directory to save checkpoints, by default 'checkpoints'
        start_epoch : int, optional
            Epoch to start training from, by default 0
        save_every : int, optional
            Save checkpoint every n epochs, by default 10
        early_stopping : EarlyStopping, optional
            Early stopping object, by default None
        use_extra_dataset : bool, optional
            Use extra dataset, by default False (syllables or BPE)

        """
        # checks
        if not isinstance(train_dataloader, torch.utils.data.DataLoader):
            raise TypeError(f'train_dataloader must be a torch.utils.data.DataLoader, but got {type(train_dataloader)}')
        if not isinstance(val_dataloader, torch.utils.data.DataLoader):
            raise TypeError(f'val_dataloader must be a torch.utils.data.DataLoader, but got {type(val_dataloader)}')
        if not isinstance(epochs, int):
            raise TypeError(f'epochs must be an int, but got {type(epochs)}')
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f'optimizer must be a torch.optim.Optimizer, but got {type(optimizer)}')
        if not isinstance(save_dir, str):
            raise TypeError(f'save_dir must be a str, but got {type(save_dir)}')
        if not isinstance(start_epoch, int):
            raise TypeError(f'start_epoch must be an int, but got {type(start_epoch)}')
        if not isinstance(save_every, int):
            raise TypeError(f'save_every must be an int, but got {type(save_every)}')
        # values checks
        if epochs < 1:
            raise ValueError(f'epochs must be >= 1, but got {epochs}')
        if start_epoch < 0:
            raise ValueError(f'start_epoch must be >= 0, but got {start_epoch}')
        if save_every < 1:
            raise ValueError(f'save_every must be >= 1, but got {save_every}')

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_loss_log = []
        self.val_loss_log = []
        self.train_loss_labels = pd.DataFrame() # loss for each label
        self.val_loss_labels = pd.DataFrame() # loss for each label

        # if save_dir doews not exist, create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load loss log
        if os.path.exists(f'{save_dir}/train_loss.npy'):
            train_loss = np.load(f'{save_dir}/train_loss.npy')
            self.train_loss_log = train_loss.tolist()

        if os.path.exists(f'{save_dir}/val_loss.npy'):
            val_loss = np.load(f'{save_dir}/val_loss.npy')
            self.val_loss_log = val_loss.tolist()
        
        if os.path.exists(f'{save_dir}/train_loss_labels.csv'):
            self.train_loss_labels = pd.read_csv(f'{save_dir}/train_loss_labels.csv', index_col=0)

        if os.path.exists(f'{save_dir}/val_loss_labels.csv'):
            self.val_loss_labels = pd.read_csv(f'{save_dir}/val_loss_labels.csv', index_col=0)

        best_model = None
        best_val = np.inf
        for epoch_num in range(start_epoch, start_epoch+epochs):
            print ('='*20)
            print(f'EPOCH {epoch_num}')

            ### TRAIN
            train_loss= []

            self.Net.train() # Training mode (e.g. enable dropout, batchnorm updates,...)
            print ("TRAINING")
            for sample_batched in tqdm(train_dataloader):
                # Move data to device
                x_batch = sample_batched[0].to(self.device)
                label_batch = sample_batched[1].to(self.device)                    
                out = self.Net(x_batch)

                # Compute loss
                # labels are int values, so we need to convert them to long
                if label_batch.dtype in [torch.int64, torch.int32]:
                    label_batch = label_batch.long()

                loss = self.loss_fn(out, label_batch)
                loss = loss.mean()
            
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # Update the weights
                self.optimizer.step()

                # Save train loss for this batch
                train_loss.append(loss.detach().cpu().numpy())
            
            # Save average train loss
            train_loss = np.mean(train_loss)

            # Validation
            val_loss= []
            self.Net.eval() # Evaluation mode (e.g. disable dropout, batchnorm,...)

            with torch.no_grad(): # Disable gradient tracking
                print ("TESTING")
                for sample_batched in tqdm(val_dataloader):
                    # Move data to device
                    x_batch = sample_batched[0].to(self.device)
                    label_batch = sample_batched[1].to(self.device)  

                    # Forward pass
                    out = self.Net(x_batch)

                    # Compute loss cross entropy
                    if label_batch.dtype in [torch.int64, torch.int32]:
                        label_batch = label_batch.long()
                        
                    loss = self.loss_fn(out, label_batch)

                    # Save val loss for this batch
                    loss_batch = loss.detach().cpu().numpy()
                    val_loss.append(np.mean(loss_batch))

                # Save average validation loss
                val_loss = np.mean(val_loss)
                self.val_loss_log.append(val_loss)

            # best model
            if val_loss < best_val:
                best_val = val_loss
                # save best model
                self.save_state_dict(f'{save_dir}/best_model.torch')
            # logs
            print(f"Epoch {epoch_num} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")

            # early stopping
            if early_stopping is not None:
                early_stopping(val_loss) # call
                if early_stopping.to_stop():
                    print ("Early stopping, saving model and optimizer states before exiting")

                    self.save_state_dict(f'{save_dir}/model_{epoch_num}.torch')
                    self.save_optimizer_state(f'{save_dir}/optimizer_{epoch_num}.torch')
                    break

            # save model every save_every epochs
            if epoch_num % save_every == 0:
                self.save_state_dict(f'{save_dir}/model_{epoch_num}.torch')
                self.save_optimizer_state(f'{save_dir}/optimizer_{epoch_num}.torch')

            np.save(f'{save_dir}/train_loss.npy', self.train_loss_log)
            np.save(f'{save_dir}/val_loss.npy', self.val_loss_log)


    def history(self):
        return self.train_loss_log, self.val_loss_log
    
    def plot_history(self, save_dir='.'):
        import seaborn as sns
        sns.set_theme (style="darkgrid", font_scale=1.5, rc={"lines.linewidth": 2.5, "lines.markersize": 10})
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_log, label='train')
        plt.plot(self.val_loss_log, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=300, bbox_inches='tight')
    
    def predict(self, x, numpy=False):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input data as a torch.Tensor
        numpy: bool
            If True, return numpy array
        """

        self.Net.eval()
        with torch.no_grad(): # turn off gradients computation
            out = self.Net(x)
            # compute prob
            out = torch.nn.functional.softmax(out, dim=1)
            # get the class
            out = torch.argmax(out, dim=1)

        print(f"Output shape: {out.shape}")
        if numpy:
            out = out.detach().cpu().numpy()

        else:
            return out
        
    def get_weights(self, numpy=True):
        dict_weights = {}
        names = self.Net.state_dict().keys()
        print (names)
        if not numpy:
            for name in names:
                dict_weights[name] = self.Net.state_dict()[name]
        else:
            for name in names:
                dict_weights[name] = self.Net.state_dict()[name].detach().cpu().numpy()

        return dict_weights

    def save_state_dict(self, path):
        """
        Save the model state dict in the path
        """

        if path.split('.')[-1] != 'torch':
            path = path + '.torch'
        print (f"Saving model to {path}")
        net_state_dict = self.Net.state_dict()
        torch.save(net_state_dict, path)

    def load_state_dict(self, path):
        """
        Load the model state dict from the path
        """
        
        if path.split('.')[-1] != 'torch':
            path = path + '.torch'
        
        # check if path exists
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        net_state_dict = torch.load(path)
        # Update the network parameters
        self.Net.load_state_dict(net_state_dict)

    def save_optimizer_state(self, path):
        """
        Save the optimizer state dict in the path
        """

        if path.split('.')[-1] != 'torch':
            path = path + '.torch'

        ### Save the self.optimizer state
        torch.save(self.optimizer.state_dict(), path)

    def load_optimizer_state(self, path):
        """
        Load the optimizer state dict from the path
        """

        if path.split('.')[-1] != 'torch':
            path = path + '.torch'
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        ### Reload the optimizer state
        opt_state_dict = torch.load(path)
        self.optimizer.load_state_dict(opt_state_dict)

    def _accuracy (self,all_outputs, all_labels):
        """Compute accuracy"""
        # the output doesnt comppute softmax, so we need to do it
        probs = torch.nn.functional.softmax(all_outputs, dim=1)
        all_output_classes = torch.argmax(probs, dim=1).detach().cpu().numpy()
        all_labels = all_labels.detach().cpu().numpy()
    
        # compute accuracy
        test_accuracy = accuracy_score(all_labels, all_output_classes)
        print(f"TEST ACCURACY: {test_accuracy:.2f}%")

        return test_accuracy

    def _recall_precision (self, all_outputs, all_labels):
        """Compute recall and precision"""
        probs = torch.nn.functional.softmax(all_outputs, dim=1)
        all_output_classes = torch.argmax(probs, dim=1).detach().cpu().numpy()
        all_labels = all_labels.detach().cpu().numpy()

        recall = recall_score(all_labels, all_output_classes, average='macro')
        precision = precision_score(all_labels, all_output_classes, average='macro')

        print(f"TEST RECALL: {recall:.2f}%")
        print(f"TEST PRECISION: {precision:.2f}%")

        return recall, precision


    def test (self, test_dataloader):
        """Test the model on the test set"""
        all_outputs = []
        all_labels = []
        self.Net.eval() # Evaluation mode (e.g. disable dropout)
        with torch.no_grad(): # Disable gradient tracking
            for sample_batched in tqdm(test_dataloader):
                # Move data to device
                x_batch = [x.to(self.device) for x in sample_batched[0] ]

                label_batch = sample_batched[1].to(self.device)
                # Forward pass
                out = self.Net(x_batch)
                # Save outputs and labels
                all_outputs.append(out)
                all_labels.append(label_batch)

        # Concatenate all the outputs and labels in a single tensor
        all_outputs = torch.cat(all_outputs)
        all_labels  = torch.cat(all_labels)

        all_probs = torch.nn.functional.softmax(all_outputs, dim=1)
        all_predictions = torch.argmax(all_probs, dim=1)

        test_acc = self._accuracy(all_outputs, all_labels)

        all_labels = all_labels.detach().cpu().numpy()
        all_outputs = all_outputs.detach().cpu().numpy()
        all_predictions = all_predictions.detach().cpu().numpy()

        # metrics
        recall = recall_score(all_labels, all_predictions, average='macro') 
        precision = precision_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        results = pd.DataFrame()
        results['accuracy'] = [test_acc]
        results['recall'] = [recall]
        results['precision'] = [precision]
        results['f1'] = [f1]

        # classificaiton report
        from sklearn.metrics import classification_report
        report = classification_report(all_labels, all_predictions, output_dict=True)
        report = pd.DataFrame(report).transpose()

        # confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_predictions, normalize='true')
        cm = pd.DataFrame(cm)

        # Top-k Accuracy classification score.
        from sklearn.metrics import top_k_accuracy_score
        top2_acc = top_k_accuracy_score(all_labels, all_outputs, k=2)
        top3_acc = top_k_accuracy_score(all_labels, all_outputs, k=3)

        r = {
            'metrics': results,
            'report': report,
            'cm': cm,
            'top2_acc': top2_acc,
            'top3_acc': top3_acc
        } 
        return r
        


    def get_prob_distribution (self, test_dataloader):
        """
        Get the probability distribution of the test set
        """

        all_inputs = []
        all_outputs = []
        all_labels = []
        self.Net.eval()
        with torch.no_grad():
            for sample_batched in tqdm(test_dataloader):
                x_batch = [x.to(self.device) for x in sample_batched[0] ]

                label_batch = sample_batched[1].to(self.device)
                out = self.Net(x_batch)
                all_inputs.append(x_batch)
                all_outputs.append(out)
                all_labels.append(label_batch)

        all_labels = torch.cat(all_labels)
        all_outputs = torch.cat(all_outputs)
    
        probs = torch.nn.functional.softmax(all_outputs, dim=1)
        return probs, all_labels