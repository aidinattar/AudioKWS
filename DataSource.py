"""
Class to manage the data source.
"""
import os
import copy
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import soundfile as sf
from IPython import display
from utils.input import get_waveform_and_label, get_spectrogram, plot_spectrogram

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class DataSource(object):
    """A class to manage the data source."""
    def __init__(self,
                 path:str,
                 batch_size:int,
                 shuffle_buffer_size:int,
                 verbose:int=0):
        """
        Initialize the class.
        
        Parameters
        ----------
        path : str
            The path to the data.
        batch_size : int
            The size of the batch to use.
        shuffle_buffer_size : int
            The size of the shuffle buffer.
        """
        # Set the attributes.
        self.DATASET_PATH = path
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.verbose = verbose

        # Get the data.
        self.data_dir = pathlib.Path(self.DATASET_PATH)
        # If the data does not exist, download it.
        if not self.data_dir.exists():
            tf.keras.utils.get_file(
                'speech_commands_v0.02.tar.gz',
                origin='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
                extract=True,
                cache_dir='.', cache_subdir='DATA')

        # Get the commands.
        commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))
        # The dataset contains a README file, and some others which are not commands.
        self.commands = commands[~np.isin(commands, ['README.md', 'testing_list.txt',
                                                     '.DS_Store', 'validation_list.txt',
                                                     '_background_noise_', 'LICENSE'])]

        # Automatically determine the optimal number of elements to prefetch or cache during
        # the execution of a dataset pipeline, based on available system resources and the
        # characteristics of the data
        self.AUTOTUNE = tf.data.AUTOTUNE


    def print_commands(self):
        """
        Prints the commands.
        """
        print('Commands:\n', self.commands)


    def get_data(self):
        """
        Gets the data from the data directory.
        """
        # Get the filenames.
        filenames = tf.io.gfile.glob(str(self.data_dir) + '/*/*')
        # Remove the background noise files.
        filenames = [filename for filename in filenames if '_background_noise_' not in filename]
        # Shuffle the filenames.
        self.filenames = tf.random.shuffle(filenames)
        # Get the number of samples.
        self.num_samples = len(filenames)


    def print_example(self):
        """
        Prints the data.
        """
        print('Number of total examples:', self.num_samples)
        print('Number of examples per label:',
              len(tf.io.gfile.listdir(str(self.data_dir/self.commands[0]))))
        print('Example file tensor:', self.filenames[0])


    def train_test_split(self, test_ratio=.8, val_ratio=.05):
        """
        Splits the data into training, testing and validation sets.
        
        Parameters
        ----------
        test_ratio : float
            The ratio of the training set.
        val_ratio : float
            The ratio of the validation set. 
        """
        # Get the number of samples for each set.
        N_train = int((1-test_ratio-val_ratio) * self.num_samples)
        N_val = int(val_ratio * self.num_samples)

        # Split the data.
        self.train_files = self.filenames[:N_train]
        self.val_files = self.filenames[N_train:N_train+N_val]
        self.test_files = self.filenames[N_train+N_val:]
        
        # Get the number of samples for each set.
        self.num_train = len(self.train_files)
        self.num_val = len(self.val_files)
        self.num_test = len(self.test_files)
        
        verbosity_actions = {
            0: lambda: None,
            1: lambda: print('Splitted data into training, validation and testing sets'),
            2: lambda: print(f'Number of samples used for training:\t{self.num_train}\n' \
                             f'Number of samples used for validation:\t{self.num_val}\n' \
                             f'Number of samples used for testing:\t{self.num_test}')
        }

        for level, action in verbosity_actions.items():
            if self.verbose > level:
                action()


    def print_split(self):
        """Print the split."""
        print('Training set size', len(self.train_files))
        print('Validation set size', len(self.val_files))
        print('Test set size', len(self.test_files))


    def get_waveform_ds(self):
        """
        Get waveform dataset.
        """

        # Create a dataset of the filenames.
        files_ds = tf.data.Dataset.from_tensor_slices(self.train_files)

        # Create a dataset of the waveforms.
        self.waveform_ds = files_ds.map(
            map_func=get_waveform_and_label,
            num_parallel_calls=self.AUTOTUNE)


    def plot_waveform_example(self,
                              rows:int=3,
                              cols:int=3,
                              return_fig:bool=False,
                              display:bool=True,
                              savefig:bool=False,
                              dir:str='figures',
                              filename:str=None):
        """
        Plot a grid of waveform examples.
        
        Parameters
        ----------
        rows : int
            The number of rows of the grid.
        cols : int
            The number of columns of the grid.
        return_fig : bool
            Whether to return the figure.
        display : bool
            Whether to show the figure.
        savefig : bool
            Whether to save the figure.
        dir : str
            The directory to save the figure.
        filename : str
            The filename of the figure.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure with the waveform examples.
        """
        # Get the number of examples.
        n = rows * cols
        _, axes = plt.subplots(rows, cols, figsize=(10, 12))

        # Plot the examples.
        for i, (audio, label) in enumerate(self.waveform_ds.take(n)):
            # Get the row and column.
            r = i // cols
            c = i % cols
            # Get the corresponding axis.
            ax = axes[r][c]
            ax.plot(audio.numpy())
            ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
            label = label.numpy().decode('utf-8')
            ax.set_title(label)

        if display:
            plt.show()
        
        if savefig:
            if filename is None:
                filename = 'waveform_examples.png'
            plt.savefig(
                os.path.join(
                    dir,
                    filename
                ),
            )
            
        if return_fig:
            return plt.gcf()


    def listen_waveform(self,
                        n:int=0,
                        rate:int=16000,
                        return_audio:bool=False,
                        display:bool=True,
                        save:bool=False,
                        dir:str='figures',
                        name:str=None):
        """
        Listen to a selected waveform.
        
        Parameters
        ----------
        n : int
            The index of the waveform to listen to.
        rate : int
            The rate of the audio.
        return_audio : bool
            Whether to return the audio.
        display : bool
            Whether to display the audio.
        save : bool
            Whether to save the audio.
        dir : str
            The directory to save the audio.
        name : str
            The name of the audio file.
            
        Returns
        -------
        waveform : numpy.ndarray
            The waveform.
        """

        # Get the waveform and label.        
        for waveform, label in self.waveform_ds.skip(n).take(1):
            label = label.numpy().decode('utf-8')
            spectrogram = get_spectrogram(waveform)

        if display:
            print('Label:', label)
            print('Waveform shape:', waveform.shape)
            print('Spectrogram shape:', spectrogram.shape)
            print('Audio playback')
            display.display(display.Audio(waveform, rate=rate))
        
        if save:
            if name is None:
                name = f'audio_Label{label}.wav'
            sf.write(
                os.path.join(
                    dir,
                    name
                ),
                waveform.numpy(),
                rate
            )

        if return_audio:
            return waveform.numpy(), rate


    def plot_spect(self,
                   n:int=0,
                   return_fig:bool=False,
                   display:bool=True,
                   savefig:bool=False,
                   dir:str='figures',
                   name:str=None):
        """
        Plot the spectrogram.
        
        Parameters
        ----------
        n : int
            The index of the waveform to plot.
        return_fig : bool
            Whether to return the figure.
        display : bool
            Whether to display the figure.
        savefig : bool
            Whether to save the figure.
        dir : str
            The directory to save the figure.
        name : str
            The name of the figure.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure with the spectrogram.
        """
        # Get the waveform and label.
        for waveform, label in self.waveform_ds.skip(n).take(1):
            label = label.numpy().decode('utf-8')
            spectrogram = get_spectrogram(waveform)

        if display:
            print('Label:', label)
            print('Waveform shape:', waveform.shape)
            print('Spectrogram shape:', spectrogram.shape)

            _, axes = plt.subplots(2, figsize=(12, 8))
            timescale = np.arange(waveform.shape[0])
            axes[0].plot(timescale, waveform.numpy())
            axes[0].set_title('Waveform')
            axes[0].set_xlim([0, 16000])

            plot_spectrogram(spectrogram.numpy(), axes[1])
            axes[1].set_title('Spectrogram')
            plt.show()
            
        if savefig:
            if name is None:
                name = f'spectrogram_Label{label}.png'
            plt.savefig(
                os.path.join(
                    dir,
                    name
                ),
            )
            
        if return_fig:
            return plt.gcf()


    def get_spectrogram_ds(self):
        """
        Get the spectrogram dataset.
        """
        def get_spectrogram_and_label_id(audio, label):
            spectrogram = get_spectrogram(audio)
            label_id = tf.argmax(label == self.commands)
            return spectrogram, label_id

        self.spectrogram_ds = self.waveform_ds.map(
            get_spectrogram_and_label_id,
            num_parallel_calls=self.AUTOTUNE)


    def plot_spectrogram_example(self,
                              rows:int=3,
                              cols:int=3,
                              return_fig:bool=False,
                              display:bool=True,
                              savefig:bool=False,
                              dir:str='figures',
                              filename:str=None):
        """
        Plot an example of some spectrogram in a grid.

        Parameters
        ----------
        rows : int
            The number of rows of the grid.
        cols : int
            The number of columns of the grid.
        return_fig : bool
            Whether to return the figure.
        display : bool
            Whether to show the figure.
        savefig : bool
            Whether to save the figure.
        dir : str
            The directory to save the figure.
        filename : str
            The filename of the figure.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure with the spectrogram examples.
        """
        # Get the number of examples.
        n = rows * cols
        _, axes = plt.subplots(rows, cols, figsize=(10, 12))

        for i, (spectrogram, label_id) in enumerate(self.spectrogram_ds.take(n)):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            plot_spectrogram(spectrogram.numpy(), ax)
            ax.set_title(self.commands[label_id.numpy()])
            ax.axis('off')

        if display:
            plt.show()
            
        if savefig:
            if filename is None:
                filename = 'spectrogram_examples.png'
            plt.savefig(
                os.path.join(
                    dir,
                    filename
                ),
            )
            
        if return_fig:
            return plt.gcf()


    def _get_spectrogram_and_label_id(self,
                                      audio,
                                      label):
        """
        Get the spectrogram and label id.
        
        Parameters
        ----------
        audio : numpy.ndarray
            The audio.
        label : numpy.ndarray
            The label.
            
        Returns
        -------
        spectrogram : numpy.ndarray
            The spectrogram.
        label_id : numpy.ndarray
            The label id.
        """
        spectrogram = get_spectrogram(audio)
        label_id = tf.argmax(label == self.commands)
        return spectrogram, label_id



    def _preprocess_dataset(self,
                            files):
        """
        Preprocess the dataset.
        
        Parameters
        ----------
        files : list
            The list of files.
            
        Returns
        -------
        output_ds : tensorflow.python.data.ops.dataset_ops.MapDataset
            The preprocessed dataset.
        """
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(
            map_func=get_waveform_and_label,
            num_parallel_calls=self.AUTOTUNE)
        output_ds = output_ds.map(
            map_func=self._get_spectrogram_and_label_id,
            num_parallel_calls=self.AUTOTUNE)
        return output_ds


    def define_ds(self):
        """
        Define the dataset.
        """
        self.train_ds = self.spectrogram_ds
        self.val_ds = self._preprocess_dataset(self.val_files)
        self.test_ds = self._preprocess_dataset(self.test_files)


    def batch_ds(self):
        """
        Batch the dataset.
        """
        train_ds = self.train_ds.batch(self.batch_size)
        val_ds  = self.val_ds.batch(self.batch_size)

        self.train_ds = train_ds.cache().prefetch(self.AUTOTUNE)
        self.val_ds = val_ds.cache().prefetch(self.AUTOTUNE)
        
        
    def copy(self):
        """
        Copy the dataset.
        """
        return copy.deepcopy(self)