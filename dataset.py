import os
import pathlib
import numpy as np
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt
from IPython import display
from utils.input import get_waveform_and_label
from utils.plot import plot_spectrogram,\
                        plot_features
from utils.preprocessing import get_spectrogram,\
                               get_spectrogram_and_label_id,\
                               get_log_mel_features_and_label_id,\
                               get_mfcc_and_label_id
from tqdm import tqdm


class DataLoader:
    """
    Class to load the dataset.
    """
    
    def __init__(self,
                 path: str,):
        """
        Initialize the DataLoader class.
        
        Parameters
        ----------
        path : str
            Path to the dataset.
        """
        self.DATASET_PATH = path
        self.DATASET_PATH = self.DATASET_PATH if isinstance(self.DATASET_PATH, str) else self.DATASET_PATH[0]

        self.data_dir = pathlib.Path(self.DATASET_PATH)


    @classmethod
    def empty(cls):
        """
        Initialize an empty DataLoader class.
        """
        obj = cls(path='')
        return obj


    def download_data(self,
                      url: str = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
                      dir: str = 'DATA'):
        """
        Download the dataset from the given url
        and extract it to the given directory.
        
        Parameters
        ----------
        url : str
            URL to the dataset.
            Default: 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
        dir : str
            Directory to extract the dataset.
            Default: 'DATA'
        """
        tf.keras.utils.get_file(
            fname='speech_commands_v0.02.tar.gz',
            origin=url,
            extract=True,
            cache_subdir=dir,
            cache_dir='.',
            archive_format='tar.gz'
        )


    def get_commands(self):
        """
        Get the commands in the dataset.
        
        Returns
        -------
        commands : np.array
            List of commands.
        """
        commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))
        commands = commands[~np.isin(commands, ['README.md', 'testing_list.txt',
                                                '.DS_Store', 'validation_list.txt',
                                                '_background_noise_', 'LICENSE'])]
        return commands


    def get_filenames(self):
        """
        Get the filenames of the dataset.
        
        Returns
        -------
        filenames : list
            List of filenames.
        """
        # Download the dataset if it doesn't exist.
        if not os.path.exists(self.data_dir):
            self.download_data()

        filenames = tf.io.gfile.glob(str(self.data_dir) + '/*/*')
        filenames = [filename for filename in filenames if '_background_noise_' not in filename]
        return filenames


    def split_data(self,
                   filenames: list,
                   test_ratio:float=0.15,
                   val_ratio:float=0.05,
                   shuffle:bool=True,
                   seed:int=42,
                   verbose:int=0):
        """
        Split the dataset into train, validation and test sets.
        
        Parameters
        ----------
        filenames : list
            List of filenames.
        test_ratio : float
            Ratio of test set.
            Default: 0.15
        val_ratio : float
            Ratio of validation set.
            Default: 0.05
        shuffle : bool
            Whether to shuffle the data.
            Default: True
        seed : int
            Random seed.
            Default: 42
        verbose : int
            Verbosity mode.
            Default: 0
            
        Returns
        -------
        train_files : list
            List of train files.
        val_files : list
            List of validation files.
        test_files : list
            List of test files.
            
        Raises
        ------
        AssertionError
            If the sum of test_ratio and val_ratio is greater than 1.
        """
        assert test_ratio + val_ratio < 1, 'The sum of test_ratio and val_ratio must be less than 1.'
        
        # Get the number of samples for each set.
        N_train = int((1 - test_ratio - val_ratio) * len(filenames))
        N_val = int(val_ratio * len(filenames))

        # Shuffle the data.
        if shuffle:
            tf.random.set_seed(seed)
            filenames = tf.random.shuffle(filenames)

        # Split the data.
        train_files = filenames[:N_train]
        val_files = filenames[N_train:N_train + N_val]
        test_files = filenames[N_train + N_val:]

        verbosity_actions = {
            0: lambda: None,
            1: lambda: print('Splitted data into training, validation and testing sets'),
            2: lambda: print(f'Number of samples used for training:\t{self.num_train}\n' \
                             f'Number of samples used for validation:\t{self.num_val}\n' \
                             f'Number of samples used for testing:\t{self.num_test}')
        }

        for level, action in verbosity_actions.items():
            if verbose > level:
                action()

        return train_files, val_files, test_files
    
    def get_waveform_data (self,
                           filenames: list,
                            ):
        """
        Get waveform dataset as numpy arrays.
        This method creates a dataset of waveforms from a list of filenames.
        
        Parameters
        ----------
        filenames : list
            List of filenames.
            
        Returns
        -------
        data: list of numpy arrays
            List of waveforms.
        labels: list of numpy arrays
            List of labels for each waveform.
        """  
        data = []
        labels = []

        # parallelize the process
        from multiprocessing import Pool
        with Pool() as p:
            for waveform, label in tqdm(p.imap(get_waveform_and_label, filenames)):
                data.append(waveform.numpy())
                labels.append(label.numpy().decode("utf-8"))

        return data, labels


    def get_waveform_ds(self,
                        filenames: list,
                        AUTOTUNE: tf.data.experimental.AUTOTUNE = tf.data.experimental.AUTOTUNE):
        """
        Get waveform dataset.
        This method creates a dataset of waveforms from a list of filenames.
        
        Parameters
        ----------
        filenames : list
            List of filenames.
        AUTOTUNE : tf.data.experimental.AUTOTUNE
            Number of files to process in parallel.
            Default: tf.data.experimental.AUTOTUNE
            
        Returns
        -------
        waveform_ds : tf.data.Dataset
            Dataset of waveforms.
        """

        # Create a dataset of the filenames.
        files_ds = tf.data.Dataset.from_tensor_slices(filenames)

        # Create a dataset of the waveforms.
        waveform_ds = files_ds.map(
            map_func=get_waveform_and_label,
            num_parallel_calls=AUTOTUNE
        )
        
        return waveform_ds
    

    def get_spectrogram_STFT_ds(self,
                                waveform_ds: tf.data.Dataset,
                                commands,
                                AUTOTUNE: tf.data.experimental.AUTOTUNE = tf.data.experimental.AUTOTUNE):
        """
        Get spectrogram dataset using STFT.
        
        Parameters
        ----------
        waveform_ds : tf.data.Dataset
            Dataset of waveforms.
        commands : np.array
            List of commands.
        AUTOTUNE : tf.data.experimental.AUTOTUNE
            Number of files to process in parallel.
            Default: tf.data.experimental.AUTOTUNE
            
        Returns
        -------
        spectrogram_ds : tf.data.Dataset
            Dataset of spectrograms.
        """
        #spectrogram_ds = waveform_ds.map(
        #    get_spectrogram_and_label_id,
        #    num_parallel_calls=AUTOTUNE
        #)
        
        spectrogram_ds = waveform_ds.map(
            lambda audio, label: get_spectrogram_and_label_id(audio, label, commands),
            num_parallel_calls=AUTOTUNE
        )
                           
        return spectrogram_ds
    
    
    def get_spectrogram_logmel_ds(self,
                                  waveform_ds: tf.data.Dataset,
                                  commands,
                                  AUTOTUNE: tf.data.experimental.AUTOTUNE = tf.data.experimental.AUTOTUNE):
        """
        Get spectrogram dataset using log mel spectrogram.
        
        Parameters
        ----------
        waveform_ds : tf.data.Dataset
            Dataset of waveforms.
        AUTOTUNE : tf.data.experimental.AUTOTUNE
            Number of files to process in parallel.
            Default: tf.data.experimental.AUTOTUNE
    
        Returns
        -------
        spectrogram_ds : tf.data.Dataset
            Dataset of spectrograms.
        """
        spectrogram_ds = waveform_ds.map(
            map_func=lambda audio, label: get_log_mel_features_and_label_id(audio, label, commands),
            num_parallel_calls=AUTOTUNE
        )

        return spectrogram_ds
    
    
    def get_spectrogram_mfcc_ds(self,
                                waveform_ds: tf.data.Dataset,
                                commands,
                                AUTOTUNE: tf.data.experimental.AUTOTUNE = tf.data.experimental.AUTOTUNE):
        """
        Get spectrogram dataset using MFCC.
        
        Parameters
        ----------
        waveform_ds : tf.data.Dataset
            Dataset of waveforms.
        commands : np.array
            List of commands.
        AUTOTUNE : tf.data.experimental.AUTOTUNE
            Number of files to process in parallel.
            Default: tf.data.experimental.AUTOTUNE
            
        Returns
        -------
        spectrogram_ds : tf.data.Dataset
            Dataset of spectrograms.
        """
        spectrogram_ds = waveform_ds.map(
            map_func=lambda audio, label: get_mfcc_and_label_id(audio, label, commands),
            num_parallel_calls=AUTOTUNE
        )
        
        return spectrogram_ds


class DataVisualizer:
    """
    Class to visualize the waveform and spectrogram dataset.
    """
    def __init__(self,
                 waveform_ds,
                 spectrogram_ds,
                 commands):
        """
        Initialize the WaveformProcessor class.
        
        Parameters
        ----------
        waveform_ds : tf.data.Dataset
            Dataset of waveforms.
        spectrogram_ds : tf.data.Dataset
            Dataset of spectrograms.
        commands : list
            List of commands.
        """
        self.waveform_ds = waveform_ds
        self.spectrogram_ds = spectrogram_ds
        self.commands = commands

    def plot_waveform_example(self,
                              rows:int=3,
                              cols:int=3,
                              return_fig:bool=False,
                              show:bool=True,
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
        show : bool
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
        _, axes = plt.subplots(
            nrows=rows,
            ncols=cols,
            figsize=(10, 12),
            sharex=True,
            sharey=True
        )

        # Plot the examples.
        for i, (audio, label) in enumerate(self.waveform_ds.take(n)):
            r = i // cols   
            c = i % cols
            ax = axes[r][c]
            ax.plot(audio.numpy())
            ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
            label = label.numpy().decode('utf-8')
            ax.set_title(label)

        if show:
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
                        show:bool=True,
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
        show : bool
            Whether to display the audio.
        save : bool
            Whether to save the audio.
        dir : str
            The directory to save the audio.
        name : str
            The name of the audio file.
            
        Returns
        -------
        waveform.numpy() : np.array
            The waveform.
        rate : int
            The rate of the audio.
        """

        # Get the waveform and label.        
        for waveform, label in self.waveform_ds.skip(n).take(1):
            label = label.numpy().decode('utf-8')
            spectrogram = get_spectrogram(waveform)

        if show:
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


    def plot_spectrogram_example(self,
                                 rows:int=3,
                                 cols:int=3,
                                 figsize:tuple=None,
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
        if figsize is None:
            figsize = (4*cols, 4*rows)
        _, axes = plt.subplots(rows, cols, figsize=figsize)

        for i, (spectrogram, label_id) in enumerate(self.spectrogram_ds.take(n)):
            print (spectrogram.numpy().shape)
            r = i // cols
            c = i % cols
            if rows>1 and cols>1:
                ax = axes[r][c]
            elif rows>1 and cols==1:
                ax = axes[r]
            elif rows==1 and cols>1:
                ax = axes[c]
            plot_features(spectrogram.numpy(), ax)
            ax.set_title(self.commands[label_id.numpy()])
            ax.axis('off')

        if display:
            plt.show()

            
        if savefig:
            if filename is None:
                filename = 'spectrogram_examples.pdf'
            plt.savefig(
                os.path.join(
                    dir,
                    filename
                ),
                bbox_inches='tight'
            )
            
        if return_fig:
            return plt.gcf()


class DatasetBuilder:
    """
    Class to build the dataset.
    """
    def __init__(self,
                 commands:list,
                 train_filenames,
                 test_filenames,
                 val_filenames=None,
                 batch_size:int=64,
                 buffer_size:int=10000,
                 AUTOTUNE=tf.data.experimental.AUTOTUNE,
                 method:str='log_mel',
                 ):
        """
        Initialize the DatasetBuilder class.

        Parameters
        ----------
        train_filenames : tensorflow.python.framework.ops.EagerTensor
            The list of training filenames.
        test_filenames : tensorflow.python.framework.ops.EagerTensor
            The list of testing filenames.
        val_filenames : tensorflow.python.framework.ops.EagerTensor
            The list of validation filenames.
            Default is None.
        batch_size : int
            The batch size.
            Default is 64.
        buffer_size : int
            The buffer size.
            Default is 10000.
        AUTOTUNE : tf.data.experimental.AUTOTUNE
            The number of parallel calls.
            Default is tf.data.experimental.AUTOTUNE.
        method : str
            The method to use for preprocessing.
            Possible values are 'log_mel', 'mfcc' and 'STFT'.            
        """
        self.train_filenames = train_filenames
        self.test_filenames = test_filenames
        self.val_filenames = val_filenames
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.AUTOTUNE = AUTOTUNE
        self.commands = commands

        assert method in ['log_mel', 'mfcc', 'STFT'], 'method must be either "log_mel", "mfcc" or "STFT"'
        self.method = method


    def _add_channels(
        self,
        data:tf.Tensor,
        label:tf.Tensor
    ):
        """
        Add a channel dimension to the data.
        
        Parameters
        ----------
        data : tf.Tensor
            The data (either waveform or spectrogram).
        label : tf.Tensor
            The label.
            
        Returns
        -------
        data : tf.Tensor
            The data with a channel dimension.
        label : tf.Tensor
            The label.
        """
        data = tf.expand_dims(
            data,
            axis=-1
        )
        
        return data, label


    def preprocess_dataset_waveform(
        self,
        AUTOTUNE=None,
        batch_size:int=None,
    ):
        """
        Preprocess the dataset and create the waveform dataset.
        
        Parameters
        ----------
        AUTOTUNE : tf.data.experimental.AUTOTUNE
            The number of parallel calls.
            Default is tf.data.experimental.AUTOTUNE.
        batch_size : int
            The batch size.
            Default is None.
        
        Returns
        -------
        train_ds : tensorflow.python.data.ops.dataset_ops.BatchDataset
            The training dataset.
        test_ds : tensorflow.python.data.ops.dataset_ops.BatchDataset
            The testing dataset.
        val_ds : tensorflow.python.data.ops.dataset_ops.BatchDataset
            The validation dataset.
        """
        
        
        if AUTOTUNE is None:
            AUTOTUNE = self.AUTOTUNE
            
        if batch_size is None:
            batch_size = self.batch_size
            
        datasets = []

        for files in [self.train_filenames, self.test_filenames, self.val_filenames]:
            if files is None:
                continue
            files_ds = tf.data.Dataset.from_tensor_slices(files)
            
            # Create a dataset of the waveforms.
            waveform_ds = files_ds.map(
                map_func=get_waveform_and_label,
                num_parallel_calls=AUTOTUNE
            )
            
            datasets.append(waveform_ds)

        self.train_ds = datasets[0].cache().shuffle(self.buffer_size).batch(self.batch_size).prefetch(AUTOTUNE)
        self.test_ds = datasets[1].cache().batch(self.batch_size).prefetch(AUTOTUNE)
        self.val_ds = datasets[2].cache().batch(self.batch_size).prefetch(AUTOTUNE)
                
        return self.train_ds, self.test_ds, self.val_ds


    def preprocess_dataset_spectrogram(
        self,
        method:str=None,
        AUTOTUNE=None,
        batch_size:int=None,
    ):
        """
        Preprocess the dataset.
        
        Parameters
        ----------
        method : str
            The method to use for preprocessing, if you want 
            to overwrite the previous one.
            Possible values are 'log_mel', 'mfcc' and 'STFT'.
            Default is None.
        AUTOTUNE : tf.data.experimental.AUTOTUNE
            The number of parallel calls.
            Default is tf.data.experimental.AUTOTUNE.
        batch_size : int
            The batch size.
            Default is None.

        Returns
        -------
        train_ds : tensorflow.python.data.ops.dataset_ops.BatchDataset
            The training dataset.
        test_ds : tensorflow.python.data.ops.dataset_ops.BatchDataset
            The testing dataset.
        val_ds : tensorflow.python.data.ops.dataset_ops.BatchDataset
            The validation dataset.
        """

        if method is None:
            method = self.method
        else:
            assert method in ['log_mel', 'mfcc', 'STFT'], 'method must be either "log_mel", "mfcc" or "STFT"'

        if AUTOTUNE is None:
            AUTOTUNE = self.AUTOTUNE
            
        if batch_size is None:
            batch_size = self.batch_size
            
        datasets = []

        for files in [self.train_filenames, self.test_filenames, self.val_filenames]:
            if files is None:
                continue
            files_ds = tf.data.Dataset.from_tensor_slices(files)
            
            # Create a dataset of the waveforms.
            waveform_ds = files_ds.map(
                map_func=get_waveform_and_label,
                num_parallel_calls=AUTOTUNE
            )
    
            # Create a dataset of the spectrograms.
            if method=='log_mel':
                spectrogram_ds = waveform_ds.map(
                    map_func=lambda audio, label: get_log_mel_features_and_label_id(audio, label, self.commands),
                    num_parallel_calls=AUTOTUNE
                )

            elif method=='mfcc':
                spectrogram_ds = waveform_ds.map(
                    map_func=lambda audio, label: get_mfcc_and_label_id(audio, label, self.commands),
                    num_parallel_calls=AUTOTUNE
                )
                
            else:
                spectrogram_ds = waveform_ds.map(
                    lambda audio, label: get_spectrogram_and_label_id(audio, label, self.commands),
                    num_parallel_calls=AUTOTUNE
                )
                
            # Add a channel dimension to the spectrograms.
            spectrogram_ds = spectrogram_ds.map(
                map_func=self._add_channels,
                num_parallel_calls=AUTOTUNE
            )

            datasets.append(spectrogram_ds)

        self.train_ds = datasets[0].cache().shuffle(self.buffer_size).batch(self.batch_size).prefetch(AUTOTUNE)
        self.test_ds = datasets[1].cache().batch(self.batch_size).prefetch(AUTOTUNE)
        self.val_ds = datasets[2].cache().batch(self.batch_size).prefetch(AUTOTUNE)
                
        return self.train_ds, self.test_ds, self.val_ds