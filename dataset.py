import os
import pathlib
import numpy as np
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt
from IPython import display
from utils.input import get_waveform_and_label,\
                        plot_spectrogram,\
                        get_spectrogram,\
                        get_spectrogram_and_label_id,\
                        log_mel_feature_extraction


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
                                AUTOTUNE: tf.data.experimental.AUTOTUNE = tf.data.experimental.AUTOTUNE):
        """
        Get spectrogram dataset using STFT.
        
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
            get_spectrogram_and_label_id,
            num_parallel_calls=self.AUTOTUNE
        )
                           
        return spectrogram_ds
    
    
    def get_spectrogram_logmel_ds(self,
                                  waveform_ds: tf.data.Dataset,
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
            map_func=lambda audio, label: (log_mel_feature_extraction(audio), label),
            num_parallel_calls=AUTOTUNE
        )
        
        return spectrogram_ds


class DataVisualizer:
    """
    Class to visualize the waveform and spectrogram dataset.
    """
    def __init__(self,
                 waveform_ds,
                 spectrogram_ds):
        """
        Initialize the WaveformProcessor class.
        
        Parameters
        ----------
        waveform_ds : tf.data.Dataset
            Dataset of waveforms.
        spectrogram_ds : tf.data.Dataset
            Dataset of spectrograms.
        """
        self.waveform_ds = waveform_ds
        self.spectrogram_ds = spectrogram_ds

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
            display.Audio(waveform, rate=rate)
        
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


    def listen_spectrogram(self,
                            n:int=0,
                            rate:int=16000,
                            return_audio:bool=False,
                            show:bool=True,
                            save:bool=False,
                            dir:str='figures',
                            name:str=None):
        """
        Listen to a selected spectrogram.

        Parameters
        ----------
        n : int
            The index of the spectrogram to listen to.
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
        audio : np.array
            The audio.
        """
        for spectrogram, label in self.spectrogram_ds.skip(n).take(1):
            label = label.numpy().decode('utf-8')
        print('Label:', label)
        print('Spectrogram shape:', spectrogram.shape)
        print('Audio playback')
        
        if show:
            display.display(display.Audio(tf.squeeze(spectrogram).numpy()))
        
        if save:
            if name is None:
                name = f'audio_Label{label}.wav'
            sf.write(
                os.path.join(
                    dir,
                    name
                ),
                tf.squeeze(spectrogram).numpy(),
                rate
            )
            
        if return_audio:
            return tf.squeeze(spectrogram).numpy(), rate


class DatasetBuilder:
    """
    Class to build the dataset.
    """
    def __init__(self,
                 filenames:list):
        """
        Initialize the DatasetBuilder class.

        Parameters
        ----------
        filenames : list
            List of filenames.
        """
        self.filenames = filenames

    def _preprocess_dataset(self,
                            files,
                            method:str='log_mel',
                            AUTOTUNE=tf.data.experimental.AUTOTUNE
                            ):
        """
        Preprocess the dataset.
        
        Parameters
        ----------
        files : list
            The list of files.
        method : str
            The method to use for preprocessing.
            Possible values are 'log_mel' and 'STFT'.
            Default is 'log_mel'.
        AUTOTUNE : tf.data.experimental.AUTOTUNE
            The number of parallel calls.
            Default is tf.data.experimental.AUTOTUNE.
            
        Returns
        -------
        output_ds : tensorflow.python.data.ops.dataset_ops.MapDataset
            The preprocessed dataset.
        """
        assert method in ['log_mel', 'STFT'], 'method must be either "log_mel" or "STFT"'
        
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(
            map_func=get_waveform_and_label,
            num_parallel_calls=AUTOTUNE)
        
        if method=='log_mel':
            output_ds = output_ds.map(
                map_func=lambda audio, label: (log_mel_feature_extraction(audio), label),
                num_parallel_calls=AUTOTUNE
            )
        else:
            output_ds = output_ds.map(
                map_func=get_spectrogram_and_label_id,
                num_parallel_calls=AUTOTUNE
            )

        
        return output_ds
    
    
    def batch_dataset(self,
                      ds,
                      batch_size:int=32,
                      AUTOTUNE=tf.data.experimental.AUTOTUNE):
        """
        Batch the dataset and prefetch it.
        
        Parameters
        ----------
        ds : tensorflow.python.data.ops.dataset_ops.MapDataset
            The dataset to batch.
        batch_size : int
            The batch size.
            Default is 32.
        AUTOTUNE : tf.data.experimental.AUTOTUNE
            The number of parallel calls.
            Default is tf.data.experimental.AUTOTUNE.
            
        Returns
        -------
        output_ds : tensorflow.python.data.ops.dataset_ops.MapDataset
            The batched dataset.
        """
        output_ds = ds.batch(batch_size)
        output_ds = output_ds.cache().prefetch(AUTOTUNE)
        return output_ds
