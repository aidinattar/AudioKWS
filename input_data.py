'''
Class to manage the data source.
'''

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display
from input_utils import get_waveform_and_label, get_spectrogram, plot_spectrogram

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class DataSource(object):
    '''A class to manage the data source.'''
    def __init__(self, path, batch_size, shuffle_buffer_size):
        '''Initialize the class.'''
        self.DATASET_PATH = path
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size

        self.data_dir = pathlib.Path(self.DATASET_PATH)
        if not self.data_dir.exists():
            tf.keras.utils.get_file(
                'speech_commands_v0.02.tar.gz',
                origin='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
                extract=True,
                cache_dir='.', cache_subdir='DATA')

        commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))
        # The dataset contains a README file, which is not a command.
        self.commands = commands[~np.isin(commands, ['README.md', 'testing_list.txt',
                                                     '.DS_Store', 'validation_list.txt',
                                                     '_background_noise_', 'LICENSE'])]

        self.AUTOTUNE = tf.data.AUTOTUNE

    def print_commands(self):
        '''Prints the commands.'''
        print('Commands:\n', self.commands)

    def get_data(self):
        '''Gets the data.'''
        filenames = tf.io.gfile.glob(str(self.data_dir) + '/*/*')
        filenames = [filename for filename in filenames if '_background_noise_' not in filename]
        self.filenames = tf.random.shuffle(filenames)
        self.num_samples = len(filenames)

    def print_n(self):
        '''Prints the data.'''
        print('Number of total examples:', self.num_samples)
        print('Number of examples per label:',
              len(tf.io.gfile.listdir(str(self.data_dir/self.commands[0]))))
        print('Example file tensor:', self.filenames[0])

    def train_test_split(self, train_ratio=.8, val_ratio=.05):
        '''Split the data into training and testing.'''
        self.train_files = self.filenames[:int(train_ratio * self.num_samples)]
        self.val_files = self.filenames[int(train_ratio * self.num_samples): int(train_ratio * self.num_samples) + int(val_ratio * self.num_samples)]
        self.test_files = self.filenames[int(train_ratio * self.num_samples) + int(val_ratio * self.num_samples):]

    def print_split(self):
        '''Print the split.'''
        print('Training set size', len(self.train_files))
        print('Validation set size', len(self.val_files))
        print('Test set size', len(self.test_files))

    def get_waveform_ds(self):
        '''Get waveform dataset.'''
        files_ds = tf.data.Dataset.from_tensor_slices(self.train_files)

        self.waveform_ds = files_ds.map(
            map_func=get_waveform_and_label,
            num_parallel_calls=self.AUTOTUNE)

    def plot_waveform_example(self, rows=3, cols=3):
        '''Plot the waveform.'''
        n = rows * cols
        _, axes = plt.subplots(rows, cols, figsize=(10, 12))

        for i, (audio, label) in enumerate(self.waveform_ds.take(n)):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            ax.plot(audio.numpy())
            ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
            label = label.numpy().decode('utf-8')
            ax.set_title(label)

        plt.show()

    def listen_waveform(self, n=0):
        '''Listen to the waveform.'''
        for waveform, label in self.waveform_ds.skip(n).take(1):
            label = label.numpy().decode('utf-8')
            spectrogram = get_spectrogram(waveform)

        print('Label:', label)
        print('Waveform shape:', waveform.shape)
        print('Spectrogram shape:', spectrogram.shape)
        print('Audio playback')
        display.display(display.Audio(waveform, rate=16000))

    def plot_spect(self, n=0):
        '''Plot the spectrogram.'''
        for waveform, label in self.waveform_ds.skip(n).take(1):
            label = label.numpy().decode('utf-8')
            spectrogram = get_spectrogram(waveform)

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

    def get_spectrogram_ds(self):
        '''Get the spectrogram dataset.'''
        def get_spectrogram_and_label_id(audio, label):
            spectrogram = get_spectrogram(audio)
            label_id = tf.argmax(label == self.commands)
            return spectrogram, label_id

        self.spectrogram_ds = self.waveform_ds.map(
            get_spectrogram_and_label_id,
            num_parallel_calls=self.AUTOTUNE)

    def plot_spectrogram_example(self, rows=3, cols=3):
        '''Plot the spectrogram.'''
        n = rows * cols
        _, axes = plt.subplots(rows, cols, figsize=(10, 12))

        for i, (spectrogram, label_id) in enumerate(self.spectrogram_ds.take(n)):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            plot_spectrogram(spectrogram.numpy(), ax)
            ax.set_title(self.commands[label_id.numpy()])
            ax.axis('off')

        plt.show()


    def define_ds(self):
        '''Define the dataset.'''
        def preprocess_dataset(files):
            def get_spectrogram_and_label_id(audio, label):
                spectrogram = get_spectrogram(audio)
                label_id = tf.argmax(label == self.commands)
                return spectrogram, label_id

            files_ds = tf.data.Dataset.from_tensor_slices(files)
            output_ds = files_ds.map(
                map_func=get_waveform_and_label,
                num_parallel_calls=self.AUTOTUNE)
            output_ds = output_ds.map(
                map_func=get_spectrogram_and_label_id,
                num_parallel_calls=self.AUTOTUNE)
            return output_ds
        self.train_ds = self.spectrogram_ds
        self.val_ds = preprocess_dataset(self.val_files)
        self.test_ds = preprocess_dataset(self.test_files)

    def batch_ds(self):
        '''Batch the dataset.'''
        train_ds = self.train_ds.batch(self.batch_size)
        val_ds  = self.val_ds.batch(self.batch_size)

        self.train_ds = train_ds.cache().prefetch(self.AUTOTUNE)
        self.val_ds = val_ds.cache().prefetch(self.AUTOTUNE)