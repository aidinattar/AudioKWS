import os
import numpy as np
import tensorflow as tf
import librosa

def decode_audio(audio_binary: tf.Tensor) -> tf.Tensor:
    """
    Decode WAV-encoded audio files to `float32` tensors, normalized
    to the [-1.0, 1.0] range.
    
    Returns
    -------
    audio : tf.Tensor
        A `float32` tensor of audio.
    """
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)

def get_label(file_path: str) -> str:
    """
    Get the label from the file path.
    
    Parameters
    ----------
    file_path : str
        A file path.
    
    Returns
    -------
    label : str
        A label.
    """
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking
    # to enable this to work in a TensorFlow graph.
    return parts[-2]

def get_waveform_and_label(file_path: str):
    """
    Get the waveform and label from the file path.
    
    Parameters
    ----------
    file_path : str
        A file path.
        
    Returns
    -------
    waveform : tf.Tensor
        A waveform tensor.
    """
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    """
    Convert the waveform to a spectrogram via a STFT.

    Parameters
    ----------
    waveform : tf.Tensor
        A waveform tensor.

    Returns
    -------
    spectrogram : tf.Tensor
        A spectrogram tensor.
    """
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_spectrogram_and_label_id(audio,
                                 label,
                                 commands):
    """
    Convert the waveform to a spectrogram via a STFT.
    
    Parameters
    ----------
    audio : tf.Tensor
        A waveform tensor.
    label : tf.Tensor
        A label tensor.
    commands : np.ndarray
        A tensor of commands.
        
    Returns
    -------
    spectrogram : tf.Tensor
        A spectrogram tensor.
    label_id : tf.Tensor
        A label ID tensor.
    """
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


####################
# Mel Spectrogram  #
# Not working      #
# with tf.Dataset  #
####################
def log_mel_feature_extraction(waveform):
    """
    Perform log mel feature extraction on the waveform.
    
    Parameters
    ----------
    waveform : numpy.ndarray
        The waveform.
        
    Returns
    -------
    log_mel_features : numpy.ndarray
        The log mel features.
    """
    
    # Convert waveform to mono if it has multiple channels
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Compute the log mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=16000,
        n_fft=400,
        hop_length=160,
        n_mels=40,
        fmin=20,
        fmax=4000
    )
    
    # Apply logarithm to the mel spectrogram to obtain log mel features
    log_mel_features = librosa.amplitude_to_db(
        mel_spectrogram,
        ref=np.max
    )
    
    return log_mel_features


def log_mel_feature_extraction(waveform):
    """
    Perform log mel feature extraction on the waveform.

    Parameters
    ----------
    waveform : tf.Tensor
        The waveform tensor.

    Returns
    -------
    log_mel_features : tf.Tensor
        The log mel features tensor.
    """
    # Convert waveform to mono if it has multiple channels
    if tf.rank(waveform) > 1:
        waveform = tf.reduce_mean(waveform, axis=-1)

    # Compute the log mel spectrogram
    spectrogram = tf.signal.stft(
        signals=waveform,
        frame_length=400,
        frame_step=160,
        fft_length=400
    )
    magnitude_spectrogram = tf.abs(spectrogram)

    # Apply mel filterbank
    num_mel_bins = 40
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=400,
        sample_rate=16000,
        lower_edge_hertz=20,
        upper_edge_hertz=4000
    )
    mel_spectrogram = tf.matmul(magnitude_spectrogram, linear_to_mel_weight_matrix)

    # Apply logarithm to the mel spectrogram to obtain log mel features
    log_mel_features = tf.math.log(mel_spectrogram + 1e-6)

    # Frame stacking
    left_context = 23
    right_context = 8
    num_frames = tf.shape(log_mel_features)[1]

    stacked_features = tf.TensorArray(dtype=log_mel_features.dtype, size=num_frames)
    for i in tf.range(num_frames):
        frame = log_mel_features[:, tf.maximum(i - left_context, 0):i + right_context + 1]
        stacked_features = stacked_features.write(i, frame)

    stacked_features = stacked_features.stack()

    return stacked_features


def plot_spectrogram(spectrogram,
                     ax):
    """
    Plot a spectrogram.
    
    Parameters
    ----------
    spectrogram : np.ndarray
        A spectrogram array.
    ax : plt.Axes
        A matplotlib axes object.
    """
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)