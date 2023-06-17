import numpy as np
import tensorflow as tf
from scipy.fftpack import dct

def get_spectrogram(
    waveform: tf.Tensor,
    input_len: int = 16000,
    frame_length: int = 400,
    frame_step: int = 160,
    fft_length: int = 512
    ) -> tf.Tensor:
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
        equal_length,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=tf.signal.hann_window,
        pad_end=False
    )
    
    # Obtain the magnitude of the STFT.
    spectrogram = tf.math.log(tf.abs(tf.transpose(spectrogram)) + np.finfo(float).eps)
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
    # remove the last dimension
    spectrogram = tf.squeeze(spectrogram, axis=-1)
     
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def get_log_mel_features_and_label_id(audio,
                                      label,
                                      commands):
    """
    Convert the waveform to log mel features via a STFT.
    
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
    log_mel_features : tf.Tensor
        A log mel features tensor.
    label_id : tf.Tensor
        A label ID tensor.
    """
    log_mel_features = log_mel_feature_extraction(audio)
    label_id = tf.argmax(label == commands)
    return log_mel_features, label_id


def log_mel_feature_extraction(
    waveform: tf.Tensor,
    input_len: int = 16000,
    frame_length: int = 400,
    frame_step: int = 160,
    fft_length: int = 512
    )-> tf.Tensor:
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
    if waveform.shape.ndims > 1:
        waveform = tf.reduce_mean(waveform, axis=-1)

    # Zero-padding for an audio waveform with less than 16,000 samples.
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    waveform = tf.concat([waveform, zero_padding], 0)

    # Compute the log mel spectrogram
    mel_spectrogram = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=257,
        sample_rate=16000,
        lower_edge_hertz=20,
        upper_edge_hertz=4000
    )

    stft = tf.signal.stft(
        waveform,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=tf.signal.hann_window,
        pad_end=False
    )

    mel_spectrogram = tf.matmul(tf.abs(stft), mel_spectrogram)

    # Apply logarithm to the mel spectrogram to obtain log mel features
    log_mel_features = tf.math.log(mel_spectrogram + np.finfo(float).eps)

    # transpose log mel features so that the
    # width dimension corresponds to the time axis
    log_mel_features = tf.transpose(log_mel_features)

    return log_mel_features


def log_mel_feature_extraction_(waveform):
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


def compute_mfcc(
    log_mel_features: tf.Tensor,
    num_mfcc:int=13
) -> tf.Tensor:
    """
    Compute MFCC from log mel features.

    Parameters
    ----------
    log_mel_features : tf.Tensor
        Log mel features tensor of shape (batch_size, time_steps, num_mel_bins).
    num_mfcc : int, optional
        Number of MFCC coefficients to compute (default is 13).
    dct_type : int, optional
        Type of DCT (discrete cosine transform) to use (default is 2).

    Returns
    -------
    mfcc : tf.Tensor
        MFCC tensor of shape (batch_size, time_steps, num_mfcc).
    """
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_features
    )
    return mfcc[..., :num_mfcc]


def get_mfcc_and_label_id(audio,
                          label,
                          commands):
    """
    Convert the waveform to MFCC features via a STFT.

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
    mfcc_features : tf.Tensor
        A MFCC features tensor.
    label_id : tf.Tensor
        A label ID tensor.
    """
    log_mel_features = tf.transpose(log_mel_feature_extraction(audio))
    mfcc_features = tf.transpose(compute_mfcc(log_mel_features))
    label_id = tf.argmax(label == commands)
    return mfcc_features, label_id


def compute_mfcc_map(
    log_mel_spectrum: tf.Tensor,
    num_cepstral_coeffs:int=13,
    dct_type:int=2,
)-> tf.Tensor:
    """
    Compute MFCC from log mel features.
    
    Parameters
    ----------
    log_mel_features : tf.Tensor
        Log mel features tensor of shape (batch_size, time_steps, num_mel_bins).
    num_mfcc : int, optional
        Number of MFCC coefficients to compute (default is 13).
    dct_type : int, optional
        Type of DCT (discrete cosine transform) to use (default is 2).
    
    Returns
    -------
    mfcc : tf.Tensor
        MFCC tensor of shape (batch_size, time_steps, num_mfcc).
    """
    mfcc = dct(log_mel_spectrum.numpy(), type=dct_type, axis=-1, norm='ortho')[:, :num_cepstral_coeffs]
    mfcc = tf.convert_to_tensor(mfcc)
    return mfcc


def compute_delta(
    features: tf.Tensor,
    window: int = 2
) -> tf.Tensor:
    """
    Compute delta features.

    Parameters
    ----------
    features : tf.Tensor
        Input features tensor of shape (batch_size, time_steps, num_features).
    window : int, optional
        Window size (default is 2).

    Returns
    -------
    delta : tf.Tensor
        Delta features tensor of shape (batch_size, time_steps, num_features).
    """
    num_frames = tf.shape(features)[1]
    padded_features = tf.pad(features, [[0, 0], [window, window], [0, 0]])
    delta = tf.TensorArray(dtype=features.dtype, size=num_frames)
    for i in tf.range(num_frames):
        frame = padded_features[:, i:i + 2 * window + 1]
        delta = delta.write(i, tf.tensordot(tf.range(-window, window + 1), frame, axes=1))
    delta = delta.stack()
    return delta