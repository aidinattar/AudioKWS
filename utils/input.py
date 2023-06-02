import os
import tensorflow as tf


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