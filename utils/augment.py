import tensorflow as tf
import numpy as np
import librosa
#from tensorflow.python.ops.numpy_ops import np_config
#np_config.enable_numpy_behavior()

@tf.function
def time_mask(
    spectrogram,
    num_masks:int=4,
    mask_factor:int=4
):
    """
    Apply time masking to spectrogram
    
    Parameters
    ----------
    spectrogram : tf.Tensor
        Spectrogram to apply time masking to
    num_masks : int, optional
        Number of masks to apply, by default 4
    mask_factor : int, optional
        Maximum number of time frames to mask, by default 4
        
    Returns
    -------
    tf.Tensor
        Masked spectrogram
    """
    def apply_time_mask(
        spec,
        num_masks=4,
        mask_factor=4
    ):
        masked_spec = spec.copy()
        _, time_frames = masked_spec.shape
        n_masks = tf.random.uniform(
            shape=[],
            minval=1,
            maxval=num_masks+1,
            dtype=tf.int32
        )
        for _ in range(n_masks):
            t = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=time_frames,
                dtype=tf.int32
            )
            t_mask = tf.random.uniform(
                shape=[],
                minval=1,
                maxval=mask_factor+1,
                dtype=tf.int32
            )
            masked_spec[:, t:t + t_mask] = 0
        return masked_spec

    return tf.numpy_function(
        apply_time_mask,
        [spectrogram, num_masks, mask_factor],
        tf.float32
    )


@tf.function
def freq_mask(
    spectrogram,
    num_masks=6,
    mask_factor=16
):
    """
    Apply freq masking to spectrogram
    
    Parameters
    ----------
    spectrogram : tf.Tensor
        Spectrogram to apply freq masking to
    num_masks : int, optional
        Number of masks to apply, by default 4
    mask_factor : int, optional
        Maximum number of frequency bins to mask, by default 4

    Returns
    -------
    tf.Tensor
        Masked spectrogram
    """
    def apply_freq_mask(
        spec,
        num_masks=4,
        mask_factor=4
    ):
        masked_spec = spec.copy()
        freq_bins, _ = masked_spec.shape
        n_masks = tf.random.uniform(
            shape=[],
            minval=1,
            maxval=num_masks+1,
            dtype=tf.int32
        )
        for _ in range(n_masks):
            f = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=freq_bins,
                dtype=tf.int32
            )
            f_mask = tf.random.uniform(
                shape=[],
                minval=1,
                maxval=mask_factor+1,
                dtype=tf.int32
            )
            masked_spec[f:f + f_mask, :] = 0
        return masked_spec

    return tf.numpy_function(
        apply_freq_mask,
        [spectrogram, num_masks, mask_factor],
        tf.float32
    )


@tf.function
def time_freq_mask(
    spectrogram,
    num_masks=8,
    time_mask_factor=8,
    freq_mask_factor=16
):
    def apply_time_freq_mask(
        spec,
        num_masks=8,
        time_mask_factor=4,
        freq_mask_factor=4
    ):
        masked_spec = spec.copy()
        freq_bins, time_frames = masked_spec.shape
        n_masks = tf.random.uniform(
            shape=[],
            minval=1,
            maxval=num_masks+1,
            dtype=tf.int32
        )
        for _ in range(n_masks):
            t = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=time_frames,
                dtype=tf.int32
            )
            t_mask = tf.random.uniform(
                shape=[],
                minval=1,
                maxval=time_mask_factor+1,
                dtype=tf.int32
            )
            f = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=freq_bins,
                dtype=tf.int32
            )
            f_mask = tf.random.uniform(
                shape=[],
                minval=1,
                maxval=freq_mask_factor+1,
                dtype=tf.int32
            )
            masked_spec[f:f + f_mask, t:t + t_mask] = 0
        return masked_spec

    return tf.numpy_function(
        apply_time_freq_mask,
        [spectrogram, num_masks, time_mask_factor, freq_mask_factor],
        tf.float32
    )


@tf.function
def time_warp(spectrogram, max_warping_factor=40):
    def apply_time_warp(spectrogram, max_warp_factor=50):
        # Create a copy of the input spectrogram tensor
        warped_spectrogram = tf.identity(spectrogram)
        
        freq_bins, time_frames = spectrogram.shape
        warp_factor = np.random.uniform(0.1, max_warp_factor)  # Ensure warp_factor is non-zero
        start_time = np.random.randint(1, time_frames // 2)
        end_time = np.random.randint(start_time + 5, time_frames - 1)

        num_frames = spectrogram.shape[1]
        warp_range = end_time - start_time + 1
        warped_range = int(warp_range / warp_factor)
        scale_factor = warp_range / warped_range
        new_start_time = int(start_time + (warp_range - warped_range) / 2)
        new_end_time = new_start_time + warped_range - 1

        for i in range(spectrogram.shape[0]):
            warped_spectrogram[i, new_start_time:new_end_time + 1] = tfp.math.interp_regular_1d_grid(
                np.linspace(start_time, end_time, warped_range),
                tf.range(num_frames, dtype=tf.float32),
                spectrogram[i]
            )

        return warped_spectrogram

    return tf.numpy_function(
        apply_time_warp,
        [spectrogram, max_warping_factor],
        tf.float32
    )
    

### Augmentations for audio data ###
time_stretch_range = (0.8, 1.2)
@tf.function
def time_stretch(
    audio,
):
    """
    Apply time stretching to audio
    
    Parameters
    ----------
    audio : tf.Tensor
        Audio to apply time stretching to
    rate : float, optional
        Rate to stretch audio by, by default 1.0
    
    Returns
    -------
    tf.Tensor
        Stretched audio
    """
    def apply_time_stretch(audio, rate):
        stretched_audio = librosa.effects.time_stretch(
            audio.copy(),
            rate=rate
        )
        return stretched_audio

    rate = tf.random.uniform(
        [],
        minval=time_stretch_range[0],
        maxval=time_stretch_range[1],
        dtype=tf.float32
    )

    return tf.numpy_function(
        apply_time_stretch,
        [audio, rate],
        tf.float32
    )


pitch_shift_range = (-4, 4)
@tf.function
def pitch_shift(
    audio,
):
    """
    Apply pitch shifting to audio
    
    Parameters
    ----------
    audio : tf.Tensor
        Audio to apply pitch shifting to
    n_steps : int, optional
        Number of steps to shift pitch by, by default 0
    
    Returns
    -------
    tf.Tensor
        Pitch shifted audio
    """
    def apply_pitch_shift(audio, n_steps):
        shifted_audio = librosa.effects.pitch_shift(
            audio.copy(),
            sr=22050,
            n_steps=n_steps
        )
        return shifted_audio

    n_steps = tf.random.uniform(
        [],
        minval=pitch_shift_range[0],
        maxval=pitch_shift_range[1],
        dtype=tf.int32
    )

    return tf.numpy_function(
        apply_pitch_shift,
        [audio, n_steps],
        tf.float32
    )