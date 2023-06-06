import tensorflow as tf
import numpy as np
from scipy.ndimage import affine_transform


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
    def apply_time_mask(spec):
        masked_spec = spec.copy()
        time_frames, _ = masked_spec.shape
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
            masked_spec[t:t + t_mask, :] = 0
        return masked_spec

    return tf.py_function(
        apply_time_mask,
        [spectrogram],
        tf.float32
    )


@tf.function
def doppler_mask(
    spectrogram,
    num_masks=4,
    mask_factor=4
):
    """
    Apply doppler masking to spectrogram
    
    Parameters
    ----------
    spectrogram : tf.Tensor
        Spectrogram to apply doppler masking to
    num_masks : int, optional
        Number of masks to apply, by default 4
    mask_factor : int, optional
        Maximum number of frequency bins to mask, by default 4

    Returns
    -------
    tf.Tensor
        Masked spectrogram
    """
    def apply_doppler_mask(spec):
        masked_spec = spec.copy()
        _, freq_bins = masked_spec.shape
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
            masked_spec[:, f:f + f_mask] = 0
        return masked_spec

    return tf.py_function(
        apply_doppler_mask,
        [spectrogram],
        tf.float32
    )


@tf.function
def time_doppler_mask(
    spectrogram,
    num_masks=8,
    time_mask_factor=4,
    doppler_mask_factor=4
):
    def apply_time_doppler_mask(spec):
        masked_spec = spec.copy()
        time_frames, freq_bins = masked_spec.shape
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
            d = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=freq_bins,
                dtype=tf.int32
            )
            d_mask = tf.random.uniform(
                shape=[],
                minval=1,
                maxval=doppler_mask_factor+1,
                dtype=tf.int32
            )
            masked_spec[t:t + t_mask, d:d + d_mask] = 0
        return masked_spec

    return tf.py_function(
        apply_time_doppler_mask,
        [spectrogram],
        tf.float32
    )


@tf.function
def time_warp(
    spectrogram,
    max_warping_factor:int=80
):
    """
    Apply time warping to spectrogram

    Parameters
    ----------
    spectrogram : tf.Tensor
        Spectrogram to apply time warping to
    max_warping_factor : int, optional
        Maximum number of time frames to warp, by default 80
    
    Returns
    -------
    tf.Tensor
        Warped spectrogram
    """
    def apply_time_warp(spec):
        masked_spec = spec.copy()
        time_frames, freq_bins = masked_spec.shape
        warp_start = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=time_frames,
            dtype=tf.int32
        )
        warp_end = tf.random.uniform(
            shape=[],
            minval=warp_start + 1,
            maxval=tf.minimum(
                warp_start + max_warping_factor,
                time_frames
            ),
            dtype=tf.int32
        )
        t_s = np.array([warp_start, 0])
        t_e = np.array([warp_end, freq_bins])
        masked_spec = affine_transform(
            masked_spec,
            matrix=np.eye(2),
            offset=t_e-t_s,
            output_shape=(
                time_frames,
                freq_bins
            ),
            mode='constant',
            cval=0.0
        )
        return masked_spec

    return tf.py_function(
        apply_time_warp,
        [spectrogram],
        tf.float32
    )