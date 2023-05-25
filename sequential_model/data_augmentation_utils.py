"""
https://towardsdatascience.com/data-augmentation-techniques-for-audio-data-in-python-15505483c63c

"""
import numpy as np
import librosa
import torch
import colorednoise as cn

def add_noise(
        data,
        noise_factor=0.005,
        noise_type='white',
        ):
    
    """
    Adds noise to the data

    Args:
        data (torch.tensor): audio time series (1D
        noise_factor (float): noise factor
        noise_type (str): 'white', 'pink' or 'brown'

    Returns:
        augmented_data (torch.tensor): audio time series with added noise

    """

    data = data.numpy()
    if noise_type == 'white':
        noise = np.random.randn(len(data)) * noise_factor
    elif noise_type == 'pink':
        # see https://github.com/felixpatzelt/colorednoise/blob/master/colorednoise.py
        # flicker / pink noise:   exponent beta = 1
        noise = cn.powerlaw_psd_gaussian(1, len(data))
    elif noise_type == 'brown':
        # brown noise:   exponent beta = 2
        noise = cn.powerlaw_psd_gaussian(2, len(data))
    else:
        raise ValueError('noise_type must be "white", "pink" or "brown"')

    augmented_data = data + noise
    return torch.tensor(augmented_data)


def shift(data, shift_size, shift_direction='both'):
    """
    Shifts the data randomly to the left or right by a random number of samples

    Args:
        data (torch.tensor): audio time series
        shift (int): number of samples to shift
        shift_direction (str): 'right', 'left' or 'both'

    Returns:
        augmented_data (torch.tensor): shifted audio time series

    """
    # data to numpy
    data = data.numpy()
    # shift data
    shift = np.random.randint(-shift_size, shift_size)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    return torch.tensor(augmented_data)

def stretch(data, rate=1):
    """
    Stretches the data by rate

    Args:
        data (torch.tensor): audio time series
        rate (float): stretch rate

    Returns:
        augmented_data (torch.tensor): stretched audio time series

    """
    data = data.numpy()
    input_length = len(data)
    data = librosa.effects.time_stretch(y=data, rate=rate)
    if len(data) > input_length:
        return torch.tensor(data[:input_length])
    else:
        # pad with zeros to the input length
        return torch.tensor(np.pad(data, (0, max(0, input_length - len(data))), "constant"))
    

def pitch_shift(data, sampling_rate, pitch_factor=0.7):
    """
    Shifts the pitch of the data by pitch_factor

    Args:
        data (torch.tensor): audio time series
        sampling_rate (int): sampling rate of the audio time series
        pitch_factor (float): pitch factor

    Returns:
        augmented_data (torch.tensor): audio time series with shifted pitch

    """
    data = data.numpy()
    return torch.tensor(librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor))
