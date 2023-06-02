import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    
    
def plot_features(spectrogram: np.ndarray,
                  ax: plt.Axes,):
    """
    Plot a spectrogram.

    Parameters
    ----------
    spectrogram : np.ndarray
        A spectrogram array.
    ax : plt.Axes
        A matplotlib axes object.
    """
    # check if the spectrogram is 3D
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        # squeeze the spectrogram to remove the channel dimension
        # Find the dimension with size 1
        squeeze_axis = np.squeeze(np.where(np.array(spectrogram.shape) == 1))
        # Squeeze the array along the found dimension
        spectrogram = np.squeeze(spectrogram, axis=squeeze_axis)

    ax = sns.heatmap(spectrogram, ax=ax, cmap="viridis", cbar=False)
    ax.set_xlabel("Window")
    ax.set_ylabel("Features")
    return ax