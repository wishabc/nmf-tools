import numpy as np

from scipy.interpolate import RegularGridInterpolator

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from skimage.transform import downscale_local_mean

def interpolate_matrix(matrix, n_downs=1000):
    """
    Downsamples a 2D matrix by averaging blocks using vectorized operations.
    If possible, reshapes the matrix to perform a fast mean operation.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input 2D array.
    n_downs : int, optional
        Maximum number of points along each dimension.
    
    Returns
    -------
    new_matrix : np.ndarray
        Downsampled (smoothed) matrix.
    """
    m, n = matrix.shape
    new_m = min(m, n_downs)
    new_n = min(n, n_downs)
    
    # Compute block sizes (integer division)
    block_size_m = m // new_m
    block_size_n = n // new_n
    
    # Check if matrix dimensions are exactly divisible by target dimensions
    if m % new_m == 0 and n % new_n == 0:
        # Reshape and compute mean in a vectorized way.
        # Reshape to (new_m, block_size_m, new_n, block_size_n)
        reshaped = matrix.reshape(new_m, block_size_m, new_n, block_size_n)
        new_matrix = reshaped.mean(axis=(1, 3))
    else:
        # Option 1: Use skimage.transform.downscale_local_mean, which handles non-divisible sizes by averaging over local blocks.
        # Calculate block sizes as floats (approximately)
        block_size_m_float = m / new_m
        block_size_n_float = n / new_n
        # downscale_local_mean requires integer factors, so pad the matrix to the next multiple.
        pad_m = (np.ceil(m / new_m) * new_m - m).astype(int)
        pad_n = (np.ceil(n / new_n) * new_n - n).astype(int)
        padded_matrix = np.pad(matrix, ((0, pad_m), (0, pad_n)), mode='edge')
        # Now determine integer block sizes after padding.
        padded_m, padded_n = padded_matrix.shape
        factor_m = padded_m // new_m
        factor_n = padded_n // new_n
        new_matrix = downscale_local_mean(padded_matrix, (factor_m, factor_n))
        # In case the padded result is slightly larger, slice it.
        new_matrix = new_matrix[:new_m, :new_n]
        
    return new_matrix


def heatmap_plot(matrix, labels=None, max=False, ax=None, ylabel=None,
                 n_downs=1000, box_lw=0.15, **kwargs):
    matrix = interpolate_matrix(matrix, n_downs=n_downs)
    if ax is None:
        ax = plt.gca()
    hm = np.nan_to_num(matrix)
    if max:
        hm = hm.max(axis=1, keepdims=True)
    ax.pcolormesh(hm, **kwargs)
    ax.set_xticks([])
    if labels is not None:
        ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
        ax.set_yticklabels(labels)
    else:
        ax.set_yticks([])
    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=0, ha='right', va='center')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(box_lw)
    return ax