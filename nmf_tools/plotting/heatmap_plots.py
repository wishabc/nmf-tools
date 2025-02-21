import numpy as np

from scipy.interpolate import RegularGridInterpolator

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

def interpolate_matrix(matrix, n_downs=1000, interp_method='linear'):
    """
    Interpolates a 2D matrix to a new grid of size at most n_downs x n_downs.

    Parameters:
    -----------
    matrix : np.ndarray
        Input 2D array.
    n_downs : int
        Maximum number of points along each dimension.
    interp_method : str
        Interpolation method: 'linear' or 'nearest'. Using 'nearest' helps to preserve
        nonzero values in sparse arrays by avoiding averaging with zeros.
    
    Returns:
    --------
    new_matrix : np.ndarray
        Interpolated matrix.
    """
    m, n = matrix.shape

    # Determine new dimensions (downsample if needed)
    new_m = min(m, n_downs)
    new_n = min(n, n_downs)

    # Generate the original grid: indices from 0 to m-1 and 0 to n-1.
    x = np.linspace(0, m - 1, m)
    y = np.linspace(0, n - 1, n)

    # Generate the new grid over the same coordinate space
    new_x = np.linspace(0, m - 1, new_m)
    new_y = np.linspace(0, n - 1, new_n)

    # Create the interpolator with the specified method
    interpolator = RegularGridInterpolator((x, y), matrix, method=interp_method)

    # Create meshgrid of points where we want to interpolate
    new_grid_x, new_grid_y = np.meshgrid(new_x, new_y, indexing='ij')
    new_points = np.array([new_grid_x.ravel(), new_grid_y.ravel()]).T

    # Interpolate and reshape to new dimensions
    new_matrix = interpolator(new_points).reshape(new_m, new_n)

    return new_matrix


def heatmap_plot(matrix, labels=None, max=False, ax=None, ylabel=None,
                 n_downs=1000, interp_method='linear', box_lw=0.15, **kwargs):
    matrix = interpolate_matrix(matrix, n_downs=n_downs, interp_method=interp_method)
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