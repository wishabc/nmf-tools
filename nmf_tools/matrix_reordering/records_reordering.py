import numpy as np

import scipy.cluster.hierarchy as sch
from . import MatrixReordering
from .hcluster import hierarchical_clustering

def order_records(W, *, by='primary', normalize=False, **kwargs):
    """
    Sorts elements in each row according to the specified method.

    Parameters
    ----------
    W : np.ndarray (n_components, n_records)
        Matrix to reorder.

    by : str
        Method to use for reordering.
        Can be one of:
            [TODO]

    Returns
    -------
    component_order : MatrixReordering object
    """
    if normalize:
        W = W / W.sum(axis=0, keepdims=True)
    
    if isinstance(by, MatrixReordering):
        return by
    elif by == 'primary':
        return order_records_primary(W, **kwargs)
    elif by == 'hierarchical':
        return order_records_hierarchical(W, **kwargs)
    elif by == 'ratio':
        return order_records_by_ratio(W, **kwargs)
    elif by == 'component':
        return order_records_by_component(W, **kwargs)
    elif by is None:
        return MatrixReordering()
    else:
        raise ValueError(f"Unknown ordering method: {by}")

def order_records_primary(W):
    '''
    Reorder the records of W by primary component.
    '''
    primary_component = np.argmax(W, axis=0) # shape (n_records,)
    sep = np.max(primary_component) + np.max(W) + 1
    records_order = np.argsort(sep * primary_component + W[primary_component, np.arange(W.shape[1])])[::-1]
    return MatrixReordering(col_order=records_order)


def order_records_hierarchical(W, linkage_matrix=None, **kwargs):
    '''
    Reorder the records of W using hierarchical clustering.
    '''
    if linkage_matrix is None:
        linkage_matrix = hierarchical_clustering(W, **kwargs)
    dendrogram = sch.dendrogram(linkage_matrix, no_plot=True)
    leaves_order = dendrogram['leaves']
    return MatrixReordering(col_order=leaves_order)


def order_records_by_ratio(W, ind1, ind2):
    '''
    Order the records of W by the ratio of two components.
    '''
    ratio = W[ind1, :] / (W[ind1, :] + W[ind2, :])
    return MatrixReordering(col_order=np.argsort(ratio)[::-1])

def order_records_by_component(W, ind):
    '''
    Order the records of W by a single component loading.
    '''
    return MatrixReordering(col_order=np.argsort(W[ind, :])[::-1])


def order_records_by_top_n(W):
    pass
#     '''
#     Order the records of W by the top n components.
#     '''
#     n = min(n, W.shape[0])
#     agst = np.argsort(W, axis=0) # shape (n_components, n_records)
#     sep = np.max(agst) + np.max(W) + 1
#     primary_component = agst[0, :]
#     order = np.argsort(sum(
#         sep**i * agst[i, :] for i in range(n)
#     ) - W[primary_component, :])[::-1]
#     return MatrixReordering(col_order=order)
