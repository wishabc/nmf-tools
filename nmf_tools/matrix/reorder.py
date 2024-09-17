import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as sch


def reorder_components(W1, W2):
    '''
    Reorder the components of W2 to match the components of W1.

    Parameters
    ----------
    W1 : np.ndarray (n_components, n_records)
        Matrix to match.
    W2 : np.ndarray (n_components, n_records)
        Matrix to reorder.

    Returns
    -------
    col_ind : np.ndarray (n_components,)
        New order of components.
    '''
    cosine_similarity = 1 - cdist(W1, W2, 'cosine')
    _, col_ind = linear_sum_assignment(-cosine_similarity)
    return col_ind


def hierarchical_order_records(W):
    '''
    Reorder the records of W using hierarchical clustering.
    '''
    distance_matrix = sch.distance.pdist((W).T, metric='cosine')
    linkage_matrix = sch.linkage(distance_matrix, method='average')
    dendrogram = sch.dendrogram(linkage_matrix, no_plot=True)
    leaves_order = dendrogram['leaves']
    return leaves_order


def apply_order(W, record_order=None, component_orders=None):
    '''
    Apply record_order and component_order to W.

    Parameters
    ----------
    W : np.ndarray (n_components, n_records)
        Matrix to reorder.
    record_order : np.ndarray (n_records,)
        New order of records.
    component_orders : np.ndarray (n_records, n_components)
        New order of components for each record.

    Returns
    -------
    W_reordered : np.ndarray (n_components, n_records)
        Reordered matrix.
    '''
    if component_orders is not None:
        W = np.take_along_axis(W, component_orders, axis=0)

    if record_order is not None:
        W = W[:, record_order]

    return W


def identity_component_orders(W, order=None):
    """
    Sorts all rows in the same order.

    Parameters
    ----------
    W : np.ndarray (n_components, n_records)
        Matrix to reorder.
    order : np.ndarray (n_components,)
        New order of components.
        Can be None or an array of indices.
        If None, the order is the same as the original.
        If an array, the order is specified by the array.

    Returns
    -------
    component_orders : np.ndarray (n_components, n_records)
        New order of components for each record.

    """
    if order is None:
        order = np.arange(W.shape[0])
    order = np.array(order)
    assert len(order.shape) == 1 and order.shape[0] == W.shape[0]
    return np.tile(order, W.shape[1]).reshape(W.shape[::-1]).T


def by_cluster_component_orders(W, cluster_labels=None):
    """
    Sorts elements in each column according to the mean relative value in the specified clusters.

    Parameters
    ----------
    W : np.ndarray (n_components, n_records)
        Matrix to reorder.

    clusters_s : np.ndarray (n_records,)
        Cluster assignment for each record.
        If None, all records are considered to be in the same cluster.

    Returns
    -------
    component_orders : np.ndarray (n_components, n_records)
        New order of components for each record.
    """
    if cluster_labels is None:
        cluster_labels = np.zeros(W.shape[1], dtype=int)
    component_orders = np.zeros(W.shape, dtype=int)
    for i in np.unique(cluster_labels):
        idx = cluster_labels == i
        component_priority = np.argsort((W[:, idx] / W[:, idx].sum(axis=0)).mean(axis=1))[::-1]
        component_orders[:, idx] = component_priority[:, None]
    return component_orders