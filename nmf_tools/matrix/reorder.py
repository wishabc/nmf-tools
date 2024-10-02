import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as sch

import numpy as np

class MatrixReordering:
    def __init__(self, m, n):
        self.m = m  # Number of rows
        self.n = n  # Number of columns
        self.within_row_order = None
        self.within_col_order = None

    def reorder(self, order, axis):
        order = np.array(order)
        expected_shape = (self.m, self.n) if axis == 0 else (self.n, self.m)
        if order.ndim == 1:
            assert order.shape[0] == expected_shape[1], "Order must have the same length as the number of rows or columns"
        elif order.ndim == 2:
            assert order.shape == expected_shape, f"Order must be of shape {expected_shape}"
        else:
            raise AssertionError("Order must be a 1D or a 2D array")
        if axis == 0:
            assert order.max() < self.n
            self._check_dim(order.ndim + getattr(self.within_row_order, 'ndim', 0))
            self.within_row_order = order
        else:
            assert order.max() < self.m
            self._check_dim(order.ndim + getattr(self.within_col_order, 'ndim', 0))
            self.within_col_order = order

    @staticmethod
    def _check_dim(n):
        if n >= 4:
            raise ValueError("Within-row and within-column ordderings can not be both 2D arrays")

    def combine(self, other):
        assert self.m == other.m and self.n == other.n, "Both matrices must have the same dimensions to combine"
        self._check_dim(
            max(
                getattr(self.within_row_order, 'ndim', 1),
                getattr(other.within_row_order, 'ndim', 1)
            ) + 
            max(
                getattr(self.within_col_order, 'ndim', 1),
                getattr(other.within_col_order, 'ndim', 1)
            )
        )
        self.within_row_order = self._combine_axis(self.within_row_order, other.within_row_order)
        self.within_col_order = self._combine_axis(self.within_col_order, other.within_col_order)

    def _combine_axis(self, current_order, other_order):
        if other_order is not None:
            if current_order is not None:
                if current_order.ndim == 1:
                    current_order = current_order[None, :]
                if other_order.ndim == 1:
                    other_order = other_order[None, :]
                return np.squeeze(np.take_along_axis(current_order, other_order, axis=1))
            return other_order
        return current_order

    def _apply_reordering(self, matrix, order):
        if order.ndim == 1:
            return matrix[:, order]
        return np.take_along_axis(matrix, order, axis=1)
    
    def apply_reordering(self, matrix):
        matrix = np.array(matrix)

        assert matrix.shape == (self.m, self.n), "Matrix must have the same dimensions as the reordering object"
        
        # Apply within-row reordering using np.take_along_axis
        if self.within_row_order is not None:
            matrix = self._apply_reordering(matrix, self.within_row_order)

        # Apply within-column reordering using np.take_along_axis
        if self.within_col_order is not None:
            matrix = self._apply_reordering(matrix.T, self.within_col_order).T

        return matrix


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