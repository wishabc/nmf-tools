import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as sch

import numpy as np

class MatrixReordering:
    """
    Class to store and apply reordering of rows and columns of a matrix.
    
    Attributes:
    -----------
    m : int
        Number of rows in the matrix.
    n : int
        Number of columns in the matrix.
    within_row_order : np.ndarray or None
        Order for reordering rows of the matrix. Can be 1D or 2D.
    within_col_order : np.ndarray or None
        Order for reordering columns of the matrix. Can be 1D or 2D.
    
    Methods:
    --------
    reorder(order: np.ndarray, axis: int) -> None
        Reorders the rows or columns based on the provided order and axis.
    combine(other: 'MatrixReordering') -> None
        Combines two reorderings of the same matrix dimensions.
    apply_reordering(matrix: np.ndarray) -> np.ndarray
        Applies the reordering to a matrix.
    """

    def __init__(self, row_order: np.ndarray = None, col_order: np.ndarray = None) -> None:
        """
        Initialize the MatrixReordering object with the number of rows (m) and columns (n).
        
        Parameters:
        -----------
        m : int
            Number of rows.
        n : int
            Number of columns.
        col_order : np.ndarray or None
            Order for reordering columns of the matrix. Can be 1D or 2D.
            If 1D, the shape must be (n,).
            If 2D, the shape must be (m, n). Specifies the order of columns for each row.
        row_order : np.ndarray or None
            Order for reordering rows of the matrix. Can be 1D or 2D.
            If 1D, the shape must be (m,).
            If 2D, the shape must be (n, m). Specifies the order of rows for each column.
        """

        self.m = None # Number of rows
        self.n = None # Number of columns
            
        self.col_order = np.asarray(col_order) if col_order is not None else None
        self.row_order = np.asarray(row_order) if row_order is not None else None

        self._validate_order()


            

    @property
    def row_order_dim(self):
        return getattr(self.row_order, 'ndim', 0)
    
    @property
    def col_order_dim(self):
        return getattr(self.col_order, 'ndim', 0)

    def _validate_order(self) -> None:
        """
        Set the reordering for rows (axis=0) or columns (axis=1).
        
        Parameters:
        -----------
        order : np.ndarray
            Reordering order for rows or columns. Can be 1D or 2D.
        axis : int
            Axis to reorder, 0 for rows and 1 for columns.
        """
        
        if self.row_order_dim > 0:
            if self.row_order_dim == 1:
                self.m = self.row_order.shape[0]
            else:
                self.n, self.m = self.row_order.shape
        if self.col_order_dim > 0:
            if self.col_order_dim == 1:
                self.n = self.col_order.shape[0]
            else:
                self.m, self.n = self.col_order.shape

        if self.col_order_dim > 0:
            assert self.col_order.max() < self.n, f"Column order values exceed the number of columns {self.n}."
        if self.row_order_dim > 0:
            assert self.row_order.max() < self.m, f"Row order values exceed the number of rows {self.m}."

        self._check_dim(self.col_order_dim + self.row_order_dim)

    @staticmethod
    def _check_dim(n: int) -> None:
        """
        Checks if the dimensions of both row and column reorderings are valid.
        
        Parameters:
        -----------
        n : int
            Sum of the dimensions of the row and column orders.
        """
        if n > 2:
            raise ValueError("Total dimensions of row and column orders must be at most 2.")

    def combine(self, other: 'MatrixReordering') -> None:
        """
        Combines the reorderings from two MatrixReordering objects.
        
        Parameters:
        -----------
        other : MatrixReordering
            Another MatrixReordering object with the same dimensions.
        """
        assert self.m == other.m and self.n == other.n, \
            "Both matrices must have the same dimensions to combine."

        self._check_dim(
            max(self.row_order_dim, other.row_order_dim) +
            max(self.col_order_dim, other.col_order_dim)
        )

        self.col_order = self._combine_axis(self.col_order, other.col_order)
        self.row_order = self._combine_axis(self.row_order, other.row_order)

    def _combine_axis(self, current_order: np.ndarray, other_order: np.ndarray) -> np.ndarray:
        """
        Combines the current reordering order with another reordering order along a specific axis.
        
        Parameters:
        -----------
        current_order : np.ndarray
            The current reordering order.
        other_order : np.ndarray
            The other reordering order.
        
        Returns:
        --------
        np.ndarray
            The combined reordering order.
        """
        if other_order is None:
            return current_order
        if current_order is None:
            return other_order

        # Ensure both are at least 2D
        current_order = current_order[None, :] if current_order.ndim == 1 else current_order
        other_order = other_order[None, :] if other_order.ndim == 1 else other_order

        return np.squeeze(np.take_along_axis(current_order, other_order, axis=1))

    def _apply_reordering(self, matrix, axis):
        """
        Applies reordering based on the given order along axis 1 for the matrix.
        
        Parameters:
        -----------
        matrix : np.ndarray
            The matrix to be reordered.
        order : np.ndarray
            The order to apply for reordering. Can be 1D or 2D.
        
        Returns:
        --------
        np.ndarray
            The reordered matrix.
        """
        if axis == 1:
            order = self.col_order
        elif axis == 0:
            order = self.row_order
        else:
            raise ValueError("Axis must be 0 or 1")
        
        if order is None:
            return matrix
        
        if axis == 0:
            matrix = matrix.T

        if order.ndim == 1:
            result =  matrix[:, order]
        else:
            result = np.take_along_axis(matrix, order, axis=1)

        if axis == 0:
            result = result.T

        return result
    
    def __call__(self, matrix):
        """
        Applies the stored row and column reorderings to the given matrix.
        
        Parameters:
        -----------
        matrix : np.ndarray
            The matrix to which the stored reorderings should be applied.
        
        Returns:
        --------
        np.ndarray
            The reordered matrix.
        """
        matrix = np.array(matrix)

        if self.m is not None:
            assert matrix.shape[0] == self.m, "Matrix must have the same number of rows as the reordering object"
        if self.n is not None:
            assert matrix.shape[1] == self.n, "Matrix must have the same number of columns as the reordering object"
        
        # The 2D transformation needs to be applied first
        axis_order = [0, 1] if self.row_order_dim == 2 else [1, 0]
        for axis in axis_order:
            matrix = self._apply_reordering(matrix, axis)
        return matrix
    
    def __array__(self):
        assert self.row_order_dim + self.col_order_dim == 1, "Only 1D reordering can be used as an array index."
        if self.row_order_dim == 1:
            return self.row_order
        else: # col_order_dim == 1
            return self.col_order
    
    @property
    def inv(self):
        """
        Returns the inverse of the reordering object.
        
        Returns:
        --------
        MatrixReordering
            The inverse of the reordering object.
        """
        if self.row_order_dim == 0:
            row_order = None
        elif self.row_order_dim == 1:
            row_order = np.argsort(self.row_order)
        else:
            row_order = np.argsort(self.row_order, axis=1)

        if self.col_order_dim == 0:
            col_order = None
        elif self.col_order_dim == 1:
            col_order = np.argsort(self.col_order)
        else:
            col_order = np.argsort(self.col_order, axis=1)

        return MatrixReordering(row_order, col_order)
    
    def __repr__(self):
        return f"MatrixReordering(m={self.m}, n={self.n}, row_order_dim={self.row_order_dim}, col_order_dim={self.col_order_dim})"
    

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
    _, order = linear_sum_assignment(-cosine_similarity)
    return MatrixReordering(row_order=order)


def hierarchical_order_records(W):
    '''
    Reorder the records of W using hierarchical clustering.
    '''
    distance_matrix = sch.distance.pdist((W).T, metric='cosine')
    linkage_matrix = sch.linkage(distance_matrix, method='average')
    dendrogram = sch.dendrogram(linkage_matrix, no_plot=True)
    leaves_order = dendrogram['leaves']
    return MatrixReordering(col_order=leaves_order)


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
    return MatrixReordering(row_order=component_orders)


def order_by_primary_component(W):
    '''
    Order the components of W by the primary component.
    '''
    primary_component = np.argmax(W, axis=0) # shape (n_records,)
    sep = np.max(primary_component) + np.max(W) + 1
    record_order = np.argsort(sep * primary_component - W[primary_component, :])[::-1]
    return MatrixReordering(col_order=order)


def order_by_top_n_components(W):
    '''
    
    '''
    agst = np.argsort(W, axis=0) # shape (n_components, n_records)
    sep = np.max(agst) + np.max(W) + 1
    primary_component = agst[0, :]
    order = np.argsort(sum(
        sep**i * agst[i, :] for i in range(W.shape[0])
    ) - W[primary_component, :])[::-1]
    return MatrixReordering(row_order=order)