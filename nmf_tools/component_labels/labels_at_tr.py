import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def compute_labels(H, threshold=0.8, sep='_'):
    """
    For a (k x n) matrix H (each column is a DHS profile),
    this function computes:
      1. A string label for each column listing the selected component indices (e.g., '1_2_23'),
         where the selected indices are the smallest set (from the sorted order) that reach
         a cumulative sum of at least threshold * (column total).
      2. A sparse binary membership matrix (shape: k x n) indicating membership of components.
      3. A count vector (length n) of how many components were selected in each column.
    
    The selection is performed on a per-column basis as follows:
      - For each column j, sort H[:,j] in descending order.
      - Compute the cumulative sum along the sorted order.
      - Let m_j be the smallest index such that
            cum_sum[m_j, j] >= threshold * (total sum of H[:,j]).
      - The selected components are the indices in H corresponding to the top m_j+1 entries.
    
    Parameters:
      H : np.ndarray of shape (k, n)
          Input matrix with non-negative values.
      threshold : float (default 0.8)
          Fraction of the column total to achieve.
      sep : str (default '_')
          Separator used to join the component indices in the string label.
    
    Returns:
      labels_str : np.ndarray of shape (n,)
          String labels for each column (e.g., '1_2_23').
      sparse_membership : scipy.sparse.csr_matrix of shape (k, n)
          Sparse binary membership matrix.
      counts : np.ndarray of shape (n,)
          Number of selected components (i.e. clusters) per column.
    """
    k, n = H.shape

    # Total sum for each column (shape: (n,))
    total_sums = H.sum(axis=0)

    # Sort indices for each column in descending order of H's values.
    # sorted_indices[:, j] is a permutation of [0,1,...,k-1] such that:
    # H[sorted_indices[0, j], j] >= H[sorted_indices[1, j], j] >= ... 
    sorted_indices = np.argsort(-H, axis=0)
    # Get sorted values for each column.
    sorted_H = np.take_along_axis(H, sorted_indices, axis=0)

    # Compute cumulative sums along the sorted order (axis=0).
    cum_sums = np.cumsum(sorted_H, axis=0)
    # For each column j, we want the smallest m such that:
    # cum_sums[m, j] >= threshold * total_sums[j]
    # Broadcasting threshold*total_sums to shape (k, n):
    threshold_matrix = threshold * total_sums
    # Condition matrix: shape (k, n)
    condition = cum_sums >= threshold_matrix

    # np.argmax returns the index of the first occurrence of True along axis=0.
    # This gives us an array of indices m (shape: (n,))
    m = np.argmax(condition, axis=0)
    counts = m + 1  # number of selected components per column

    # Build a boolean mask for the sorted order: for each column j,
    # positions 0 to m[j] (inclusive) are selected.
    # r is a (k,1) array with row indices.
    r = np.arange(k)[:, None]
    selected_mask = (r <= m)  # shape (k, n)

    # For each column j, the actual component indices (in the original order)
    # that are selected are given by sorted_indices[selected_mask[:, j], j].
    # To build the sparse membership matrix, extract all selected indices:
    row_selected = sorted_indices[selected_mask]  # 1D array of selected row indices
    # np.nonzero(selected_mask) returns (row_idx, col_idx). We need col_idx.
    col_selected = np.nonzero(selected_mask)[1]
    
    # Create the sparse membership matrix (binary: 1 indicates membership)
    sparse_membership = coo_matrix(
        (np.ones_like(row_selected), (row_selected, col_selected)),
        shape=(k, n),
        dtype=bool,
    ).tocsr()

    # For string labels, for each column j, we need the list of selected components.
    # We use the fact that for column j, the selected original indices are:
    #    selected = sorted_indices[:counts[j], j]
    # We then sort these indices in ascending order (for a conventional label) and join.
    # (Using np.vectorize here is a convenience; it is not fully parallelized but avoids an explicit loop.)
    def make_label(j):
        sel = sorted_indices[:counts[j], j]
        return sep.join(map(str, np.sort(sel)))
    
    label_vectorizer = np.vectorize(make_label)
    labels_str = label_vectorizer(np.arange(n))
    
    return labels_str, sparse_membership, counts


def compute_labels_absolute(H, absolute_threshold=0.05, purity_threshold=0.5, sep='_'):
    membership = H > absolute_threshold
    membership &= (H * membership).sum(axis=0) > purity_threshold * H.sum(axis=0)
    
    labels_str = np.array([
        sep.join(map(str, 
                     np.where(membership[:, j])[0]
                     ))
        for j in range(H.shape[1])
    ])
    
    return labels_str, membership


def create_label_matrix(labels):
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    n = labels.shape[0]
    m = unique_labels.shape[0]
    data = np.ones(n, dtype=np.int8)
    row_indices = inverse  # shape: (n,)
    col_indices = np.arange(n, dtype=np.int32)
    A = csr_matrix((data, (row_indices, col_indices)), shape=(m, n))
    return A, unique_labels
