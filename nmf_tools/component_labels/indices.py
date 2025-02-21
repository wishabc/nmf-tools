import numpy as np

def get_indices_by_cluster(
    original_indices,
    cluster_labels,
    min_dhs_per_cluster=1,
    max_dhs_per_cluster='all',
    random_seed=0
):
    """
    A faster version of get_indices_by_cluster that supports string cluster labels.

    Parameters
    ----------
    original_indices : np.ndarray
        Array of shape (N,). These are the 'original' indices of your data points.
        Often just np.arange(N).
    cluster_labels : np.ndarray
        Array of shape (N,) with cluster labels for each data point. Can be strings or numbers.
    min_dhs_per_cluster : int
        Clusters must have at least this many data points to be considered.
        Clusters with fewer points are discarded.
    max_dhs_per_cluster : int or 'all'
        How many points to sample per cluster. If 'all', all points in the cluster are used.
        If more than the number of points in the cluster, all points are used.
    random_seed : int
        Seed for reproducible sampling.

    Returns
    -------
    dhs_indices : list of np.ndarray
        A list of arrays, one array per valid cluster, with the *original* data indices
        (from original_indices) that were sampled or selected.
    selected_clusters : np.ndarray
        The labels of the clusters that passed the `min_dhs_per_cluster` threshold.
        Preserves the original label type (string or numeric).
    """

    # Ensure inputs are NumPy arrays
    original_indices = np.asarray(original_indices)
    cluster_labels = np.asarray(cluster_labels)
    N = original_indices.shape[0]

    # Step 1: Map string labels to numeric labels if necessary
    unique_labels, inverse_indices = np.unique(cluster_labels, return_inverse=True)
    # unique_labels: array of unique cluster labels (strings or numbers)
    # inverse_indices: array of numeric labels corresponding to clusters_s

    # Step 2: Sort by numeric cluster label
    sort_order = np.argsort(inverse_indices)
    clusters_numeric_sorted = inverse_indices[sort_order]
    sorted_original_indices = original_indices[sort_order]

    # Step 3: Find the boundaries for each cluster in the sorted array
    boundaries = np.r_[
        0,
        np.flatnonzero(np.diff(clusters_numeric_sorted) != 0) + 1,
        N
    ]

    # Initialize results
    dhs_indices = []
    selected_clusters_numeric = []

    # Initialize RNG
    rng = np.random.default_rng(seed=random_seed)

    # Step 4: Iterate through each cluster
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        stop = boundaries[i + 1]

        # Numeric cluster label
        c_numeric = clusters_numeric_sorted[start]

        # Number of points in the cluster
        cluster_size = stop - start

        # Filter based on min_dhs_per_cluster
        if cluster_size < min_dhs_per_cluster:
            continue

        # Decide number of points to sample
        if max_dhs_per_cluster == 'all':
            chosen_indices = sorted_original_indices[start:stop]
        else:
            n_i = min(max_dhs_per_cluster, cluster_size)
            # Ensure the indices are sorted for consistency
            cluster_data_indices = np.sort(sorted_original_indices[start:stop])
            chosen_indices = rng.choice(cluster_data_indices, size=n_i, replace=False)
            chosen_indices = np.sort(chosen_indices)  # Optional: sort the sampled indices

        # Append results
        dhs_indices.append(chosen_indices)
        selected_clusters_numeric.append(c_numeric)

    # Step 5: Map numeric cluster labels back to original labels
    selected_clusters = unique_labels[selected_clusters_numeric]

    return dhs_indices, selected_clusters
