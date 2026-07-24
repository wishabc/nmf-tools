def get_category_inverse_weights(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_to_count = dict(zip(unique_labels, counts))
    count_array = np.vectorize(label_to_count.get)(labels)
    w = 1 / count_array
    return label_to_count, w


def normalize_weights(weights, cap_coef=None, cap=None):
    """
    Normalize and optionally cap a set of weights.

    This function rescales the given weight vector so that the sum of the 
    normalized weights equals the length of the vector (i.e., the average weight 
    becomes 1). Optionally, individual weights can be capped before normalization.

    Parameters
    ----------
    weights : array_like
        Input vector of positive weights to be normalized.
    cap_coef : float, optional
        Multiplicative coefficient used to determine the cap if `cap` is not 
        provided. The cap is computed as:
            cap = min(weights) * cap_coef
        If both `cap` and `cap_coef` are None, no capping is applied.
    cap : float, optional
        Explicit cap value applied elementwise to the weights. If provided, 
        `cap_coef` is ignored. If None, and `cap_coef` is also None, no capping 
        is applied.

    Returns
    -------
    cap : float
        The effective cap value used (either the provided `cap`, or computed from 
        `cap_coef`, or ∞ if no capping).
    w : numpy.ndarray
        The normalized weight vector, after applying capping and rescaling such that:
            sum(w) = len(w)
    """
    if cap_coef is None and cap is None:
        w = weights
        cap = np.inf
    else:
        if cap is None:
            cap = np.min(weights) * cap_coef
        w = np.minimum(weights, cap)

    w = w / w.sum() * len(w)
    return cap, w


def get_capped_weights_times_spot(labels, spots, cap_coef=None, cap=None):
    _, w = get_category_inverse_weights(labels)
    w *= spots
    cap, w = normalize_weights(w, cap_coef, cap=cap)
    print(w.max(), w.min())
    return cap, w