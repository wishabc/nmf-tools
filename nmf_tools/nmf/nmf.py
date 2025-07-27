import numpy as np
from sklearn.base import clone

from .weighted_NMF import WeightedNMF
import scipy.sparse as sp
from nmf_tools.matrix_reordering.components_reordering import order_components_by_template

dtype = np.float64


def get_transform_params_WH_to_ref(W_new, H_new, W_ref, H_ref):
    reorder = order_components_by_template(W_new, W_ref).row_order
    coefs = (W_new * W_new)[reorder, :].sum(axis=1) / (W_ref * W_ref).sum(axis=1)
    coefs /= (H_new * H_new)[reorder, :].sum(axis=1) / (H_ref * H_ref).sum(axis=1)
    coefs = np.sqrt(coefs)
    return reorder, coefs

def apply_transform_WH(W, H, transform_params):
    reorder, coefs = transform_params
    return W[reorder, :] / coefs[:, None], H[reorder, :] * coefs[:, None]


def data_to_sparse(X: np.ndarray) -> sp.csr_matrix:
    if sp.issparse(X):
        return X.astype(dtype)
    return sp.coo_matrix(X).tocsr().astype(dtype)


def get_mock_weights(X: sp.csr_matrix, which='W'):
    if which == 'W':
        shape = X.shape[0]
    elif which == 'H':
        shape = X.shape[1]
    else:
        raise ValueError(f'Unknown weights type: {which}')
    return np.ones(shape, dtype=dtype)


def validate_input_args(func):
    def wrapper(self, X, *args, W_weights=None, H_weights=None, **kwargs):
        X = data_to_sparse(X)
        if W_weights is None:
            W_weights = get_mock_weights(X, which='W')
        if H_weights is None:
            H_weights = get_mock_weights(X, which='H')
        assert W_weights.shape[0] == X.shape[0]
        assert H_weights.shape[0] == X.shape[1]
        assert W_weights.ndim == 1 and H_weights.ndim == 1, 'Weights are expected to be 1D arrays'
        return func(self, X=X, *args, W_weights=W_weights, H_weights=H_weights, **kwargs)

    return wrapper
    

class NMFModel:
    def __init__(self, n_components, extra_params: dict=None):
        params = dict(
            n_components=n_components,
            solver='mu',
            beta_loss='frobenius',
            random_state=0,
            init="nndsvda",
            max_iter=1000,
            tol=1e-4,
            alpha_W=0.0,
            l1_ratio=1.0,
            verbose=True
        )
        if extra_params:
            overwritten = {
                key: f'New: {extra_params[key]}. Old: {params[key]}' for key in extra_params
                if key in params and extra_params[key] != params[key]
            }
            params.update(extra_params)
            if overwritten:
                print("Overwritten params:", overwritten, flush=True)
    
        self.model = WeightedNMF(**params)

    def _run_fit_transform(self, X, *, 
                        H=None, W=None, W_weights=None, H_weights=None, error_at_init=None,
                        update_H=True):
        W, *_ = self.model._fit_transform(
            X=X,
            H=H,
            W=W,
            update_H=update_H,
            W_weights=W_weights[:, None],
            H_weights=H_weights[None, :],
            error_at_init=error_at_init
        )
        return W 

    @validate_input_args
    def fit_transform(self, X, *, W=None, H=None, W_weights=None, H_weights=None, error_at_init=None):
        """
        X: samples x peaks
        W: samples x components
        H: components x peaks
        NMF: X = W @ H
        NMF: samples x peaks = samples x components @ components x peaks
        """
        W = self._run_fit_transform(
            X=X,
            H=H,
            W=W,
            update_H=True,
            W_weights=W_weights,
            H_weights=H_weights,
            error_at_init=error_at_init
        )

        H = self.model.components_ # components x peaks
        return W, H

    @validate_input_args
    def reconstruction_error(self, X, *, W=None, H=None, W_weights=None, H_weights=None, **model_kwargs):
        model: WeightedNMF = clone(self.model)
        model = model.set_params(**model_kwargs)
        W, H = model._check_w_h(X, W, H, update_H=False)

        return model.reconstruction_error(
            X=X,
            W=W,
            H=H,
            W_weights=W_weights[:, None],
            H_weights=H_weights[None, :]
        )

    @validate_input_args
    def project_samples(self, X, H, *, W=None, W_weights=None, H_weights=None, error_at_init=None):
        """
        X: samples x peaks
        H: components x peaks
        NMF: X = W @ H
        NMF: samples x peaks = samples x components * components x peaks
        """
        W = self._run_fit_transform(
            X=X,
            H=H,
            W=W,
            update_H=False,
            W_weights=W_weights,
            H_weights=H_weights,
            error_at_init=error_at_init
        )
        return W # samples x components 

    @validate_input_args
    def project_peaks(self, X, W, *, H=None, W_weights=None, H_weights=None, error_at_init=None):
        """
        X: samples x peaks
        W: samples x components
        NMF: X.T = H.T @ W.T
        NMF: peaks x samples = peaks x components * components x samples
        """
        projected_peaks, *_ = self.model._fit_transform(
            X=X.T,
            H=W.T,
            W=None if H is None else H.T,
            update_H=False,
            W_weights=H_weights,
            H_weights=W_weights,
            error_at_init=error_at_init
        )
        return projected_peaks.T

    @validate_input_args
    def update_WH(self, X_initial, X_new, W, H, *, W_weights=None, H_weights=None):
        X_new = data_to_sparse(X_new)
        old_n_samples = X_initial.shape[0]
        X = sp.vstack([X_initial, X_new])
        new_W = self.project_samples(X, H, W_weights=W_weights, H_weights=H_weights)

        transform_params = get_transform_params_WH_to_ref(
            new_W[:old_n_samples, :],
            new_H,
            W,
            H
        )


        new_H = self.project_peaks(
            X, new_W,
            W_weights=W_weights,
            H_weights=H_weights
        )

        new_W[:old_n_samples, :], new_H = get_transform_params_WH_to_ref(
            new_W[:old_n_samples, :], 
            new_H,
            W,
            H
        )

        return new_W, new_H
    
