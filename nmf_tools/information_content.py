import pandas as pd
import numpy as np
import scipy.sparse as sp


def annotate_dhss_with_sample_statistics(adata, statistic, reduction='sum'):
    """
    Annotate DHSs (peaks) with sample-level statistics, aggregated over samples 
    where the DHS is present (binary matrix = 1).

    Parameters
    ----------
    adata : AnnData
        AnnData object with:
          - `adata.layers['binary']`: binary (cells x peaks) matrix.
          - `adata.obs`: sample-level metadata.
          - `adata.var_names`: peak identifiers.
    statistic : str, pandas.Series, or pandas.DataFrame
        Sample-level statistic(s) to annotate peaks with.
    reduction : {'sum', 'mean'}, default='sum'
        Aggregation method:
        - 'sum': total statistic values across samples with DHS.
        - 'mean': average statistic values across samples with DHS.

    Returns
    -------
    pandas.DataFrame
        DataFrame (n_peaks x n_statistics) with aggregated statistics per peak.
    """
    if isinstance(statistic, str):
        statistic = adata.obs[[statistic]] # 3000 x 1

    csr = sp.csr_matrix(statistic.values.T)
    
    value_sums = pd.DataFrame(
        (csr @ adata.layers['binary']).todense().A.T, 
        index=adata.var_names, 
        columns=statistic.columns
    )
    if reduction == 'mean':
        peak_sums = adata.layers['binary'].sum(axis=0).A1      # of samples with a DHS
        return value_sums / peak_sums[:, None]
    elif reduction == 'sum':
        return value_sums
    else:
        raise ValueError('Unknown reduction. Use sum or mean')


def get_annotation_by_dhs_p(adata, annotation_column, selected_annotations=None):
    """
    Compute per-annotation proportions of samples in which DHS is present.

    Parameters
    ----------
    adata : AnnData
        AnnData object with:
          - `adata.layers['binary']`: binary (cells x peaks) matrix,
            where 1 indicates the presence of a DHS in a sample.
          - `adata.obs`: sample-level metadata (rows = samples).
          - `adata.var_names`: identifiers for peaks (DHSs).
    annotation_column : pandas.Series
        Categorical sample annotation aligned with `adata.obs_names`
        (e.g., tissue, cell type).
    selected_annotations : list, optional
        Subset of annotation categories to include. If None, all unique
        categories in `annotation_column` are used.

    Returns
    -------
    pandas.DataFrame
        DataFrame of shape (n_annotations, n_peaks). Each entry is the
        proportion of samples from a given annotation in which a DHS is present.
        Index = annotation categories, columns = `adata.var_names`.
    """
    # Get an average proportion of samples for each annotation and IC
    if selected_annotations is None:
        selected_annotations = annotation_column.unique()

    
    # One hot encoded sample annotation filtered to selected annotations
    annotation_by_sample = pd.get_dummies(adata.obs[annotation_column], prefix='', prefix_sep='').T.loc[selected_annotations, adata.obs_names].astype(int)
    samples_per_annotation = annotation_by_sample.sum(axis=1) # not all samples can be used due to filtering
    
    # how many samples of the annotation have certain DHS
    annotation_by_dhs_count = annotate_dhss_with_sample_statistics(adata, annotation_by_sample.T, reduction='sum').T
    
    # what proportion of samples of the annotation have certain DHS
    annotation_by_dhs_p = annotation_by_dhs_count / samples_per_annotation.values[:, None]
    
    return annotation_by_dhs_p

def get_information_content(annotation_by_dhs_p):
    """
    Compute information content (IC) of DHSs with respect to annotations.

    Parameters
    ----------
    annotation_by_dhs_p : pandas.DataFrame
        DataFrame of shape (n_annotations, n_peaks). Entries are the
        per-annotation proportions of samples in which a DHS is present.
        Typically the output of `get_annotation_by_dhs_p`.

    Returns
    -------
    pandas.DataFrame
        DataFrame of shape (n_annotations, n_peaks). Each value reflects
        how strongly a DHS is associated with an annotation compared to the
        overall background frequency of that annotation.
    """
    annotation_p_marginal = annotation_by_dhs_p.sum(axis=1)
    
    # Normalize for IC
    annotation_by_dhs_p_normed = annotation_by_dhs_p / annotation_by_dhs_p.sum(axis=0)
    annotation_p_marginal /= annotation_p_marginal.sum()
    
    # IC = p * (log2(p) - log2(q))
    annotation_by_dhs_IC = (annotation_by_dhs_p_normed * (np.log2(annotation_by_dhs_p_normed) - np.log2(annotation_p_marginal.values[:, None]))).fillna(0)
    return annotation_by_dhs_IC
