import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import scipy.cluster.hierarchy as sch

from nmf_tools import in_vierstra_style
from nmf_tools.matrix_reordering.records_reordering import order_records
from nmf_tools.matrix_reordering.components_reordering import order_components
from nmf_tools.matrix_reordering.hcluster import hierarchical_clustering


def order_matrix(matrix,
                 order_components_by='primary',
                 order_components_kwargs=None,
                 order_records_by='primary',
                 order_records_kwargs=None,
                 normalize_for_plotting=True):
    
    if order_components_kwargs is None:
        order_components_kwargs = {}
    if order_records_kwargs is None:
        order_records_kwargs = {}
    
    components_order = order_components(matrix,
                            by=order_components_by,
                            **order_components_kwargs)

    records_order = order_records(matrix,
                            by=order_records_by,
                            **order_records_kwargs)
    
    if normalize_for_plotting:
        matrix = matrix / matrix.sum(axis=0, keepdims=True)
   
    sorted_matrix = components_order(matrix)
    tops = sorted_matrix.cumsum(axis=0)
    bottoms = tops - sorted_matrix

    inverse_component_order = components_order.inv
    tops = records_order(inverse_component_order(tops))
    bottoms = records_order(inverse_component_order(bottoms))
  
    return tops, bottoms, components_order, records_order


@in_vierstra_style
def plot_stacked_barplot(bottoms, tops, colors, xvals=None, ax=None, orient='horizontal'):
    assert bottoms.shape == tops.shape
    assert bottoms.shape[0] == len(colors)

    fb_tops = np.repeat(tops, 2, axis=1)
    fb_bottoms = np.repeat(bottoms, 2, axis=1)
    if xvals is None:
        xvals = np.arange(bottoms.shape[1] + 1)
    xvals = np.concatenate([[xvals[0]], np.repeat(xvals[1:-1], 2), [xvals[-1]]])

    if ax is None:
        fig, ax = plt.subplots(figsize=(len(xvals)/200, 2) if orient == 'horizontal' else (2, len(xvals)/200))
    for btms, tps, color in zip(fb_bottoms, fb_tops, colors):
        if orient == 'horizontal':
            ax.fill_between(xvals, btms, tps, lw=0, color=color)
            ax.set_xlim(0, xvals[-1])
            ax.set_ylim(0, tops.max())
        elif orient == 'vertical':
            ax.fill_betweenx(xvals, btms, tps, lw=0, color=color)
            ax.set_ylim(0, xvals[-1])
            ax.set_xlim(0, tops.max())

    return ax


@in_vierstra_style
def component_barplot(matrix, component_data, box_lw=0.15, ax=None, **kwargs):
    bottoms, tops, components_order, records_order = order_matrix(matrix, **kwargs)

    ax = plot_stacked_barplot(bottoms, tops, component_data.sort_values('index')['color'],
                              ax=ax, orient='horizontal')

    ax.set_xticks([])
    ax.set_yticks([])

    for s in ax.spines.values():
        s.set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(box_lw)

    return ax, components_order, records_order


@in_vierstra_style
def component_barplot_at_scale(
    matrix, component_data, records_labels,
    label_colors=None, **kwargs
):
    assert len(records_labels) == matrix.shape[1]
    
    tops, bottoms, components_order, records_order = order_matrix(matrix, **kwargs)
    order = records_order.col_order

    colors = component_data.sort_values('index')['color']
    
    bars_per_panel = 100
    n_chunks = np.ceil(matrix.shape[1] / bars_per_panel).astype(int)
    
    fig, axes = plt.subplots(n_chunks, 1, figsize=(20, 4*n_chunks))
    if n_chunks == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=1.5)

    
    maxv = np.max(tops)
    for k in tqdm(np.arange(n_chunks)):
        ax = axes[k]
        sl = slice(bars_per_panel*k, bars_per_panel*(k+1), 1)
        num_elements = order[sl].shape[0]

        plot_stacked_barplot(
            bottoms[:, sl], tops[:, sl], colors,
            ax=ax, orient='horizontal',
        )
        ax.set_xticks(np.arange(num_elements) + 0.5)
        ax.set_xticklabels(
            records_labels[order[sl]],
            rotation=90
        )
        ax.set_xlim(0, bars_per_panel)
        if label_colors is not None:
            assert len(label_colors) == matrix.shape[1]
            for xtick, col in zip(ax.get_xticklabels(), label_colors[sl]):
                xtick.set_color(col)
    
    for ax in axes:
        ax.set_ylim(0, maxv*1.05)

    return fig, components_order, records_order


@in_vierstra_style
def component_barplot_hcl(matrix, component_data, ax=None, separator_lw=0.5,
                          cluster_threshold=0.7, criterion='distance',
                          linkage_matrix=None, normalize_for_plotting=False,
                          **kwargs):
    if linkage_matrix is None:
        linkage_matrix = hierarchical_clustering(matrix, **kwargs)
    
    tops, bottoms, components_order, records_order = order_matrix(
        matrix,
        order_components_by='cluster',
        order_records_by='hierarchical',
        order_records_kwargs=dict(linkage_matrix=linkage_matrix),
        order_components_kwargs=dict(
            linkage_matrix=linkage_matrix,
            cluster_threshold=cluster_threshold,
            criterion=criterion
        ),
        normalize_for_plotting=normalize_for_plotting,
    )

    clusters = sch.fcluster(linkage_matrix, t=cluster_threshold, criterion=criterion)


    ax = plot_stacked_barplot(bottoms, tops, component_data.sort_values('index')['color'],
                         ax=ax, orient='horizontal')
    
    for i in np.where(np.diff(records_order(clusters)) != 0)[0]:
        ax.axvline(i+1, color='k', lw=separator_lw)

    return ax, components_order, records_order


@in_vierstra_style
def component_barplot_with_dendrogram(
    matrix, component_data, records_labels=None,
    linkage_matrix=None,
    cluster_threshold=0.7,
    criterion='distance',
    normalize_for_plotting=False,
    fig=None,
    **kwargs
):
    if records_labels is not None:
        assert len(records_labels) == matrix.shape[1]

    if linkage_matrix is None:
        linkage_matrix = hierarchical_clustering(matrix, **kwargs)

    if fig is None:
        fig = plt.figure(figsize=(20, 4))
    gs = gridspec.GridSpec(2, 1, hspace=0)

    ax1 = fig.add_subplot(gs[0])
    sch.dendrogram(linkage_matrix, color_threshold=cluster_threshold, ax=ax1)
    ax1.axhline(cluster_threshold, color='r')
    ax1.set_xticks([])

    ax2 = fig.add_subplot(gs[1])
    ax2, components_order, records_order = component_barplot_hcl(
        matrix, component_data,
        ax=ax2,
        linkage_matrix=linkage_matrix,
        cluster_threshold=cluster_threshold,
        criterion=criterion,
        normalize_for_plotting=normalize_for_plotting,
    )
    if records_labels is not None:
        ax2.set_xticks(np.arange(len(records_labels)) + 0.5)
        ax2.set_xticklabels(records_labels, rotation=90)
    else:
        ax2.set_xticks([])
    ax2.set_xlim(0, matrix.shape[1])

    return ax1, ax2, components_order, records_order
