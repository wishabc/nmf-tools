import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm

from nmf_tools import in_vierstra_style
from nmf_tools.matrix.reorder import apply_order


def get_order(agst, sorted_matrix, by='primary'):
    sep = np.max(agst) + np.max(sorted_matrix) + 1
    if by == 'secondary':
        order = np.argsort(agst[0, :] * sep**2 +  agst[1, :] * sep + sorted_matrix[0])[::-1]
    elif by == 'primary':
        order = np.argsort(agst[0, :] * sep + sorted_matrix[0])[::-1]
    return order


def get_tops_and_bottoms(agst, heights):
    tops = heights.cumsum(axis=0)
    bottoms = tops - heights
    
    idxs = np.argsort(agst, axis=0)
    return np.take_along_axis(tops, idxs, axis=0), np.take_along_axis(bottoms, idxs, axis=0)


@in_vierstra_style
def plot_stacked(matrix, colors, ax=None, order_by='primary',
                 normalize=True, order=None, agst=None, orient='horizontal'):
    if normalize:
        matrix = matrix / matrix.sum(axis=0, keepdims=True)

    if agst is None:
        agst = np.argsort(matrix, axis=0)[::-1, :]
    heights = np.take_along_axis(matrix, agst, axis=0)

    if order is None:
        order = get_order(agst, heights, by=order_by)

    tops, bottoms = get_tops_and_bottoms(agst[:, order], heights[:, order])

    fb_tops = np.repeat(tops, 2, axis=1)
    fb_bottoms = np.repeat(bottoms, 2, axis=1)
    xvals = np.concatenate([[0], np.repeat(np.arange(1, matrix.shape[1]), 2), [matrix.shape[1]]])

    if ax is None:
        fig, ax = plt.subplots(figsize=(matrix.shape[1]/100, 2))
    for i, color in enumerate(colors):
        if orient == 'horizontal':
            ax.fill_between(xvals, fb_bottoms[i], fb_tops[i], lw=0, color=color)
            ax.set_xlim(0, matrix.shape[1])
            ax.set_ylim(0, None)
        elif orient == 'vertical':
            ax.fill_betweenx(xvals, fb_bottoms[i], fb_tops[i], lw=0, color=color)
            ax.set_ylim(0, matrix.shape[1])
            ax.set_xlim(0, None)

    return ax, agst, order


@in_vierstra_style
def barplot_at_scale(matrix, metadata, colors, order=None, agst=None, label_colors=None):
    assert len(metadata) == matrix.shape[1]
    
    if agst is None:
        agst = np.argsort(matrix, axis=0)[::-1, :]
    if order is None:
        sep = np.max(agst) + np.max(matrix) + 1
        max_r = matrix.max(axis=0)
        order = np.argsort(agst[0, :] * sep + max_r / matrix.sum(axis=0))[::-1]

    ordered_matrix = matrix[:, order]
    
    per_bar = 100
    chunks = np.ceil(matrix.shape[1] / per_bar).astype(int)
    
    fig, axes = plt.subplots(chunks, 1, figsize=(20, 4*chunks))
    if chunks == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=1.5)
    
    maxv = np.max(matrix.sum(axis=0))
    for k in tqdm(np.arange(chunks)):
        ax = axes[k]
        sl = slice(per_bar*k, per_bar*(k+1), 1)
        num_elements = order[sl].shape[0]
        plot_stacked(ordered_matrix[:, sl], colors, ax=ax,
                     order=np.arange(num_elements),
                     agst=agst[:, order[sl]])
        ax.set_xticks(np.arange(num_elements) + 0.5)
        ax.set_xticklabels(
            metadata.iloc[order, :]['sample_label'][sl],
            rotation=90
        )
        if label_colors is not None:
            assert len(label_colors) == matrix.shape[1]
            for xtick, col in zip(ax.get_xticklabels(), label_colors[sl]):
                xtick.set_color(col)
        ax.set_xlim(0, per_bar)
    
    for ax in axes:
        ax.set_ylim(0, maxv*1.05)

    return order, fig


@in_vierstra_style
def component_barplot(H, component_data, box_lw=0.15, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    mat = H[component_data['index'], :]
    plot_stacked(mat, component_data['color'], ax=ax, **kwargs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    for s in ax.spines.values():
        s.set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(box_lw)

    return ax