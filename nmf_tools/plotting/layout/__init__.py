import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class BroadcastIter:
    """
    Wrapper for iterable arguments to mark them as broadcastable.

    When used with `broadcast_zip`, this wrapper signals that the
    underlying data should be advanced one element at a time on
    each loop cycle. Arguments not wrapped in `BroadcastIter` are
    repeated for every cycle instead.

    Parameters
    ----------
    data : iterable
        An iterable whose elements will be yielded one per cycle.

    Attributes
    ----------
    data : iterable
        The underlying data.
    length : int or float('inf')
        Length of the iterable if known, otherwise infinity.
    """
    def __init__(self, data):
        self.data = data
        self.length = len(data) if hasattr(data, '__len__') else float('inf')
    
    def __iter__(self):
        return iter(self.data)


def broadcast_zip(*args, **kwargs):
    """
    Zip together positional and keyword arguments with broadcasting.

    Any argument wrapped in `BroadcastIter` is treated as an iterable
    that advances one element at a time on each loop cycle. Arguments
    not wrapped are repeated for every cycle.

    Parameters
    ----------
    *args : any
        Positional arguments. Wrap in `BroadcastIter` to step through
        a sequence instead of repeating the same value.
    **kwargs : any
        Keyword arguments. Wrap in `BroadcastIter` to step through
        a sequence instead of repeating the same value.

    Yields
    ------
    tuple
        A 2-tuple `(values, kw_values)` where:
        values : tuple
            The current positional arguments for this cycle.
        kw_values : dict
            The current keyword arguments for this cycle.

    Notes
    -----
    All `BroadcastIter` instances must have the same length. Otherwise,
    `broadcast_zip` cannot align them and will raise a ValueError.
    """
    iterables = []
    lengths = set()
    
    # Handle positional arguments
    for arg in args:
        if isinstance(arg, BroadcastIter):
            iterables.append(arg.data)
            lengths.add(arg.length)
        else:
            iterables.append(itertools.repeat(arg))
    
    # Handle keyword arguments
    kw_iterables = {}
    for key, value in kwargs.items():
        if isinstance(value, BroadcastIter):
            kw_iterables[key] = iter(value.data)
            lengths.add(value.length)
        else:
            kw_iterables[key] = itertools.repeat(value)
    
    if len(lengths) > 1:
        raise ValueError("All BroadcastIterable instances must have the same length")
    
    for values in zip(*iterables):
        kw_values = {k: next(v) for k, v in kw_iterables.items()}
        yield values, kw_values


def plot_row_panels(
    plotting_function,
    broadcasted_data,
    *args,
    w_ratios='even',
    wspace=0.1,
    fig=None,
    ax=None,
    **kwargs
):
    """
    Arrange multiple plots side by side in a single row.

    Each element in `broadcasted_data` is drawn in its own subplot,
    using the provided `plotting_function`. Extra arguments can be
    broadcasted across columns with `BroadcastIter`.

    Parameters
    ----------
    plotting_function : callable
        A function with signature `plotting_function(*args, ax=axis, **kwargs)`
        that performs the actual plotting.
    broadcasted_data : iterable
        Sequence of data elements to plot, one per column.
    *args : any
        Additional positional arguments. Wrap in `BroadcastIter` to step
        through values across columns.
    w_ratios : {'even', 'proportional'} or array-like, default="even"
        Column width ratios. If 'even', all columns equal width.
        If 'proportional', widths are proportional to `len(x)` for each
        item in `broadcasted_data`.
    wspace : float, default=0.1
        Spacing between subplots (passed to `GridSpecFromSubplotSpec`).
    fig : matplotlib.figure.Figure, optional
        Figure to plot into. Defaults to current figure.
    ax : matplotlib.axes.Axes or SubplotSpec, optional
        Parent axes or subplot spec. Defaults to current axes.
    **kwargs : any
        Additional keyword arguments. Wrap in `BroadcastIter` to step
        through values across columns.

    Returns
    -------
    list of matplotlib.axes.Axes
        The list of created axes, one for each subplot.

    Notes
    -----
    - All `BroadcastIter` arguments must have the same length.
    - Useful for modular heatmaps or row layouts.
    """
    n_columns = len(broadcasted_data)
    if w_ratios == 'even':
        w_ratios = np.ones(n_columns)
    elif w_ratios == 'proportional':
        w_ratios = [len(x) for x in broadcasted_data]

    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    gs = gridspec.GridSpecFromSubplotSpec(1, n_columns, width_ratios=w_ratios, wspace=wspace, subplot_spec=ax)
    axes = []
    for i, (arg, kwarg) in enumerate(
        broadcast_zip(
            BroadcastIter(broadcasted_data),
            *args,
            **kwargs,
        )
    ):
        g_ax = fig.add_subplot(gs[i])
        axes.append(g_ax)
        plotting_function(*arg, ax=g_ax, **kwarg)
    return axes



def plot_grid_panels(
    plotting_function,
    broadcasted_grid,
    *args,
    h_ratios="even",
    w_ratios="even",
    hspace=0.1,
    wspace=0.1,
    fig=None,
    ax=None,
    **kwargs
):
    """
    Arrange multiple rows x columns of panels.

    Each element in `broadcasted_grid` is a row: an iterable of data
    elements, one per column. Within each row, data is passed to
    `plotting_function` via `plot_row_panels`. Extra arguments can be
    broadcast across rows and/or columns with `BroadcastIter`.

    Parameters
    ----------
    plotting_function : callable
        Function with signature `plotting_function(*args, ax=axis, **kwargs)`
        used for each panel.
    broadcasted_grid : iterable of iterables
        Outer iterable = rows, inner iterable = columns.
    *args : any
        Additional positional args.
    h_ratios : {'even', array-like}, default="even"
        Row height ratios.
    w_ratios : {'even', 'proportional', array-like}, default="even"
        Column width ratios, passed to each `plot_row_panels` call.
    hspace, wspace : float
        Grid spacing parameters.
    fig : matplotlib.figure.Figure, optional
        Figure to plot into. Defaults to current figure.
    ax : matplotlib.axes.Axes or SubplotSpec, optional
        Parent axes or subplot spec. Defaults to current axes.
    **kwargs : dict
        Keyword args. Can wrap in `BroadcastIter` to step across rows.

    Returns
    -------
    list of list of Axes
        A nested list: outer list = rows, inner list = axes in that row.
    """
    n_rows = len(broadcasted_grid)
    if h_ratios == "even":
        h_ratios = np.ones(n_rows)

    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    subplot_spec = ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else ax

    gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, 1, height_ratios=h_ratios, hspace=hspace, subplot_spec=subplot_spec
    )

    all_axes = []
    for i, (row_data, (args_row, kwargs_row)) in enumerate(
        zip(
            broadcasted_grid,
            broadcast_zip(*args, **kwargs)  # broadcast across rows
        )
    ):
        row_ax = fig.add_subplot(gs[i])
        row_ax.axis("off")
        axes_row = plot_row_panels(
            plotting_function,
            row_data,
            *args_row,
            fig=fig,
            ax=row_ax,
            w_ratios=w_ratios,
            wspace=wspace,
            **kwargs_row
        )
        all_axes.append(axes_row)

    return all_axes


    