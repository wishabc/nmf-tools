from nmf_tools.plotting.modular_plot import PlotComponent, uses_loaders
from matplotlib.offsetbox import TextArea, VPacker, HPacker, AnnotationBbox

import numpy as np

from .loaders import *

from typing import Union, Sequence
from matplotlib import gridspec
import matplotlib.pyplot as plt

from genome_tools.plotting import signal_plot, segment_plot
from genome_tools.plotting.gene_annotation import gene_annotation_plot
from genome_tools.plotting.ideogram import ideogram_plot
from genome_tools.plotting.utils import clear_spines
from genome_tools.plotting.pwm import plot_motif_logo
from genome_tools.utils.signal import smooth_and_aggregate_per_nucleotide_signal
from genome_tools.plotting.colors.cm import get_vocab_color

from nmf_tools import in_vierstra_style
from nmf_tools.plotting.matrices_barplots import component_barplot


class VerticalPlotComponent(PlotComponent):
    """
    An extension of the PlotComponent class that plots data vertically.

    Accepts height and margins arguments to control the size of the plot,
    that are only used when plotting with the IntervalPlotter.plot_interval method.
    Additionaly, the interval_key argument can be used to specify different
    intervals for each plot component.
    Name field is used by the IntervalPlotter to refer to the plot component,
    defaulting to the class name.
    """
    def __init__(self, 
                 height: float = 1.0, 
                 margins: Union[float, Sequence[float]] = 0.1, 
                 interval_key=None,
                 **kwargs):
        self.height = height
        self.margin_top, self.margin_bottom = self._parse_margins(margins)
        self.interval_key = interval_key

        super().__init__(**kwargs)


    def _parse_margins(self, margins):
        """Helper function to parse margins input."""
        if isinstance(margins, Sequence) and not isinstance(margins, str):
            if len(margins) == 2:
                return margins
            else:
                raise ValueError("If margins is a sequence, it must have exactly two elements.")
        return margins, margins


    def _plot(self, data, ax, **kwargs):
        """
        Abstract plot method to be implemented by specific plot components.
        """
        raise NotImplementedError("Plot method should be implemented in subclasses.")
    
    
    @staticmethod
    def set_xlim_interval(func):
        """
        Decorator to set the x-axis limits for the plot.
        The interval attribute of the data object is used to set the limits.
        """
        def wrapper(self, data, ax, **kwargs):
            # TODO: figure out ho to store data with named arguments
            # if not hasattr(data, 'interval') or not data.interval:
            #     raise ValueError("Data must have 'interval' attribute.")
            interval = data.interval
            ax.set_xlim(interval.start, interval.end)
            return func(self, data, ax, **kwargs)
        return wrapper
    

    @staticmethod
    def add_axes_at_middle_points(genomic_intervals, interval, bp_width=None, summit_field='dhs_summit', ax=None):
        """
        Add axes at the middle points of the genomic intervals.

        Parameters
        ----------
        genomic_intervals : list of GenomicInterval
            List of genomic intervals.
        interval : GenomicInterval
            Interval to restrict the axes.
        bp_width : float, optional
            Width in base pairs around the summit to plot. If None, plot the whole genomic interval.
        summit_field : str, optional
            Field of the genominc_intervals to use as the summit
            to plot around. Default is 'dhs_summit'. Only used if bp_width is not None.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, use the current axes.

        Returns
        -------
        axes : list of matplotlib.axes.Axes
            List of axes added at the middle points of the genomic intervals.
        """
        # bp_width is None: add axes the size of the region, otherwise - plot bp_width around summit
        if ax is None:
            ax = plt.gca()

        axes = []
        for genomic_interval in genomic_intervals:
            parent_pos = ax.get_position()

            try:
                summit = getattr(genomic_interval, summit_field)
            except AttributeError:
                summit = (genomic_interval.start + genomic_interval.end) / 2

            if bp_width is None:
                x0_rel = (genomic_interval.start - interval.start) / (interval.end - interval.start)
                x1_rel = (genomic_interval.end - interval.start) / (interval.end - interval.start)
            else:
                ax_width = bp_width / (interval.end - interval.start)
                x0_rel = (summit - interval.start) / (interval.end - interval.start) - (ax_width / 2)
                x1_rel = x0_rel + ax_width

            new_axes_width = parent_pos.width * (x1_rel - x0_rel)
            new_axes_height = parent_pos.height
            new_axes_x = parent_pos.x0 + (x0_rel * parent_pos.width)
            new_axes_y = parent_pos.y0

            new_ax = ax.get_figure().add_axes([new_axes_x, new_axes_y, new_axes_width, new_axes_height])
            new_ax.set_xticks([])
            new_ax.set_yticks([])
            clear_spines(new_ax)

            axes.append(new_ax)
        return axes


class SingleBPObjectsComponent(VerticalPlotComponent):
    """
    A vertical plot component that plots single base pair objects
    within a genomic interval.
    self.plot accepts a data object with 'positions' and 'values' fields
    to plot them on the x-axis and y-axis, respectively as scatter points
    and vertical lines from the x-axis.

    kwargs are passed to the scatter plot function.
    """
    @in_vierstra_style
    @VerticalPlotComponent.set_xlim_interval
    def _plot(self, data, ax, **kwargs):
        self.plot_single_bp_objects(data.positions, data.values, data.interval, ax=ax, **kwargs)
        return ax
    
    @staticmethod
    def plot_single_bp_objects(variant_intervals, interval, ax=None, s=1.0, **kwargs):
        if ax is None:
            ax = plt.gca()

        positions = [v.pos - 0.5 for v in variant_intervals]
        values = [v.value for v in variant_intervals]
        colors = [getattr(v, 'color', 'k') for v in variant_intervals]
        annotations = [getattr(v, 'annotation', None) for v in variant_intervals]

        ax.scatter(x=positions, y=values, s=s, c=colors, **kwargs)
        for val, pos, annotation, color in zip(values, positions, annotations, colors):
            ax.plot([pos, pos], [0, val], lw=0.5, color=color)
            if annotation is not None:
                ax.text(s=annotation, x=pos + (interval.end - interval.start) * 0.01,
                        y=val, fontsize=5, ha='left', va='center')
        if all(val >= 0 for val in values):
            ax.set_ylim(0, max(values) * 1.1)
        return ax
    
    @staticmethod
    def annotate_variant_alleles(variant_intervals, ax=None, box_alignment=(0.5, -0.6), **kwargs):
        if ax is None:
            ax = plt.gca()

        for v in variant_intervals:
            x = v.pos - 0.5
            y = v.value
            a1, a2 = v.ref, v.alt

            text_areas = []
            for text, color in zip([a1, '/', a2], [get_vocab_color(a1, 'dna'), 'k', get_vocab_color(a2, 'dna')]):
                text_area = TextArea(text, textprops=dict(color=color, fontsize=5, ha='center', va='center'))
                text_areas.append(text_area)
            
            hp = HPacker(children=text_areas, align="bottom", pad=0, sep=1)
            ab = AnnotationBbox(hp, (x, y), frameon=False, box_alignment=box_alignment, xycoords='data', **kwargs)
            ax.add_artist(ab)
            
        return ax


class SegmentPlotComponent(VerticalPlotComponent):
    __intervals_attr__ = 'intervals'

    @in_vierstra_style
    @VerticalPlotComponent.set_xlim_interval
    def _plot(self, data, ax, **kwargs):
        segment_plot(data.interval, getattr(data, self.__intervals_attr__), ax=ax, **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        clear_spines(ax)
        return ax
