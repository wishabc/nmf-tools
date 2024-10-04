from typing import Sequence
from collections import namedtuple
from matplotlib import gridspec
from matplotlib import pyplot as plt

import pandas as pd

from .plot_components import VerticalPlotComponent

from nmf_tools.plotting.modular_plot import LoggerMixin, PlotComponent


class IntervalPlotter(LoggerMixin):
    """
    Class to plot a genomic interval with multiple vertical plot components.

    Parameters
    ----------
    plot_components : Sequence[VerticalPlotComponent]
        The vertical plot components to plot in the interval.

    inches_per_unit : float, optional
        The number of inches per unit for the figure size.
        Default is 1.0.

    width : float, optional
        The width of the figure in inches * inches_per_unit.
        Default is 2.5.

    Usage
    -----
    # Define the preprocessor
    preprocessor = MultipleIntervalDataPreprocessor(...)

    # Define plot components
    plot_components = [
        IdeogramComponent(height=0.1, margins=(0.1, 0.1), interval_key='gene'),
        GencodeComponent(height=0.2, margins=(0.1, 0.1), interval_key='gene'),
        FinemapComponent(height=0.2, margins=(0.1, 0.1), interval_key='dhs'),
        DNaseTracksComponent(height=1.5, margins=(0.1, 0.1), interval_key='dhs', component_data=component_data),
        DHSIndexComponent(height=0.1, margins=(0.1, 0.0), interval_key='dhs'),
        DHSLoadingsComponent(height=0.2, margins=(0.0, 0.1), interval_key='dhs', component_data=component_data),
        FootprintsComponent(height=0.1, margins=(0.1, 0.1), interval_key='footprint'),
        MotifComponent(height=0.2, margins=(0.1, 0.1), interval_key='footprint'),
        CAVComponent(height=0.2, margins=(0.1, 0.1), interval_key='footprint'),
    ]

    # Create an interval plotter
    interval_plotter = IntervalPlotter(plot_components)

    # Get the data for the interval
    data = preprocessor.get_interval_data(interval, plot_components)

    # Plot the interval
    component_axes = interval_plotter.plot_interval(data)
    """
    def __init__(self, plot_components: Sequence[VerticalPlotComponent],
                 inches_per_unit=1.0, width=2.5, **kwargs):
        super().__init__(**kwargs)
        self.component_names = [c.name for c in plot_components]

        repeating_names = self._check_unique_component_names(self.component_names)
        if len(repeating_names) > 0:
            message = f"""Component names must be unique. 
            Please, explicitly set the name for each repeating component class. 
            Repeating component names: {repeating_names}"""
            self.logger.error(message)
            raise ValueError(message)

        self.CompTuple = namedtuple('ComponentNamesTuple', self.component_names)

        self.plot_components = self.CompTuple(*plot_components)
        self.inches_per_unit = inches_per_unit
        self.width = width

        self.gridspec = self.setup_default_gridspec()

    @staticmethod
    def _check_unique_component_names(names):
        """
        Check if the component names are unique. And return the non-unique names.
        """
        value_counts = pd.Series(names).value_counts()
        return value_counts[value_counts > 1].index.tolist()

    @staticmethod
    def _convert_component_to_name(component):
        """
        Convert a vertical plot component to its name.
        """
        if isinstance(component, PlotComponent):
            return component.name
        elif isinstance(component, type) and issubclass(component, PlotComponent):
            return component.__name__
        else:
            assert isinstance(component, str)
            return component
        
    def _sort_components(self, components: Sequence[str]):
        """
        Sort the components in the order of self.plot_components.
        """
        if not all(c in self.component_names for c in components):
            self.logger.error(f"Invalid component names. Must be one of the following: {self.component_names}")
            raise AssertionError
        return [c for c in self.component_names if c in components]

    def setup_default_gridspec(self):
        """
        Setup the GridSpec for the vertical components in the figure.
        Components are plotted in a vertical stack with specified heights and margins.
        """
        height_ratios = [
            x
            for c in self.plot_components
            for x in [c.margin_top, c.height, c.margin_bottom]
        ]
        return gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0)

    def get_all_component_gridspecs(self):
        """
        Get the GridSpecs for each vertical plot component.
        """
        return self.CompTuple(*[self.gridspec[3 * i + 1, :] for i in range(len(self.plot_components))])
    
    def get_gridspec_for_component(self, component, include_top_margin=False,
                                   include_bottom_margin=False):
        """
        Get the GridSpec slice for a specific vertical plot component.
        """
        index = self.component_names.index(self._convert_component_to_name(component))
        start = 3 * index if include_top_margin else 3 * index + 1
        end = 3 * index + 3 if include_bottom_margin else 3 * index + 2
        return self.gridspec[start:end, :]
    
    def setup_default_figure(self):
        """
        Setup a default figure with the appropriate size for the vertical components.
        """
        return plt.figure(figsize=(
            self.inches_per_unit * self.width,
            sum(x for c in self.plot_components for x in [c.margin_top, c.height, c.margin_bottom]) * self.inches_per_unit
        ))

    def plot_interval(self, data, fig=None, gridspecs=None, **kwargs):
        """
        Plot the genomic interval with all the provided vertical plot components.

        Parameters
        ----------
        data : Sequence[DataBundle] # Named tuple
            The data bundles for each vertical plot component.
            Should be a named tuple with the same names as the component names.
        
        fig : Figure, optional
            The figure to plot the interval.
            If None, a new figure is created.

        gridspecs : Sequence[GridSpec], optional
            The GridSpecs for each vertical plot component.
            If None, the default GridSpecs are used.

        Usage
        -----
        # Get the data for the interval
        get_interval_data(interval, plot_components)
        # Plot the interval
        component_axes = interval_plotter.plot_interval(data)
        """
        if fig is None:
            fig = self.setup_default_figure()
        
        component_axes = []

        if gridspecs is None:
            gridspecs = self.get_all_component_gridspecs()

        assert len(gridspecs) == len(self.plot_components) == len(data)

        for gs, component, data_bundle in zip(gridspecs, self.plot_components, data):
            ax = fig.add_subplot(gs)
            component_axes.append(component.plot(data_bundle, ax=ax, **kwargs))
        component_axes = self.CompTuple(*component_axes)

        return component_axes

