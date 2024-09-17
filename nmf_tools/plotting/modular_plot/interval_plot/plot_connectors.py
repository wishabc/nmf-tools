from typing import Literal
import matplotlib.pyplot as plt

from nmf_tools.plotting.modular_plot import LoggerMixin
from nmf_tools.plotting.modular_plot.interval_plot import IntervalPlotter
from nmf_tools.plotting.connectors import connect_axes_lines, connect_axes_area


class VerticalAxesConnector(LoggerMixin):
    """
    Connector to connect two vertical plot components.
    
    Parameters
    ----------
    component1 : str
        The name of the first vertical plot component.
    component2 : str
        The name of the second vertical plot component. Should be below component1.
    
    type : str, optional
        The type of connector. Must be one of 'line' or 'area'.
        Default is 'area'.
        'line' is used to connect single points across the components with lines.
        'area' is used to connect intervals across the components with shaded areas.

    **kwargs
        Additional keyword arguments to pass to the connector plot method.
        The arguments are connector-specific.
        x : array-like, optional
            The x-coordinates to connect with lines.
            Required for 'line' connectors.
        x1 : array-like, optional
        x2 : array-like, optional
            The x-coordinates to connect with areas.
        interval : GenomicInterval, optional
            The interval to connect with areas.
            Overrides x1 and x2 if provided.
            Either x1 and x2 or interval is required for 'area' connectors.
        extend_to_top : bool, optional
            If True, the connector extends to the top of the top component axes.
            Default is False.
        extend_to_bottom : bool, optional
            If True, the connector extends to the bottom of the bottom component axes.
            Default is False.
        Other keyword arguments are passed to the connector plot method:
        connect_axes_lines or connect_axes_area.
        And then to the Line2D or Polygon object created for the connector.

    Usage
    -----
    # Create a connector between two components
    connector = VerticalAxesConnector('ComponentA', 'ComponentB', type='area', facecolor='green', alpha=0.5)
    connector.plot(component_axes)
    """
    def __init__(self, 
                component1: str,
                component2: str,
                type: Literal['line', 'area'] = 'area',
                 **kwargs):
        super().__init__(kwargs.pop('logger_level', None))

        self.component1 = component1
        self.component2 = component2
        self.type = type
        
        self.connector_kwargs = kwargs

    def get_component_axes(self, component_axes, component):
        """
        Get the axes object for a specific component.
        If the component's plot method returns multiple axes, return the last one.
        """
        try:
            axes = getattr(component_axes, component)
            if not isinstance(axes, plt.Axes):
                assert len(axes) > 0
                return axes[-1]
            return axes
        except AttributeError:
            self.logger.error(f"Component axes do not have attribute {component}.")
            raise

    def _validate_connector_kwargs(self, x, x1, x2, interval):
        """
        Validate the connector kwargs based on the connector type.
        """
        valid = True
        if self.type == 'line':
            if x is None:
                message = "x argument is required for line connectors."
                valid = False
        elif self.type == 'area':
            if (x1 is None or x2 is None) and interval is None:
                message = "Either x1 and x2 or interval argument is required for area connectors."
                valid = False
        else:
            message = f"Invalid connector type: {self.type}. Must be one of 'line' or 'area'."
            valid = False
        if not valid:
            self.logger.error(message)
            raise ValueError(message)

    def plot(self, component_axes, x=None, x1=None, x2=None, interval=None, **kwargs):
        """
        Plot the connector between the two components.

        Parameters
        ----------
        component_axes : namedtuple
            The axes objects for the vertical plot components.
            The axes objects should be named the same as the component names
            (e.g. component_axes.ComponentA, component_axes.ComponentB).

        x : array-like, optional
            The x-coordinates to connect with lines.
            Required for 'line' connectors.

        x1 : array-like, optional
        x2 : array-like, optional
            The x-coordinates to connect with areas.
            Required for 'area' connectors.
        
        interval : GenomicInterval, optional
            The interval to connect with areas.
            Overrides x1 and x2 if provided.
            Either x1 and x2 or interval is required for 'area' connectors.

        **kwargs
            Additional keyword arguments to pass to the connector plot method.
            Overrides the VerticalAxesConnector.connector_kwargs.
            See VerticalAxesConnector for more details.
        """
        ax1 = self.get_component_axes(component_axes, self.component1)
        ax2 = self.get_component_axes(component_axes, self.component2)

        kwargs.update(self.connector_kwargs)

        self._validate_connector_kwargs(x, x1, x2, interval)

        if self.type == 'line':
            connect_axes_lines(ax1, ax2, x, **kwargs)
        elif self.type == 'area':
            if interval is not None:
                x1, x2 = interval.start, interval.end
            connect_axes_area(ax1, ax2, x1, x2, **kwargs)


class ConnectorPlotter(LoggerMixin):
    """
    Class to manage and plot connectors between multiple vertical plot components.

    Parameters
    ----------
    interval_plotter : IntervalPlotter
        The IntervalPlotter object that contains the vertical plot components.
    
    Methods
    -------
    add_connectors(*components, type='area', extend_to_top=False, extend_to_bottom=False, **kwargs)
        Add a connector between two or more vertical plot components.
        See VerticalAxesConnector for more details.
        Components are automatically sorted based on their order in the IntervalPlotter.
        extend_to_top and extend_to_bottom are only applied to the first and last components, respectively,
        and the connectors are automatically extended to the intermediate components.

    plot(component_axes, connectors=None, **kwargs)
        Plot the connectors between the vertical plot components for the given axes and connectors.

    Usage
    -----
    # Create a connector plotter
    connector_plotter = ConnectorPlotter(interval_plotter)
    # Add connectors between components
    connectors = connector_plotter.add_connectors('ComponentA', 'ComponentB', type='line', x=[1, 2, 3])
    # Plot the connectors
    connector_plotter.plot(component_axes, connectors)
    """
    def __init__(self, interval_plotter: IntervalPlotter, **kwargs):
        super().__init__(**kwargs)
        self.interval_plotter = interval_plotter
        self.connectors = []

    def add_connectors(self, *components, type='area',
                       extend_to_top=False, extend_to_bottom=False,
                       **kwargs):
        """
        Add a connector between two vertical plot components.
        
        Parameters
        ----------
        components : Sequence[Union[str, VerticalPlotComponent, type]]
            The names of the vertical plot components to connect.
            The components can be specified as strings, VerticalPlotComponent objects, or their classes.
            At least two components are required to add a connector.

        type : str, optional
            The type of connector. Must be one of 'line' or 'area'.
            Default is 'area'.
            'line' is used to connect single points across the components with lines.
            'area' is used to connect intervals across the components with shaded areas.

        extend_to_top : bool, optional
            If True, the connector extends to the top of the first component axes.
            Default is False.

        extend_to_bottom : bool, optional
            If True, the connector extends to the bottom of the last component axes.
            Default is False.

        **kwargs
            Additional keyword arguments to pass to the connector plot method.
            See VerticalAxesConnector for more details.

        Returns
        -------
        connectors : list[VerticalAxesConnector]
            The connectors that were added.

        Usage
        -----
        # Add connectors between components
        connectors = connector_plotter.add_connectors('ComponentA', 'ComponentB', type='line', x=[1, 2, 3])
        """
        components = [self.interval_plotter._convert_component_to_name(c) for c in components]
        components = self.interval_plotter._sort_components(components)
        n = len(components)
        if n < 2:
            self.logger.error("At least two components are required to add a connector.")
            raise ValueError("At least two components are required to add a connector.")
        connectors = []
        for i in range(n - 1):
            component1 = components[i]
            component2 = components[i + 1]
            connector_kwargs = kwargs.copy()
            connector_kwargs.update(dict(
                extend_to_top=extend_to_top if i == 0 else False,
                extend_to_bottom=extend_to_bottom if i == n - 2 else True,
            ))
            connectors.append(VerticalAxesConnector(
                component1, component2,
                type=type,
                **connector_kwargs,
            ))
        self.connectors.extend(connectors)
        return connectors

    def plot(self, component_axes, connectors=None, **kwargs):
        """
        Plot the connectors between the vertical plot components.

        Parameters
        ----------
        component_axes : namedtuple
            The axes objects for the vertical plot components.
            The axes objects should be named the same as the component names
            (e.g. component_axes.ComponentA, component_axes.ComponentB).
        
        connectors : Sequence[VerticalAxesConnector], optional
            The connectors to plot.
            If None, all connectors added with add_connectors are plotted.

        **kwargs
            Additional keyword arguments to pass to the connector plot method.
            Overrides the connector-specific kwargs.
            See VerticalAxesConnector for more details.

        Usage
        -----
        # Plot the connectors
        # For line connectors
        connector_plotter.plot(component_axes, line_connectors, x=[1, 2, 3], color='red')
        # For area connectors
        connector_plotter.plot(component_axes, area_connectors, x1=2, x2=4, facecolor='green', alpha=0.5)
        # Or use intervals
        connector_plotter.plot(component_axes, area_connectors, interval=GenomicInterval('chr1', 2, 4), facecolor='green', alpha=0.5)
        """
        if connectors is None:
            connectors = self.connectors

        assert all(c in self.connectors for c in connectors)
        
        for connector in connectors:
            connector.plot(component_axes, **kwargs)
