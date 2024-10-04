import logging
from nmf_tools.plotting.modular_plot import DataBundle, LoggerMixin
from nmf_tools.plotting.modular_plot.interval_plot.plot_components import VerticalPlotComponent
import numpy as np
import pandas as pd

from . import LoggerMixin, DataBundle, PlotComponent

from typing import Sequence


class DataPreprocessor(LoggerMixin):
    """
    An extension of the DataPreprocessor class that can handle multiple intervals.
    The get_interval_data method is modified to accept a dict of intervals,
    where the key corresponds to the interval_key for each plot component
    to know which interval to use for that component.
    """
    def __init__(self, logger_level=None, **get_data_kwargs):
        LoggerMixin.__init__(self, logger_level=logger_level)

        for kwarg, value in get_data_kwargs.items():
            setattr(self, kwarg, value)

    @staticmethod
    def _parse_interval(interval, interval_key: str):
        """
        Parse the interval argument to a GenomicInterval object.
        If a dict is provided, the interval_key is used to extract the interval.
        """
        if isinstance(interval, dict):
            try:
                return interval[interval_key]
            except KeyError:
                raise ValueError(f"Interval key '{interval_key}' not found.")
        else:
            return interval

    def get_interval_data(
            self,
            interval,
            plot_components: Sequence[VerticalPlotComponent],
            **kwargs
            ):
        """
        Get the data for the specified interval(s) and plot components.

        Parameters
        ----------
        interval : GenomicInterval or dict
            The genomic interval to plot.
            If a dict is provided, the component-specific interval key is used to extract the interval.

        plot_components : Sequence[VerticalPlotComponent]
            The vertical plot components to plot.

        Returns
        -------
        data : list[DataBundle]
            A list of DataBundle objects containing the data for each plot component
        """
        data = []

        for component in plot_components:
            component_interval = self._parse_interval(interval, getattr(component, 'interval_key', None))
            data_bundle = DataBundle(
                component_interval,
                logger_level=self.logger.level
            )

            required_loaders = component.__required_loaders__
            for loader_class in required_loaders:
                if loader_class not in data_bundle.processed_loaders:
                    loader = loader_class(self, component_interval, logger_level=self.logger.level)
                    passed_kwargs = {k: v for k, v in kwargs.items() if k in loader.get_fullargspec()}
                    loader_kwargs = {**component.loader_kwargs, **passed_kwargs}
                    data_bundle = loader.load(data_bundle, **loader_kwargs)
                    data_bundle.processed_loaders.append(loader_class)
            data.append(data_bundle)

        return data

