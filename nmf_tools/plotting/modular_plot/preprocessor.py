import logging
import numpy as np
import pandas as pd

from . import LoggerMixin, DataBundle, PlotComponent

from typing import Sequence

class DataPreprocessor(LoggerMixin):
    """
    Main class for preprocessing data for modular plots.
    Can be used to load data for multiple plot components
    for a given interval.
    """
    def __init__(self, **kwargs):
        super().__init__(logger_level=kwargs.pop('logger_level', logging.INFO))

        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])


    def get_interval_data(
            self,
            interval,
            plot_components: Sequence[PlotComponent],
            **kwargs
            ):
        """
        Get the data for the specified interval and plot components.

        Parameters
        ----------
        interval : GenomicInterval
            The genomic interval to plot.
            It is used to initialize the DataBundle object
            and passed to the loaders to filter the data.

        plot_components : Sequence[PlotComponent]
            The plot components to plot.
            For each component, its required_loaders are loaded
            using the passed interval as an argument.

        Returns
        -------
        data : list[DataBundle]
            A list of DataBundle objects containing the data for each plot component
            in the order of the plot_components.
        """
        data = []

        for component in plot_components:
            required_loaders = getattr(component, 'required_loaders', [])
            data_bundle = DataBundle(interval, logger_level=self.logger.level)
            for loader_class in required_loaders:
                if loader_class not in data_bundle.processed_loaders:
                    loader = loader_class(self, interval, logger_level=self.logger.level)
                    data_bundle = loader.load(data_bundle, loader_kwargs=component.loader_kwargs, **kwargs)
                    data_bundle.processed_loaders.append(loader_class)
            data.append(data_bundle)

        return data
