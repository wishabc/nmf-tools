
from nmf_tools.plotting.modular_plot.interval_plot.plot_components import IntervalPlotComponent
from nmf_tools.plotting.modular_plot.shared import LoggerMixin, DataBundle

from genome_tools import GenomicInterval

from typing import Sequence


class IntervalDataPreprocessor(LoggerMixin):
    """
    An extension of the DataPreprocessor class that can handle multiple intervals.
    The get_interval_data method is modified to accept a dict of intervals,
    where the key corresponds to the interval_key for each plot component
    to know which interval to use for that component.
    """
    def __init__(self, logger_level=None, **data_kwargs):
        LoggerMixin.__init__(self, logger_level=logger_level)

        self.data_kwargs = data_kwargs
        # for kwarg, value in get_data_kwargs.items():
        #     setattr(self, kwarg, value)

        self.__loaders_cache__ = {}

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
            interval: GenomicInterval,
            plot_components: Sequence[IntervalPlotComponent],
            **data_kwargs
        ):
        """
        Get the data for the specified interval(s) and plot components.

        Parameters
        ----------
        interval : GenomicInterval or dict {key: GenomicInterval}
            The genomic interval to plot.
            If a dict is provided, the component-specific interval key is used to extract the interval.

        plot_components : Sequence[VerticalPlotComponent]
            The vertical plot components to plot.

        **data_kwargs : dict
            Keyword arguments to pass to the loaders function.

        Returns
        -------
        data : Iter[DataBundle]
            A Iter of DataBundle objects containing the data for each plot component
        """
        common_kwargs = set(data_kwargs) & set(self.data_kwargs)
        if common_kwargs:
            self.logger.debug(
                f"Found {len(common_kwargs)} overlapping data kwargs: {list(common_kwargs)}"
            )
            self.logger.debug("Using values passed to get_interval_data function.")

        for component in plot_components:
            data = DataBundle(
                interval=self._parse_interval(
                    interval,
                    getattr(component, 'interval_key', None)
                )
            )
            yield component.load_data(
                data,
                **{**self.data_kwargs, **data_kwargs}
            )


DataPreprocessor = IntervalDataPreprocessor
