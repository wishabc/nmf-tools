import logging

from matplotlib import pyplot as plt

class LoggerMixin:
    def __init__(self, logger_level=None):
        if logger_level is None:
            logger_level = logging.INFO
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            self.logger.setLevel(logger_level)


class DataBundle(LoggerMixin):
    """
    A container for holding all data needed by plot components.
    Has one required field, interval, which indicates
    the genomic interval for which the data is loaded.

    Other attributes are added dynamically by the loaders
    depending on the data needed by the plot components.
    """
    def __init__(self, interval, **kwargs):
        object.__setattr__(self, '_initialized_attributes', set())
        self.interval = interval
        # Add additional data fields here (not strictly necessary)

        self.processed_loaders = []
        super().__init__(**kwargs)

    
    def __setattr__(self, name, value):
        if name != 'logger' and name in self._initialized_attributes:
            self.logger.warning(f"Warning: Attribute '{name}' is overridden.")
        self._initialized_attributes.add(name)
        super().__setattr__(name, value)


class PlotComponent(LoggerMixin):
    """
    Base class for all plot components.
    Each plot component should implement the `plot` method,
    which takes the data and an axis object and plots the data on it.
    """
    def __init__(self, name=None, loader_kwargs=None, **kwargs):
        super().__init__(logger_level=kwargs.pop('logger_level', None))

        if name is None:
            name = self.__class__.__name__
        self.name = name

        if loader_kwargs is None:
            loader_kwargs = {}
        self.loader_kwargs = loader_kwargs

        self.plot_kws = kwargs

    def plot(self, data, ax, **kwargs):
        """
        Wrapper for the plot method to pass the plot_kws to the plot method.
        Also supports axes set methods e.g. xlim -> ax.set_xlim
        """
        kws = self.plot_kws.copy()
        kws.update(kwargs)

        set_methods = [method.strip('set_') for method in dir(plt.Axes) if method.startswith('set_')]
        axes_kws = kws.pop('axes_kwargs', {})
        axes_kws = {key: axes_kws[key] for key in axes_kws if key in set_methods}

        axes = self._plot(data, ax, **kws)
        for key in axes_kws:
            getattr(ax, 'set_' + key)(axes_kws[key])

        return axes

    def _plot(self, data, ax, **kwargs):
        """
        Abstract plot method to be implemented by specific plot components.

        Should not include any axes set methods as arguments,
        as they will be intercepted by the plot method.
        E.g. xlim, ylim, xlabel, etc.
        """
        raise NotImplementedError("Plot method should be implemented in subclasses.")
    

class DataLoader(LoggerMixin):
    """
    Base class for all data loaders.
    Each loader should also specify the required_fields,
    which are the fields that the preprocessor should have to load the data.
    Each loader should implement the `_load` method to load and filter data based on the interval.
    By default (if not implemented), the _load method sets the required_fields from the preprocessor.
    """
    __required_fields__ = []

    def __init__(self, preprocessor, interval, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = preprocessor
        self.interval = interval

    def _load(self, data: DataBundle, **kwargs):
        """
        Load and filter data based on the interval.
        Default implementation of the load method that sets the required fields from the preprocessor.
        """
        for field in self.__required_fields__:
            value = getattr(self.preprocessor, field)
            setattr(data, field, value)
        return data
    
    def load(self, data: DataBundle, loader_kwargs=None, **kwargs):
        """
        Validate the preprocessor fields and call the load method.
        """
        self._validate()
        if loader_kwargs is None:
            loader_kwargs = {}
        loader_kwargs.update(kwargs)
        return self._load(data, **loader_kwargs)
    
    def _validate(self):
        """
        Validate that the preprocessor has all the required fields.
        """
        missing_fields = [field for field in self.__required_fields__ if not hasattr(self.preprocessor, field)]
        if missing_fields:
            raise ValueError(f"Preprocessor is missing required field(s): {', '.join(missing_fields)} for loader: {self.__class__.__name__}")


def uses_loaders(*loaders):

    """
    Class decorator to specify which data loaders a plot component requires.
    """
    # Impelemted as a decorator factory to allow for additional processing if needed.
    # This allows passing class objects directly to the decorator.
    def decorator(cls):
        cls.required_loaders = loaders
        return cls
    return decorator
