import logging

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
    def __init__(self, name=None, loader_kws=None, **kwargs):
        super().__init__(logger_level=kwargs.pop('logger_level', None))

        if name is None:
            name = self.__class__.__name__
        self.name = name

        if loader_kws is None:
            loader_kws = {}
        self.loader_kws = loader_kws

        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])


    def plot(self, data, ax, **kwargs):
        """
        Abstract plot method to be implemented by specific plot components.
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
    
    def load(self, data: DataBundle, loader_kws=None, **kwargs):
        """
        Validate the preprocessor fields and call the load method.
        """
        self._validate()
        if loader_kws is None:
            loader_kws = {}
        loader_kws.update(kwargs)
        return self._load(data, **loader_kws)
    
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
