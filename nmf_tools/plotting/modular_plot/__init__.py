import logging
import inspect
from functools import wraps
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
    
    def __getattr__(self, name):
        loaders = ', '.join([loader.__name__ for loader in self.processed_loaders])
        initialized_attributes = ', '.join(self._initialized_attributes)
        raise AttributeError(f"Data loaded with {loaders} is missing attribute '{name}'. Available attributes: {initialized_attributes}")
    
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
    __default_loader_kwargs__ = {}
    __required_loaders__ = []

    def __init__(self, name=None, logger_level=None, **kwargs):
        LoggerMixin.__init__(self, logger_level=logger_level)

        if name is None:
            name = self.__class__.__name__
        self.name = name

        self.loader_kwargs = {
            key: kwargs.pop(key, v)
            for key, v in self.__class__.__default_loader_kwargs__.items()
        }

        self.plot_kwargs = kwargs

    def plot(self, data, ax, **plot_kwargs):
        """
        Wrapper for the plot method to pass the plot_kws to the plot method.
        Also supports axes set methods e.g. xlim -> ax.set_xlim
        """
        kws = {**self.plot_kwargs, **plot_kwargs}

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

    def __init__(self, preprocessor, interval, logger_level=None):
        LoggerMixin.__init__(self, logger_level=logger_level)
        self.preprocessor = preprocessor
        self.interval = interval

    @classmethod
    def get_fullargspec(cls):
        """
        Get the fullargspec of the load method.
        Returns a dictionary of the arguments and their default values,
        excluding the 'self' and 'data' arguments.
        """
        fullargspec = inspect.getfullargspec(cls._load)
        if fullargspec.varkw is not None or fullargspec.varargs is not None:
            raise ValueError(f"{cls.__name__} '_load' method should not have *args or **kwargs.")
        loader_args = fullargspec.args
        loader_defaults = fullargspec.defaults or []
        extended_defaults = [RequiredArgument()] * (len(loader_args) - len(loader_defaults)) + list(loader_defaults)
        signature = dict(zip(loader_args, extended_defaults))
        if 'data' not in signature:
            raise ValueError(f"{cls.__name__} '_load' method should have 'data' argument.")
        return {k: v for k, v in signature.items() if k not in ['self', 'data']}

    def _load(self, data: DataBundle):
        """
        Load and filter data based on the interval.
        Default implementation of the load method that sets the required fields from the preprocessor.
        """
        for field in self.__required_fields__:
            value = getattr(self.preprocessor, field)
            setattr(data, field, value)
        return data
    
    def load(self, data: DataBundle, **loader_kwargs):
        """
        Validate the preprocessor fields and call the load method.
        """
        self._validate(**loader_kwargs)
        return self._load(data, **loader_kwargs)
    
    def _validate(self, **loader_kwargs):
        """
        Validate that the preprocessor has all the required fields.
        """
        missing_fields = [field for field in self.__required_fields__ if not hasattr(self.preprocessor, field)]
        if missing_fields:
            raise ValueError(f"Preprocessor is missing required field(s): {', '.join(missing_fields)} for loader: {self.__class__.__name__}")
        
        missing_args = [arg for arg in loader_kwargs if isinstance(loader_kwargs[arg], RequiredArgument)]
        if missing_args:
            raise ValueError(f"Loader {self.__class__.__name__} is missing required argument(s): {', '.join(missing_args)}")


class RequiredArgument:
    def __repr__(self):
        return 'Required loader arg'


def uses_loaders(*loaders):

    """
    Class decorator to specify which data loaders a plot component requires.
    Updates the __required_loaders__ and __default_loader_kwargs__ attributes of the class.
    """
    def decorator(cls):
        cls.__required_loaders__ = loaders
        loader_kwargs = _collect_all_kwargs(*loaders)
        cls.__default_loader_kwargs__ = {**cls.__default_loader_kwargs__, **loader_kwargs}
        cls.__init__ = _update_signature(cls.__init__, loader_kwargs)
        return cls
    return decorator


def _collect_all_kwargs(*loaders):
    loader_kwargs = {}
    for loader in loaders[::-1]:
        if not issubclass(loader, DataLoader):
            raise ValueError(f"Loader {loader} is not a subclass of DataLoader.")
        loader_kwargs.update(loader.get_fullargspec())
    return loader_kwargs


def _update_signature(original_init, loader_kwargs):
    """
    A decorator to dynamically update the init signature of a class
    by gathering parameters from the entire inheritance chain.
    """
    original_signature = inspect.signature(original_init)
    original_params = list(original_signature.parameters.values())

    # Required positional/keyword arguments (no default values)
    original_required = [
        p for p in original_params
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default == inspect.Parameter.empty
    ]

    # Optional positional/keyword arguments (with default values)
    original_optional = [
        p for p in original_params
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default != inspect.Parameter.empty
    ]

    # Keyword-only arguments
    original_kwonly = [p for p in original_params if p.kind == inspect.Parameter.KEYWORD_ONLY]

    # Variadic *args
    original_var_arg = [p for p in original_params if p.kind == inspect.Parameter.VAR_POSITIONAL]

    # Variadic **kwargs
    new_var_kwarg = [inspect.Parameter('plotting_kwargs', inspect.Parameter.VAR_KEYWORD)]

    # New arguments to add
    # new_pos_or_kw_params = [
    #     inspect.Parameter(arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    #     for arg, value in loader_kwargs.items()
    #     if isinstance(value, RequiredArgument) and arg not in [p.name for p in original_params]
    # ]

    new_kw_only_params = [
        inspect.Parameter(arg, inspect.Parameter.KEYWORD_ONLY, default=value)
        for arg, value in loader_kwargs.items()
        if arg not in [p.name for p in original_params]
    ]

    # Combine parameters in the correct order
    combined_params = (
        original_required +                # Required positional/keyword arguments (no defaults)
        #new_pos_or_kw_params +             # New positional/keyword arguments (required, no defaults)
        original_optional +                # Original optional positional/keyword arguments (with defaults)
        original_var_arg +                 # *args (variadic positional)
        original_kwonly +                  # Original keyword-only arguments
        new_kw_only_params +               # New keyword-only arguments
        new_var_kwarg                      # **kwargs (variadic keyword)
    )

    # Step 5: Create a new signature with the combined parameters
    new_signature = original_signature.replace(parameters=combined_params)


    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        return original_init(self, *args, **kwargs)

    wrapped_init.__signature__ = new_signature
    return wrapped_init
