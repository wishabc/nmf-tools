import inspect

from matplotlib import pyplot as plt
from typing import List, Type
import types
from nmf_tools.plotting.modular_plot.shared import LoggerMixin, DataBundle
from makefun import wraps, add_signature_parameters, remove_signature_parameters, with_signature
import warnings


class PlotDataLoader(LoggerMixin):
    """
    Base class for all data loaders.
    Each loader should also specify the required_fields,
    which are the fields that the preprocessor should have to load the data.
    Each loader should implement the `_load` method to load and filter data based on the interval.
    By default (if not implemented), the _load method is created from the required_loader_kwargs.
    """
    required_loader_kwargs = []

    def __init__(self, logger_level=None):
        LoggerMixin.__init__(self, logger_level=logger_level)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._load is PlotDataLoader._load:
            cls._set_default_load()
        elif len(cls.required_loader_kwargs) > 0:
            warnings.warn(
                f"Both required_loader_kwargs and _load are specified for loader {cls.__name__}. required_loader_kwargs are ignored."
            )

    @classmethod
    def _set_default_load(cls):
        """
        Set the default load method if not already set.
        """
        if len(cls.required_loader_kwargs) == 0:
            warnings.warn(
                f"Loader {cls.__name__} has no required_loader_kwargs and no _load method implemented. The loader does not modify the data."
            )

        args_string = ', '.join(cls.required_loader_kwargs)
        @with_signature(f"_load(self, data, {args_string})")
        def _default_load(self, data, **kwargs):
            """
            Default load method when _load is not implemented.

            Sets the fields of the data object based on the required_loader_kwargs.
            """
            for field, value in kwargs.items():
                setattr(data, field, value)
            return data

        cls._load = types.MethodType(_default_load, cls)

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
        This class should load the data
        """
        raise NotImplementedError
    
    def load(self, data: DataBundle, **loader_kwargs):
        """
        Validate the preprocessor fields and call the load method.
        """
        self._validate(**loader_kwargs)
        loaded_data = self._load(data, **loader_kwargs)
        if loaded_data is None:
            loaded_data = data
        return loaded_data

    def _validate(self, **loader_kwargs):
        """
        Validate that the preprocessor has all the required fields.
        """
        missing_args = [
            arg for arg in loader_kwargs 
            if isinstance(loader_kwargs[arg], RequiredArgument)
        ]
        if missing_args:
            raise ValueError(f"Loader {self.__class__.__name__} is missing required argument(s): {', '.join(missing_args)}")


class DataLoader(PlotDataLoader):

    def __init__(self, logger_level=None):
        print("DataLoader is deprecated and will be soon removed. Please use PlotDataLoader instead.")
        super().__init__(logger_level=logger_level)


class PlotComponent(LoggerMixin):
    """
    Base class for all plot components.
    Each plot component should implement the `plot` method,
    which takes the data and an axis object and plots the data on it.
    """
    __loader_kwargs_signature__ = {} # contains all loader kwargs for updated signature
    __required_loaders__: List[Type[PlotDataLoader]] = []
    __original_init__ = None

    def __init__(self, name=None, logger_level=None, **kwargs):
        LoggerMixin.__init__(self, logger_level=logger_level)

        if name is None:
            name = self.__class__.__name__
        self.name = name

        # Separate loader kwargs from plot kwargs
        self.loader_kwargs = {
            key: kwargs.pop(key, v)
            for key, v in self.__class__.__loader_kwargs_signature__.items()
        }

        self.plot_kwargs = kwargs

    @classmethod
    def with_loaders(cls, *loaders, new_class_name=None):
        """
        Create a new class that inherits from the current class
        but requires the specified loaders.
        """
        if new_class_name is None:
            new_class_name = cls.__name__
        new_class = type(
            new_class_name,
            (cls,),
            {}
        )
        return uses_loaders(*loaders)(new_class)
    
    def load_data(self, data: DataBundle, **loader_kwargs):
        """
        Modifies data for the plot component using the required loaders load function.
        """
        for LoaderClass in self.__required_loaders__:
            all_loader_kwargs = {**self.loader_kwargs, **loader_kwargs}
            all_loader_kwargs = {
                k: v for k, v in all_loader_kwargs.items()
                if k in LoaderClass.get_fullargspec()
            }

            loader = LoaderClass(
                logger_level=self.logger.level
            )
            data = loader.load(data, **all_loader_kwargs)
            data.processed_loaders.append(LoaderClass)
        return data

    def plot(self, data, ax, **plot_kwargs):
        """
        Wrapper for the plot method to pass the plot_kws to the plot method.
        Also supports axes set methods e.g. xlim -> ax.set_xlim
        """
        kws = {**self.plot_kwargs, **plot_kwargs}

        set_methods = [method[len('set_'):] for method in dir(plt.Axes) if method.startswith('set_')]
        axes_kws = kws.pop('axes_kwargs', {})
        axes_kws = {key: axes_kws[key] for key in axes_kws if key in set_methods}

        axes = self._plot(data, ax, **kws)
        if axes is None:
            axes = ax
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


class RequiredArgument:
    def __repr__(self):
        return 'Required loader arg'


def uses_loaders(*loaders):
    """
    Class decorator to specify which data loaders a plot component requires.
    Updates the __required_loaders__ and __default_loader_kwargs__ attributes of the class.
    Also updates the __init__ method to include the loader arguments in the signature.
    If uses_loaders has been called multiple times, the loader arguments and the __init__ signature
    are overwritten.
    """
    def decorator(cls: Type[PlotComponent]):
        cls.__required_loaders__ = loaders
        loader_kwargs = _collect_all_kwargs(*loaders)
        cls.__loader_kwargs_signature__ = {**cls.__loader_kwargs_signature__, **loader_kwargs}
        if cls.__original_init__ is None:
            cls.__original_init__ = cls.__init__
        cls.__init__ = _update_signature(
            cls.__original_init__,
            cls.__loader_kwargs_signature__
        )
        return cls
    return decorator


def _collect_all_kwargs(*loaders):
    loader_kwargs = {}
    for loader in loaders[::-1]:
        if not issubclass(loader, PlotDataLoader):
            raise ValueError(f"Loader {loader} is not a subclass of DataLoader.")
        loader_kwargs.update(loader.get_fullargspec())
    return loader_kwargs


def _update_signature(original_init, loader_kwargs: dict):
    """
    A decorator to dynamically update the init signature of a class
    by gathering parameters from all loaders.
    """
    original_signature = inspect.signature(original_init)
    signature = remove_signature_parameters(
        original_signature,
        'kwargs'
    )
    params = [
        inspect.Parameter(arg, inspect.Parameter.KEYWORD_ONLY, default=value)
        for arg, value in loader_kwargs.items()
    ]
    signature = add_signature_parameters(
        signature,
        last=params
    )

    signature = add_signature_parameters(
        signature,
        last=inspect.Parameter('plotting_kwargs', inspect.Parameter.VAR_KEYWORD)
    )
    @wraps(original_init, new_sig=signature)
    def wrapped_init(self, *args, **kwargs):
        return original_init(self, *args, **kwargs)
    
    return wrapped_init
