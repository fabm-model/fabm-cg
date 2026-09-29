from typing import (
    TypeVar,
    overload,
    Optional,
    NoReturn,
    Generic,
    Callable,
    Any,
    Type,
    Mapping,
    Union,
    Iterable,
)

T = TypeVar("T")
VT = TypeVar("VT")

import numpy as np


class Base(Generic[T, VT]):
    def __init__(
        self,
        long_name: str,
        units: str,
        cast: Callable[[VT], T],
        default: Optional[VT] = None,
    ) -> None:
        self.__doc__ = f"{long_name} ({units}) [{self.__class__.__name__}]"
        self.long_name = long_name
        self.units = units
        self.cast = cast
        self.default = default

    def __set_name__(self, owner: Type["Model"], name: str) -> None:
        self.name = name
        self._name = "_" + name
        owner._vars.append(self)

    @overload
    def __get__(self, instance: None, owner: Type["Model"]) -> "Base[T, VT]": ...
    @overload
    def __get__(self, instance: "Model", owner: Type["Model"]) -> T: ...
    def __get__(
        self, instance: Union["Model", None], owner: Type["Model"]
    ) -> Union[T, "Base[T, VT]"]:
        if instance is None:
            return self
        return getattr(instance, self._name)

    def __set__(self, instance: "Model", value: Any) -> None:
        setattr(instance, self._name, self.transform(value))

    def transform(self, value: VT) -> T:
        return self.cast(value)


class Parameter(Base[T, T]):
    def __init__(
        self, long_name: str, units: str, type: Type[T], default: Optional[T] = None
    ):
        self.type = type
        super().__init__(long_name, units, default=default, cast=type)


class BoolParameter(Parameter[bool]):
    def __init__(self, long_name: str, default: Optional[bool] = None):
        super().__init__(long_name, "", default=default, type=bool)


class FloatParameter(Parameter[float]):
    def __init__(
        self,
        long_name: str,
        units: str,
        *,
        default: Optional[float] = None,
        scale_factor: float = 1.0,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ):
        self.scale_factor = scale_factor
        self.minimum = minimum
        self.maximum = maximum
        super().__init__(long_name, units, default=default, type=float)

    def transform(self, value: float) -> float:
        return super().transform(value) * self.scale_factor


class BaseDependency(Base[float, float]):
    id_type: str
    get_macro: str

    def __init__(self, long_name: str, units: str) -> None:
        super().__init__(long_name, units, cast=float)


class InteriorDependency(BaseDependency):
    id_type = "type_dependency_id"
    get_macro = "_GET_"


class BottomDependency(BaseDependency):
    id_type = "type_bottom_dependency_id"
    get_macro = "_GET_BOTTOM_"


class SurfaceDependency(BaseDependency):
    id_type = "type_surface_dependency_id"
    get_macro = "_GET_SURFACE_"


class SourceValue:
    def __init__(self) -> None:
        self._value: float = 0.0

    def __iadd__(self, value: float) -> "SourceValue":
        self._value += value
        return self

    def __isub__(self, value: float) -> "SourceValue":
        self._value -= value
        return self


class Source:
    def __set_name__(self, owner: Type["State"], name: str) -> None:
        self._name = "_" + name

    @overload
    def __get__(self, instance: None, owner: Type["State"]) -> "Source": ...
    @overload
    def __get__(self, instance: "State", owner: Type["State"]) -> SourceValue: ...
    def __get__(
        self, instance: Optional["State"], owner: Type["State"]
    ) -> Union["Source", SourceValue]:
        if instance is None:
            return self
        if not hasattr(instance, self._name):
            setattr(instance, self._name, SourceValue())
        return getattr(instance, self._name)

    def __set__(self, instance: "State", value: SourceValue) -> None:
        if value is not getattr(instance, self._name, None):
            raise Exception("Cannot set source value directly, use += or -= instead")


class State(float):
    source = Source()


class InteriorState(State):
    bottom_flux = Source()
    surface_flux = Source()


class BaseStateVariable(Base[T, float]):
    def __init__(
        self,
        long_name: str,
        units: str,
        *,
        initial_value: float = 0.0,
        cast: Callable[[float], T] = State,
    ) -> None:
        super().__init__(long_name, units, cast=cast, default=initial_value)


class InteriorStateVariable(BaseStateVariable[InteriorState]):
    def __init__(self, long_name: str, units: str, initial_value: float = 0.0) -> None:
        super().__init__(
            long_name, units, cast=InteriorState, initial_value=initial_value
        )

    id_type = "type_state_variable_id"
    get_macro = "_GET_"
    add_source_macros = {
        "source": "_ADD_SOURCE_",
        "bottom_flux": "_ADD_BOTTOM_FLUX_",
        "surface_flux": "_ADD_SURFACE_FLUX_",
    }


class BottomStateVariable(BaseStateVariable[State]):
    id_type = "type_bottom_state_variable_id"
    get_macro = "_GET_BOTTOM_"
    add_source_macros = {"source": "_ADD_BOTTOM_SOURCE_"}


class SurfaceStateVariable(BaseStateVariable[State]):
    id_type = "type_surface_state_variable_id"
    get_macro = "_GET_SURFACE_"
    add_source_macros = {"source": "_ADD_SURFACE_SOURCE_"}


class BaseDiagnosticVariable(Base[float, float]):
    id_type: str
    set_macro: str

    def __init__(self, long_name: str, units: str) -> None:
        super().__init__(long_name, units, cast=float)

    @overload
    def __get__(
        self, instance: None, owner: Type["Model"]
    ) -> "BaseDiagnosticVariable": ...
    @overload
    def __get__(self, instance: "Model", owner: Type["Model"]) -> NoReturn: ...
    def __get__(
        self, instance: Optional["Model"], owner: Type["Model"]
    ) -> Union["BaseDiagnosticVariable", NoReturn]:
        if instance is None:
            return self
        raise Exception("Diagnostics cannot be read, only written")


class InteriorDiagnosticVariable(BaseDiagnosticVariable):
    id_type = "type_diagnostic_variable_id"
    set_macro = "_SET_DIAGNOSTIC_"


# Aliases for interior variables
StateVariable = InteriorStateVariable
Dependency = InteriorDependency
DiagnosticVariable = InteriorDiagnosticVariable


class Collection(Generic[T]):
    def __init__(self, type: Type[T]) -> None:
        self.type = type

    def __set_name__(self, owner: Type["Model"], name: str) -> None:
        self._name = "_" + name

    def __get__(
        self, instance: Optional["Model"], owner: Type["Model"]
    ) -> dict[str, T]:
        result = getattr(owner, self._name, None)
        if result is None:
            result = {}
            for name, value in vars(owner).items():
                if isinstance(value, self.type):
                    result[name] = value
            setattr(owner, self._name, result)
        return result


class Model:
    _vars: list["Base[Any, Any]"] = []

    def __init__(self) -> None:
        for obj in self._vars:
            default = obj.default
            if default is not None:
                default = obj.transform(default)
            setattr(self, obj._name, default)

    diagnostic_variables = Collection(BaseDiagnosticVariable)
    state_variables = Collection(BaseStateVariable)
    interior_state_variables = Collection(InteriorStateVariable)
    bottom_state_variables = Collection(BottomStateVariable)
    surface_state_variables = Collection(SurfaceStateVariable)
    dependencies = Collection(BaseDependency)
    parameters = Collection(Parameter)

    def describe(self) -> None:
        collections: dict[str, Mapping[str, Base[Any, Any]]] = {
            "interior state variables": self.interior_state_variables,
            "bottom state variables": self.bottom_state_variables,
            "surface state variables": self.surface_state_variables,
            "diagnostic variables": self.diagnostic_variables,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
        }
        for n, collection in collections.items():
            if collection:
                print("%i %s:" % (len(collection), n))
                for name, obj in collection.items():
                    value = getattr(self, "_" + name)
                    print(f"  {obj.name}: {obj.long_name} = {value} {obj.units}")

    def _collect_sources(
        self, collection: Mapping[str, Base[Any, Any]], attribute: str = "source"
    ) -> np.ndarray:
        sources = []
        for name in collection:
            state: State = getattr(self, name)
            source: SourceValue = getattr(state, attribute)
            sources.append(source._value)
        return np.array(sources)

    @property
    def state(self) -> np.ndarray:
        """Values of the model state variables."""
        return np.array([getattr(self, name) for name in self.state_variables])

    @state.setter
    def state(self, values: Iterable[float]) -> None:
        for name, value in zip(self.state_variables, values):
            setattr(self, name, value)

    @property
    def sources(self) -> np.ndarray:
        return self._collect_sources(self.state_variables)

    @property
    def bottom_fluxes(self) -> np.ndarray:
        return self._collect_sources(self.interior_state_variables, "bottom_flux")

    @property
    def surface_fluxes(self) -> np.ndarray:
        return self._collect_sources(self.interior_state_variables, "surface_flux")

    def reset_sources(self):
        for name in self.state_variables:
            state: State = getattr(self, name)
            state.source._value = 0.0
        for name in self.interior_state_variables:
            interior_state: InteriorState = getattr(self, name)
            interior_state.bottom_flux._value = 0.0
            interior_state.surface_flux._value = 0.0
