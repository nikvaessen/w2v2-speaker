################################################################################
#
# Provide a dataclass which automatically tries to cast any type-hinted field
# to the type hint. It also as provides an abstract method
# for constructing the object it configures.
#
# Author(s): Nik Vaessen
################################################################################

import dataclasses

from abc import abstractmethod
from enum import Enum
from typing import TypeVar, Generic

################################################################################
# base configuration which supports casting to type hint and provides abstract
# interface for creating an object based on the configuration

C = TypeVar("C")


@dataclasses.dataclass()
class CastingConfig(Generic[C]):
    def __post_init__(self):
        post_init_type_cast(self)


def post_init_type_cast(dataclass):
    if not dataclasses.is_dataclass(dataclass):
        raise Exception("Can only type-cast dataclass classes.")

    for field in dataclasses.fields(dataclass):
        value = getattr(dataclass, field.name)
        typehint_cls = field.type

        if value is None:
            # no value specified to type-convert
            continue

        elif isinstance(value, typehint_cls):
            # no need for type-conversion
            continue

        elif isinstance(value, dict):
            """
            if execution gets here, we know
            value is not an instance of typehinted-type but
            is a dictionary. It contains the contents
            of a nested dataclass
            """
            obj = typehint_cls(**value)

            # recursively perform type casting
            post_init_type_cast(obj)

        elif issubclass(typehint_cls, Enum):
            # enum's have a different init procedure
            obj = typehint_cls[value]

        else:
            # simply type-cast the object
            obj = typehint_cls(value)

        setattr(dataclass, field.name, obj)
