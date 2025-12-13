from __future__ import annotations

from dataclasses import is_dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    if is_dataclass(value):
        return to_jsonable(value.__dict__)

    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(by_alias=False))
        except TypeError:
            return to_jsonable(value.model_dump())

    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return to_jsonable(value.to_dict())

    return str(value)
