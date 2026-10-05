"""Numeric threshold predicates shared by signal and decision models."""

import math

from pydantic import BaseModel, ConfigDict, model_validator


class NumericPredicate(BaseModel):
    """Numeric threshold predicate for structure signals."""

    model_config = ConfigDict(extra="forbid")

    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None

    @model_validator(mode="after")
    def validate_contract(self):
        for value in (self.gt, self.gte, self.lt, self.lte):
            if value is not None and not math.isfinite(value):
                raise ValueError("numeric predicate values must be finite")
        if all(value is None for value in (self.gt, self.gte, self.lt, self.lte)):
            raise ValueError("numeric predicate requires at least one comparator")
        if self.gt is not None and self.gte is not None:
            raise ValueError("numeric predicate cannot set both gt and gte")
        if self.lt is not None and self.lte is not None:
            raise ValueError("numeric predicate cannot set both lt and lte")
        lower = self.gt if self.gt is not None else self.gte
        upper = self.lt if self.lt is not None else self.lte
        if lower is not None and upper is not None:
            strict = self.gt is not None or self.lt is not None
            if lower > upper or (lower == upper and strict):
                raise ValueError("numeric predicate defines an empty range")
        return self
