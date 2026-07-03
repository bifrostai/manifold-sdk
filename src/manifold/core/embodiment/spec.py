"""The shared base for specs describing one flat numeric value."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from manifold.core.conventions import check_length


class ValueSpec(BaseModel):
    """A spec for one flat numeric value, an action or an observation.

    Subclasses implement `expected_length`; the length-based `example` and
    `validate_value` are shared. Specs are immutable: they are declarations,
    compared by value.
    """

    model_config = ConfigDict(frozen=True)

    def expected_length(self) -> int:
        """Number of floats in one value laid out under this spec."""
        raise NotImplementedError

    def example(self) -> list[float]:
        """A zero-filled value of the right length."""
        return [0.0] * self.expected_length()

    def validate_value(self, value: Any) -> None:
        """Check `value` has the right length. This is a structural check only."""
        check_length(value, self.expected_length(), type(self).__name__)


__all__ = ["ValueSpec"]
