"""Apply an action adapter to one group of a whole-body action.

The action adapters are written for one arm: a rotation re-encode or a gripper
polarity remap reads an `EEActionSpace`. A `CompositeActionSpace` holds such an arm
as one group among others, so those adapters never apply to it directly.
`GroupAdapter(index, inner)` runs `inner` on group `index`, on that group's spec and
on its slice of each action, and leaves the other groups and the mode slot as they
are. `inner` keeps the group's kind but may change its width, as a quaternion arm
re-encoded to axis-angle does.
"""

from __future__ import annotations

from typing import Any, ClassVar, get_args

from pydantic import BaseModel

from manifold.core.adapter import ActionAdapter
from manifold.core.embodiment import ActionGroup, CompositeActionSpace

_GROUP_KINDS = get_args(get_args(ActionGroup)[0])


class GroupAdapter(ActionAdapter):
    """Run a stateless `inner` adapter on group `index` of a composite action."""

    from_spec: ClassVar[type[BaseModel]] = CompositeActionSpace
    to_spec: ClassVar[type[BaseModel]] = CompositeActionSpace
    # Each instance takes `lossless` from the adapter that it wraps. The class value
    # is False, so code that reads the class never treats a GroupAdapter as lossless.
    lossless: ClassVar[bool] = False

    def __init__(self, index: int, inner: ActionAdapter) -> None:
        if index < 0:
            raise ValueError(f"GroupAdapter needs a group index of 0 or more, not {index}")
        if inner.state_key is not None:
            raise ValueError("GroupAdapter lifts stateless adapters only")
        if inner.to_spec is not inner.from_spec or not issubclass(inner.from_spec, _GROUP_KINDS):
            raise ValueError(
                f"{type(inner).__name__} maps {inner.from_spec.__name__} to "
                f"{inner.to_spec.__name__}; a group adapter needs one that keeps an arm, "
                "torso or base spec of its kind"
            )
        self.index = index
        self.inner = inner
        self.lossless = inner.lossless  # ty: ignore[invalid-attribute-access]

    def _group(self, source: BaseModel) -> BaseModel | None:
        if not isinstance(source, CompositeActionSpace) or self.index >= len(source.groups):
            return None
        return source.groups[self.index]

    def applies(self, source: BaseModel) -> bool:
        """True when group `index` exists and `inner` applies to its spec."""
        group = self._group(source)
        return (
            group is not None and type(group) is self.inner.from_spec and self.inner.applies(group)
        )

    def produce(self, source: BaseModel) -> CompositeActionSpace:
        """The composite with group `index` replaced by what `inner` produces."""
        if not isinstance(source, CompositeActionSpace):
            raise TypeError("GroupAdapter transforms CompositeActionSpace only")
        groups: list[BaseModel] = list(source.groups)
        groups[self.index] = self.inner.produce(groups[self.index])
        return source.model_copy(update={"groups": tuple(groups)})

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        """Run `inner` on group `index`'s slice of the action; keep the rest."""
        if not isinstance(source, CompositeActionSpace):
            raise TypeError("GroupAdapter needs a CompositeActionSpace source")
        out = [float(v) for v in values]
        source.validate_value(out)
        _, start, stop = source.group_slices()[self.index]
        inner = self.inner.adapt(out[start:stop], source=source.groups[self.index])
        return [*out[:start], *(float(v) for v in inner), *out[stop:]]

    def __repr__(self) -> str:
        return f"GroupAdapter({self.index}, {type(self.inner).__name__})"


__all__ = ["GroupAdapter"]
