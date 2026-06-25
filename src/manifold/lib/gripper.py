"""Convert a gripper value between encodings.

Every gripper encoding expresses the same one thing: how open the gripper is.
The two helpers here pass through that common ground. `openness` maps a value in
some encoding to a fraction in [0, 1] where 1 is fully open; `from_openness` maps
that fraction back to another encoding. `remap` is the round trip, the conversion
an adapter applies to one gripper value.

Encodings differ in two ways the conventions name: their numeric range, and
which end of it is open. These helpers handle both, so a policy trained with one
gripper convention can drive a benchmark that expects another.
"""

from __future__ import annotations

from manifold.core.conventions import GripperFormat

# For each encoding: (closed value, open value). Openness is the position of a
# value on the line from closed to open.
_CLOSED_OPEN: dict[GripperFormat, tuple[float, float]] = {
    GripperFormat.SIGNED: (-1.0, 1.0),
    GripperFormat.SIGNED_OPEN_LOW: (1.0, -1.0),
    GripperFormat.UNSIGNED: (0.0, 1.0),
    GripperFormat.UNSIGNED_OPEN_LOW: (1.0, 0.0),
    GripperFormat.BINARY: (0.0, 1.0),
    GripperFormat.BINARY_OPEN_LOW: (1.0, 0.0),
}

_DISCRETE = {GripperFormat.BINARY, GripperFormat.BINARY_OPEN_LOW}

# Every gripper encoding must appear in the table above, so an unhandled format
# fails at import rather than with a KeyError mid-conversion. A raise (not assert)
# keeps the guard under `python -O`.
if _CLOSED_OPEN.keys() != set(GripperFormat):
    raise RuntimeError("_CLOSED_OPEN is missing a GripperFormat member")


def openness(value: float, fmt: GripperFormat) -> float:
    """Fraction in [0, 1] of how open `value` is under `fmt`, where 1 is open."""
    closed, opened = _CLOSED_OPEN[fmt]
    return (value - closed) / (opened - closed)


def from_openness(fraction: float, fmt: GripperFormat) -> float:
    """The value under `fmt` for a given openness fraction.

    The fraction is clamped to [0, 1] first, so a value outside an encoding's
    range never produces one outside the target's. A discrete encoding then
    rounds to its closed or open value; a continuous one interpolates.
    """
    fraction = min(1.0, max(0.0, fraction))
    closed, opened = _CLOSED_OPEN[fmt]
    if fmt in _DISCRETE:
        fraction = round(fraction)
    return closed + fraction * (opened - closed)


def remap(value: float, source: GripperFormat, target: GripperFormat) -> float:
    """Convert one gripper value from the `source` encoding to the `target`."""
    return from_openness(openness(value, source), target)


__all__ = ["from_openness", "openness", "remap"]
