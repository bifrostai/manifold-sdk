"""Apply a `NativeLayout` entry's structural ops to one channel value.

The pack and unpack adapters share the same structural arithmetic — cast, batch
axis, slice, split, assemble — differing only in which channel an entry pulls from
(pack) or pushes into (unpack). That arithmetic lives here so the two adapters
cannot drift from each other or from `NativeLayout`'s declared semantics. It is
pure numpy on plain arrays and does not apply convention math.
"""

from __future__ import annotations

import numpy as np

from manifold.core.native_layout import (
    Assemble,
    BatchAxis,
    Component,
    DtypeCast,
    LayoutEntry,
    Slice,
    Split,
)


def _component_window(value: np.ndarray, component: Component) -> np.ndarray:
    """The `[start:stop)` window of `value` on its last axis for a split/assemble part."""
    return value[..., component.start : component.stop]


def _apply_dtype_cast(value: np.ndarray, op: DtypeCast) -> np.ndarray:
    """Cast to the target dtype, making a C-contiguous copy when requested."""
    if op.contiguous:
        return np.ascontiguousarray(value, dtype=np.dtype(op.dtype))
    return value.astype(np.dtype(op.dtype))


def _apply_batch_axis(value: np.ndarray, op: BatchAxis) -> np.ndarray:
    """Prepend a leading batch and/or time axis of size one.

    `time` adds a second leading axis after `batch`, so `batch=True, time=True`
    turns `(dim,)` into `(1, 1, dim)`. Applied outermost-last so the resulting
    order is `(batch, time, …)`.
    """
    out = value
    if op.time:
        out = out[np.newaxis, ...]
    if op.batch:
        out = out[np.newaxis, ...]
    return out


def apply_entry(value: np.ndarray, entry: LayoutEntry) -> dict[str, np.ndarray]:
    """Run an entry's ops over a single channel value, returning the keys it emits.

    Returns a dict because a `Split` op fans the value out into several keys; every
    other op leaves it the entry's single `key`. A `Split` rewrites the working set
    into its component keys, and any later op (a `DtypeCast`, a `BatchAxis`) applies
    to every split component uniformly — matching the RLDX packings, which cast and
    batch each per-axis key the same way. The ops apply left to right in `entry.ops`.
    """
    working: dict[str, np.ndarray] = {entry.key: value}
    for op in entry.ops:
        if isinstance(op, Slice):
            working = {key: arr[..., op.start : op.stop] for key, arr in working.items()}
        elif isinstance(op, DtypeCast):
            working = {key: _apply_dtype_cast(arr, op) for key, arr in working.items()}
        elif isinstance(op, BatchAxis):
            working = {key: _apply_batch_axis(arr, op) for key, arr in working.items()}
        elif isinstance(op, Assemble):
            # All components index the same (single) working array; concatenate
            # their windows on the feature axis in the declared order.
            (source,) = working.values()
            assembled = np.concatenate(
                [_component_window(source, component) for component in op.components], axis=-1
            )
            working = {entry.key: assembled}
        elif isinstance(op, Split):
            # Fan the (single) working array out into the component keys. The group
            # key is dropped; the component keys take its place for any later op.
            (source,) = working.values()
            working = {
                component.key: _component_window(source, component) for component in op.components
            }
    return working


__all__ = ["apply_entry"]
