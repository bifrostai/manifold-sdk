"""The native-packing phase: cross the seam to a checkpoint's tensor dict and back.

Convention adapters (`observation/`, `action/`) keep a value a valid SDK
`Observation`/`Action` — they are phase one of the pipeline. The packing adapters
here are phase two: they cross the seam to the model's native tensor dict on the
observation side, and read the model's raw output chunk back into an SDK `Action`
on the action side, applying a declared `NativeLayout` (ADR-0001, "The pipeline
reaches the model: convention bridging, then native packing").

There are two, one per direction:

- `PackToNativeLayout` — observation-side. `from_spec` is `ObservationSpace`,
  `to_spec` is `NativeLayout`; `adapt(Observation) -> dict[str, np.ndarray]` applies
  the layout's structural ops (rename, dtype cast, batch/time axis, slice, split,
  assemble) to the channel values, producing the flat tensor dict the checkpoint
  consumes.
- `UnpackFromNativeLayout` — action-side. It reads a raw model-output chunk at a
  given step through a declared output layout into an SDK `Action`; `to_spec` is the
  policy's action space, so the action chain continues from there into the
  convention adapters that follow.

These do not apply convention math — that is the codomain rule (ADR-0001). They are
author-inserted, never resolved: a resolver walks the channel-spec graph and packing
does not advance it. They live here, not in `observation/`/`action/`, because they
are the one phase whose codomain is the model's wire dict, not an SDK channel form.
"""

from __future__ import annotations

from manifold.adapters.packing.pack import PackToNativeLayout
from manifold.adapters.packing.unpack import UnpackFromNativeLayout
from manifold.adapters.packing.unpack_obs import UnpackToObservation

__all__ = ["PackToNativeLayout", "UnpackFromNativeLayout", "UnpackToObservation"]
