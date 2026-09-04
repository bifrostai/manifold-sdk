"""Stack the last N camera frames into a clip — a stateful observation adapter.

Some checkpoints consume a short video clip per camera rather than a single
frame: RLDX's video modality, for instance, stacks 4 frames spaced 2 env-steps
apart. A policy server could keep that buffer inside its inference loop; this
adapter puts it in the pipeline instead, where it is a real, checkable,
spec-changing transform — the escape hatch the ADR blesses ("write an adapter").

It is stateful: the per-camera frame history is per-episode runtime state, not
config, so it lives in a `PipelineState` slice threaded into `apply_observation`
rather than on the frozen pipeline (ADR-0001, "pipelines may carry episodic
state"). Its `state_key` labels that slice; `reset_lane` at an episode boundary
drops it so the next first frame repeat-pads afresh.

On the spec it lengthens each named camera from a single frame `(H, W, C)` to a
clip `(n_frames, H, W, C)` — a new leading time axis. The inference batch dim
(`(1, n_frames, H, W, C)`) stays OUT of the spec: that is packaging the policy's
`infer` adds, not a seam convention. `stride` affects which buffered frames are
sampled, not the shape, so two instances differing only in stride produce
identical specs — correct, since only the target shape is declared.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from manifold.core.adapter import ObservationAdapter
from manifold.core.observation_space import ObservationSpace
from manifold.core.values import Observation


class StackFrameHistory(ObservationAdapter):
    """Stack each named camera's recent strided frame history into a clip.

    Lossless on the seam: it discards no environment data, it re-represents a
    stream of frames as a clip. Parameterized by the camera names to stack, the
    number of frames `n_frames`, and the env-step `stride` between sampled frames.
    """

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True
    state_key: ClassVar[str | None] = "frame_history"

    def __init__(self, cameras: Sequence[str], *, n_frames: int = 4, stride: int = 2) -> None:
        self.cameras = tuple(cameras)
        self.n_frames = n_frames
        self.stride = stride
        # Buffer the frames spanned by the strided sample window: with n_frames=4
        # and stride=2 the window is 7 frames deep, sampled at offsets (-7, -5, -3,
        # -1) from the end (newest last) — matching RLDX's delta_indices.
        self._buffer_depth = (n_frames - 1) * stride + 1
        self._stack_indices = tuple(-(1 + i * stride) for i in reversed(range(n_frames)))

    def applies(self, source: BaseModel) -> bool:
        """True while a named camera is still a single frame (rank-3, `(H, W, C)`).

        `produce` lengthens each named camera to rank-4, after which `applies` is
        False. (The adapter is excluded from resolution pools regardless — it is
        author-inserted, not resolver-discovered.)
        """
        if not isinstance(source, ObservationSpace):
            return False
        return any(
            (camera := source.camera(name)) is not None and len(camera.shape) == 3
            for name in self.cameras
        )

    def produce(self, source: BaseModel) -> ObservationSpace:
        """The same spec with each named single-frame camera lengthened to a clip.

        Rewrites `(H, W, C)` to `(n_frames, H, W, C)` — a new leading time axis. A
        camera already rank-4 (or absent) is left untouched.
        """
        if not isinstance(source, ObservationSpace):
            raise TypeError("StackFrameHistory transforms an ObservationSpace source only")
        out = source
        for name in self.cameras:
            camera = source.camera(name)
            if camera is not None and len(camera.shape) == 3:
                out = out.with_camera(
                    camera.model_copy(update={"shape": (self.n_frames, *camera.shape)})
                )
        return out

    def adapt(
        self, observation: Any, /, *, source: BaseModel, state: dict[str, Any] | None = None
    ) -> Observation:
        """Append each named camera's current frame and emit the strided clip.

        The per-camera history lives in `state` (the adapter's per-episode slice):
        a `deque(maxlen=buffer_depth)` keyed by camera name. The first frame of an
        episode repeat-pads the buffer (no motion history yet); thereafter each
        step appends one frame and evicts the oldest. The clip is `np.stack` of the
        strided offsets along a new leading axis. Cameras absent from the
        observation are skipped; state and instruction pass through.
        """
        if state is None:  # Pipeline._adapt always supplies it for a stateful adapter.
            raise ValueError("StackFrameHistory requires a state bag (a PipelineState slice)")
        sensors = dict(observation.sensors)
        for name in self.cameras:
            frame = sensors.get(name)
            if frame is None:
                continue
            # Copy: the env may reuse/mutate a preallocated obs buffer between steps,
            # which would retroactively corrupt the frames already buffered as history.
            frame = np.array(frame)
            buffer = state.get(name)
            if buffer is None:
                buffer = state[name] = collections.deque(maxlen=self._buffer_depth)
            if buffer:
                buffer.append(frame)
            else:
                # First frame of the episode: repeat-pad the history (no motion yet).
                for _ in range(self._buffer_depth):
                    buffer.append(frame)
            sensors[name] = np.stack([buffer[i] for i in self._stack_indices], axis=0)
        return replace(observation, sensors=sensors)


__all__ = ["StackFrameHistory"]
