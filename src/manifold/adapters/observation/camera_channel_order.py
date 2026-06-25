"""Swap the colour-channel byte order of named cameras on the observation side.

A policy trained on RGB frames cannot be served BGR ones (or vice versa): the
frame looks plausible — same shape, same dtype — but the red and blue planes are
transposed, the silent-and-plausible mismatch the channel-order convention
exists to catch (ADR-0001, decision 5). This adapter reverses the trailing
channel axis of each named camera array and — because channel order is a declared
convention — also advances the spec: it sets the declared `channel_order` of each
named camera to the target. So a static `check_compatibility` can prove the swap
is what bridges a policy that consumes RGB against a benchmark that publishes BGR;
the swap is no longer a spec-invisible side effect. It is lossless — a
representation transform that reorders bytes.

Everything else about the observation — the proprioception, the instruction, and
any cameras not named — passes through.

Parameterized by the target `ChannelOrder` and the camera names to swap:
`SwapChannelOrder(target=ChannelOrder.RGB, cameras=("agentview", "wrist"))`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from manifold.core.adapter import ObservationAdapter
from manifold.core.observation_space import ObservationSpace
from manifold.core.sensor import ChannelOrder
from manifold.core.values import Observation


class SwapChannelOrder(ObservationAdapter):
    """Reverse the channel axis of the named cameras and set their declared channel order."""

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True

    def __init__(self, target: ChannelOrder, cameras: Sequence[str]) -> None:
        self.target = target
        self.cameras = tuple(cameras)

    def applies(self, source: BaseModel) -> bool:
        """True when any named camera's channel order differs from the target."""
        if not isinstance(source, ObservationSpace):
            return False
        return any(
            (camera := source.camera(name)) is not None and camera.channel_order is not self.target
            for name in self.cameras
        )

    def produce(self, source: BaseModel) -> ObservationSpace:
        """The same spec with each named camera's channel order set to the target."""
        if not isinstance(source, ObservationSpace):
            raise TypeError("SwapChannelOrder transforms an ObservationSpace source only")
        out = source
        for name in self.cameras:
            camera = source.camera(name)
            if camera is not None:
                out = out.with_camera(camera.model_copy(update={"channel_order": self.target}))
        return out

    def adapt(self, observation: Any, *, source: BaseModel) -> Observation:
        """Reverse the trailing channel axis of each named camera; the rest passes through.

        A camera absent from `observation.sensors` is skipped, so the adapter is safe
        on a superset of the cameras a given observation carries. The reversal is made
        C-contiguous, preserving dtype.
        """
        sensors = dict(observation.sensors)
        for name in self.cameras:
            frame = sensors.get(name)
            if frame is None:
                continue
            arr = np.asarray(frame)
            sensors[name] = np.ascontiguousarray(arr[..., ::-1])
        return Observation(
            state=observation.state, sensors=sensors, instruction=observation.instruction
        )


__all__ = ["SwapChannelOrder"]
