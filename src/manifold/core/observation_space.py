"""The composite observation spec: the channel aggregate at the seam.

An observation is not flat. It is an aggregate of typed **channels** —
proprioception, zero or more cameras, and a language instruction — each carrying
its own conventions. The benchmark publishes one of these (what it sees); the
policy declares one of these (the form it consumes); a check folds an observation
chain from the former toward the latter, and `first_unmet` decides whether the
result supplies everything the policy needs, channel by channel.

This widens the observation side from bare `Proprioception` to the channel
aggregate, so a camera flip stops being a spec-invisible side effect and becomes
a real spec-changing transform — the precondition for any camera-convention
checking (ADR-0001, decision 5).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from manifold.core.embodiment import Proprioception
from manifold.core.sensor import Camera


def _calibration_matches(wanted: Camera, offered: Camera) -> bool:
    """Whether the offered camera supplies the calibration the wanted one consumes.

    Asymmetric, like the channel comparison around it: a policy that does not
    declare a calibration is met by a benchmark that publishes one, and never the
    reverse. When both declare it, a field of view is not bridgeable by a resize
    and an axis or frame mismatch is the silent kind, so both have to agree
    (ADR 0008).

    The two halves are compared differently. `axes` and `frame` are
    closed sets the SDK defines, so equality is the whole question. The four
    intrinsics are derived floats — a benchmark reaches them through `from_fov`, an
    adapter through `scaled` — so two correct descriptions of one camera differ in
    their last bits, and comparing those exactly would refuse a pairing over an error
    smaller than the arithmetic that produced it. `is_close_to` carries the tolerance.
    """
    if wanted.calibration is None:
        return True
    if offered.calibration is None:
        return False
    return (
        wanted.calibration.axes == offered.calibration.axes
        and wanted.calibration.frame == offered.calibration.frame
        and wanted.calibration.intrinsics.is_close_to(offered.calibration.intrinsics)
    )


def _camera_conventions_match(wanted: Camera, offered: Camera) -> bool:
    """Whether the consumed convention axes agree (mount excluded — it is provenance)."""
    return (
        wanted.orientation == offered.orientation
        and wanted.channel_order == offered.channel_order
        and wanted.shape == offered.shape
        and wanted.dtype == offered.dtype
        and wanted.modality == offered.modality
        and _calibration_matches(wanted, offered)
    )


class ObservationSpace(BaseModel):
    """The aggregate of channels at the observation seam.

    `proprioception` is the robot-state channel; `cameras` are the exteroceptive
    channels, name-keyed (order is not significant); `instruction` records whether
    a language instruction is present. A benchmark derives one of these from its
    embodiment and sensors (`Benchmark.observation`); a policy declares one as the
    form it consumes (`PolicySignature.observation_space`).
    """

    model_config = ConfigDict(frozen=True)

    proprioception: Proprioception = Field(default_factory=Proprioception)
    cameras: tuple[Camera, ...] = ()
    instruction: bool = False

    def camera(self, name: str) -> Camera | None:
        """The camera channel named `name`, or None if this spec has none."""
        for cam in self.cameras:
            if cam.name == name:
                return cam
        return None

    def with_proprioception(self, proprioception: Proprioception) -> ObservationSpace:
        """A copy with the proprioception channel replaced."""
        return self.model_copy(update={"proprioception": proprioception})

    def with_camera(self, camera: Camera) -> ObservationSpace:
        """A copy with the same-named camera replaced, preserving order.

        If no camera of that name is present the cameras are unchanged — an
        adapter rewriting a channel it does not carry is a no-op, not an append.
        """
        cameras = tuple(camera if c.name == camera.name else c for c in self.cameras)
        return self.model_copy(update={"cameras": cameras})

    def first_unmet(self, available: ObservationSpace) -> str | None:
        """The first channel this (the policy form) needs that `available` lacks.

        `self` is what the policy consumes; `available` is what the benchmark
        publishes (possibly after an observation chain). Returns a reason string
        for the first unmet channel, or None when every consumed channel is met.
        This is the single home for all observation-side matching.

        Camera channels are compared on the axes a policy actually *consumes* —
        orientation, channel_order, shape, dtype, modality — and not
        on `mount`, which is provenance (a deferred tag): comparing it would
        manufacture a spurious mismatch between a wrist and a scene view that are
        byte-identical at the seam. A policy declaring no cameras matches any
        benchmark, since only `self.cameras` is iterated.
        """
        proprio_unmet = self.proprioception.first_unmet(available.proprioception)
        if proprio_unmet is not None:
            return f"proprioception.{proprio_unmet}"
        for wanted in self.cameras:
            offered = available.camera(wanted.name)
            if offered is None:
                return f"camera {wanted.name!r}: not published"
            if not _camera_conventions_match(wanted, offered):
                return f"camera {wanted.name!r}: conventions differ"
        if self.instruction and not available.instruction:
            return "instruction: required but not published"
        return None


__all__ = ["ObservationSpace"]
