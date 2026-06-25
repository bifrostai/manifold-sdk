"""Rebase a reported end-effector pose into another frame using a runtime base pose.

The static `FrameRebaseAdapter` applies a *constant* change of basis baked into the
adapter at construction. This adapter instead reads the transform at runtime from a
named state channel on each observation — the robot's `base_pose` for the current
step — and rebases the `ee_pose` from the world frame into the base frame by the
homogeneous transform ``ee_in_base = inv(base_pose) @ ee_pose``.

This reproduces the SIMPLER/GR00T-Bridge runner's `_ee_pose` math exactly: the
runner reads the world-frame TCP pose and the world-frame robot base pose off the
unwrapped env and composes ``inv(base_pose) @ ee_pose`` before publishing. Here that
same math is moved to the seam: the runner publishes the world-frame `ee_pose` plus
the `base_pose` on a side channel, and this adapter performs the rebase, transforming
both position and rotation. The rotation is decoded to a matrix (via `lib.rotation`),
rotated by the base orientation's inverse, and re-encoded in the `ee_pose`'s current
rotation format; position is the translated, rotated offset. The gripper slot and
every other channel pass through untouched.

The base pose is read from `base_pose_channel` (default ``"base_pose"``) and may be
either a flat 16-element row-major 4x4 homogeneous matrix, or a 7-element
``[x, y, z, qx, qy, qz, qw]`` position+quaternion (scalar-last, matching the SDK's
QUATERNION convention). Because the rebase is a rigid transform, the rotation
re-encoding loses nothing, so it is declared lossless.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from manifold.core.adapter import ObservationAdapter
from manifold.core.conventions import Frame, ee_step_layout
from manifold.core.observation_space import ObservationSpace
from manifold.core.values import Observation
from manifold.lib.rotation import from_matrix, to_matrix


def _homogeneous(value: Any) -> np.ndarray:
    """Coerce a base-pose channel value to a 4x4 homogeneous matrix.

    Accepts a 16-element row-major 4x4 flat array (or a (4, 4) array), or a
    7-element ``[x, y, z, qx, qy, qz, qw]`` position+quaternion (scalar-last).
    """
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 16:
        return array.reshape(4, 4)
    if array.size == 7:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_quat(array[3:7]).as_matrix()
        matrix[:3, 3] = array[:3]
        return matrix
    raise ValueError(
        f"base pose must be a 4x4 (16) matrix or a 7-element pos+quat, got length {array.size}"
    )


class DynamicFrameRebaseAdapter(ObservationAdapter):
    """Rebase the `ee_pose` from `source_frame` to `target_frame` via a runtime base pose."""

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True

    def __init__(
        self,
        source_frame: Frame,
        target_frame: Frame,
        *,
        base_pose_channel: str = "base_pose",
    ) -> None:
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.base_pose_channel = base_pose_channel

    def applies(self, source: BaseModel) -> bool:
        """True when the reported pose is in `source_frame`."""
        return (
            isinstance(source, ObservationSpace)
            and source.proprioception.ee_pose is not None
            and source.proprioception.ee_pose.frame == self.source_frame
        )

    def produce(self, source: BaseModel) -> ObservationSpace:
        """The same spec with its `ee_pose` frame set to the target frame."""
        spec = self._spec(source)
        ee_pose = spec.proprioception.ee_pose
        if ee_pose is None:
            raise TypeError("DynamicFrameRebaseAdapter needs a proprioception with an ee_pose")
        new_proprio = spec.proprioception.model_copy(
            update={"ee_pose": ee_pose.model_copy(update={"frame": self.target_frame})}
        )
        return spec.with_proprioception(new_proprio)

    def adapt(self, observation: Any, *, source: BaseModel) -> Observation:
        """Rebase the `ee_pose` by ``inv(base_pose) @ ee_pose``; other channels pass through."""
        ee_pose = self._spec(source).proprioception.ee_pose
        if ee_pose is None:
            raise TypeError("DynamicFrameRebaseAdapter needs a proprioception with an ee_pose")
        if self.base_pose_channel not in observation.state:
            raise KeyError(
                f"DynamicFrameRebaseAdapter needs a {self.base_pose_channel!r} state channel"
            )
        base = _homogeneous(observation.state[self.base_pose_channel])

        block = [float(v) for v in observation.state["ee_pose"]]
        pos_len, rot_len, _ = ee_step_layout(ee_pose.rotation, None)
        pos = np.asarray(block[:pos_len], dtype=np.float64)
        rot_block = block[pos_len : pos_len + rot_len]
        ee = np.eye(4, dtype=np.float64)
        ee[:3, :3] = to_matrix(rot_block, ee_pose.rotation)
        ee[:3, 3] = pos

        ee_in_target = np.linalg.inv(base) @ ee
        new_pos = ee_in_target[:3, 3].tolist()
        new_rot = from_matrix(ee_in_target[:3, :3], ee_pose.rotation)
        reencoded = new_pos + new_rot + block[pos_len + rot_len :]

        # Pass through every other state channel (including the base pose channel
        # itself) unchanged; only ee_pose is rebased.
        state = {**observation.state, "ee_pose": np.asarray(reencoded, dtype=np.float32)}
        return Observation(
            state=state, sensors=observation.sensors, instruction=observation.instruction
        )

    @staticmethod
    def _spec(source: BaseModel) -> ObservationSpace:
        if not isinstance(source, ObservationSpace):
            raise TypeError("DynamicFrameRebaseAdapter transforms an ObservationSpace source only")
        return source


__all__ = ["DynamicFrameRebaseAdapter"]
