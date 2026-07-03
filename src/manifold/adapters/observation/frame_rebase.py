"""Statically reorient a reported end-effector pose into a different frame.

Some envs report their proprioceptive pose in one fixed reference frame while the
policy was trained reading it in another, where "another" is a static reorientation
of the first — a constant change of basis, not a kinematic transform. GR00T/RLDX's
WidowXBridgeEnv is the case this exists for: it rotates the base-frame pose matrix
by a fixed `_DEFAULT_ROT` before reading euler angles.

This adapter rewrites the `ee_pose` rotation of the proprioception channel: it
decodes the rotation to a matrix `M`, applies the fixed change of basis
`M @ rotation.T`, and re-encodes in the ee_pose's *current* rotation format. The
frame label advances `source_frame -> target_frame`; position, gripper, and the
other channels pass through. Because the change of basis is a fixed rotation, the
re-encoding loses nothing, so it is declared lossless.

It only rewrites the rotation block, so it composes before a
`ProprioRotationAdapter` that then re-encodes the rotation format (e.g. quaternion
-> euler): rebase first, format second.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from manifold.core.adapter import ObservationAdapter
from manifold.core.conventions import Frame, ee_step_layout
from manifold.core.observation_space import ObservationSpace
from manifold.core.values import Observation
from manifold.lib.rotation import from_matrix, to_matrix


class FrameRebaseAdapter(ObservationAdapter):
    """Reorient the `ee_pose` rotation from `source_frame` into `target_frame`."""

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True

    def __init__(self, source_frame: Frame, target_frame: Frame, rotation: np.ndarray) -> None:
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.rotation = np.asarray(rotation, dtype=np.float64)

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
            raise TypeError("FrameRebaseAdapter needs a proprioception with an ee_pose")
        new_proprio = spec.proprioception.model_copy(
            update={"ee_pose": ee_pose.model_copy(update={"frame": self.target_frame})}
        )
        return spec.with_proprioception(new_proprio)

    def adapt(self, observation: Any, *, source: BaseModel) -> Observation:
        """Reorient the rotation portion of the `ee_pose` state array.

        Decodes the rotation block to a matrix, applies the fixed change of basis
        `M @ rotation.T`, and re-encodes in the same rotation format the block
        arrived in. Position and gripper qpos are untouched.
        """
        ee_pose = self._spec(source).proprioception.ee_pose
        if ee_pose is None:
            raise TypeError("FrameRebaseAdapter needs a proprioception with an ee_pose")
        block = [float(v) for v in observation.state["ee_pose"]]
        pos_len, rot_len, _ = ee_step_layout(ee_pose.rotation, None)
        rot_block = block[pos_len : pos_len + rot_len]
        matrix = to_matrix(rot_block, ee_pose.rotation)
        rebased = from_matrix(matrix @ self.rotation.T, ee_pose.rotation)
        reencoded = block[:pos_len] + rebased + block[pos_len + rot_len :]
        state = {**observation.state, "ee_pose": np.asarray(reencoded, dtype=np.float32)}
        return Observation(
            state=state, sensors=observation.sensors, instruction=observation.instruction
        )

    @staticmethod
    def _spec(source: BaseModel) -> ObservationSpace:
        if not isinstance(source, ObservationSpace):
            raise TypeError("FrameRebaseAdapter transforms an ObservationSpace source only")
        return source


__all__ = ["FrameRebaseAdapter"]
