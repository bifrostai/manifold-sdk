"""Re-encode the observed gripper of a reported end-effector pose to a target encoding.

The observation-side analogue of the action-side `GripperPolarityAdapter`. A
benchmark may report its grasp state in one gripper encoding while the policy was
trained reading another. The canonical case is SIMPLER's ``1 - closedness``: the env
reports how *closed* the gripper is (an open-low reading), and a policy trained on an
open-high *width* expects the polarity flipped. Rather than bake ``1 - x`` into the
runner, the benchmark declares its observed gripper `encoding` on the
`GripperObservationSpec`, and this adapter remaps each gripper value to the target
encoding via `lib.gripper.remap` — the same openness round-trip the action-side
gripper adapters use, so the obs and action sides cannot drift.

The gripper occupies the trailing `gripper.dim` floats of the `ee_pose` value (after
position and rotation); each is remapped in place. Position, rotation, and every other
channel pass through. Because `remap` is a lossless representation conversion between
two continuous encodings, this adapter is declared lossless.

Parameterized by the target encoding: ``ObservedGripperAdapter(target=GripperFormat.
UNSIGNED)``. It only fires when the reported gripper declares an `encoding` that
differs from the target; a spec whose gripper is raw qpos (``encoding=None``, the
default) is left untouched — there is no declared polarity to bridge.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from manifold.core.adapter import ObservationAdapter
from manifold.core.conventions import GripperFormat, ee_step_layout
from manifold.core.observation_space import ObservationSpace
from manifold.core.values import Observation
from manifold.lib.gripper import remap


class ObservedGripperAdapter(ObservationAdapter):
    """Re-encode the observed `ee_pose` gripper to the `target` encoding."""

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True

    def __init__(self, target: GripperFormat) -> None:
        self.target = target

    def applies(self, source: BaseModel) -> bool:
        """True when the reported gripper declares an encoding differing from the target.

        A gripper with no declared encoding (raw qpos) is never touched — there is no
        declared polarity to bridge.
        """
        if not isinstance(source, ObservationSpace):
            return False
        ee_pose = source.proprioception.ee_pose
        return (
            ee_pose is not None
            and ee_pose.gripper is not None
            and ee_pose.gripper.encoding is not None
            and ee_pose.gripper.encoding != self.target
        )

    def produce(self, source: BaseModel) -> ObservationSpace:
        """The same spec with its observed gripper encoding set to the target."""
        spec = self._spec(source)
        ee_pose = spec.proprioception.ee_pose
        if ee_pose is None or ee_pose.gripper is None:
            raise TypeError("ObservedGripperAdapter needs an ee_pose with a gripper")
        new_gripper = ee_pose.gripper.model_copy(update={"encoding": self.target})
        new_proprio = spec.proprioception.model_copy(
            update={"ee_pose": ee_pose.model_copy(update={"gripper": new_gripper})}
        )
        return spec.with_proprioception(new_proprio)

    def adapt(self, observation: Any, *, source: BaseModel) -> Observation:
        """Remap each gripper-slot value of the `ee_pose` to the target encoding."""
        ee_pose = self._spec(source).proprioception.ee_pose
        if ee_pose is None or ee_pose.gripper is None or ee_pose.gripper.encoding is None:
            raise TypeError("ObservedGripperAdapter needs an ee_pose with an encoded gripper")
        source_encoding = ee_pose.gripper.encoding
        block = [float(v) for v in observation.state["ee_pose"]]
        pos_len, rot_len, _ = ee_step_layout(ee_pose.rotation, None)
        # The gripper occupies the trailing `gripper.dim` floats, after pose.
        gripper_start = pos_len + rot_len
        for i in range(gripper_start, gripper_start + ee_pose.gripper.dim):
            block[i] = remap(block[i], source_encoding, self.target)
        state = {**observation.state, "ee_pose": np.asarray(block, dtype=np.float32)}
        return replace(observation, state=state)

    @staticmethod
    def _spec(source: BaseModel) -> ObservationSpace:
        if not isinstance(source, ObservationSpace):
            raise TypeError("ObservedGripperAdapter transforms an ObservationSpace source only")
        return source


__all__ = ["ObservedGripperAdapter"]
