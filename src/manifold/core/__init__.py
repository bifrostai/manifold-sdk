"""The kinds a policy and a benchmark are described in, and the check between them.

This is the mechanism layer: the primitives (embodiment, sensor, benchmark,
policy signature), the per-step values (observation, action), the wire codec, the
typed adapters and the pipeline, and `check_compatibility`. It does not decide
pairing strategy (which adapter to use, how to chain, how to default); that lives in
`recipes/`.
"""

from __future__ import annotations

from manifold.core.adapter import (
    ActionAdapter,
    ObservationAdapter,
)
from manifold.core.benchmark import Benchmark
from manifold.core.check import Compatibility, Report, check_compatibility
from manifold.core.conventions import Frame, GripperFormat, RotationFormat
from manifold.core.embodiment import (
    ActionSpace,
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    GripperObservationSpec,
    JointActionSpace,
    JointObservationSpec,
    Proprioception,
    UnifiedActionSpace,
)
from manifold.core.native_layout import (
    Assemble,
    BatchAxis,
    ChannelKind,
    Component,
    DtypeCast,
    LayoutEntry,
    LayoutOp,
    NativeLayout,
    Slice,
    SourceKind,
    Split,
)
from manifold.core.observation_space import ObservationSpace
from manifold.core.pipeline import Pack, Pipeline, Unpack
from manifold.core.policy import PolicySignature
from manifold.core.sensor import Camera, CameraOrientation, ChannelOrder, Modality, Mount, Sensor
from manifold.core.state import DEFAULT_LANE, LaneKey, PipelineState
from manifold.core.values import Action, Observation
from manifold.core.verify import VerifyCheck, VerifyReport, verify

__all__ = [
    "DEFAULT_LANE",
    "Action",
    "ActionAdapter",
    "ActionSpace",
    "Assemble",
    "BatchAxis",
    "Benchmark",
    "Camera",
    "CameraOrientation",
    "ChannelKind",
    "ChannelOrder",
    "Compatibility",
    "Component",
    "DtypeCast",
    "EEActionSpace",
    "EEObservationSpec",
    "Embodiment",
    "Frame",
    "GripperFormat",
    "GripperObservationSpec",
    "JointActionSpace",
    "JointObservationSpec",
    "LaneKey",
    "LayoutEntry",
    "LayoutOp",
    "Modality",
    "Mount",
    "NativeLayout",
    "Observation",
    "ObservationAdapter",
    "ObservationSpace",
    "Pack",
    "Pipeline",
    "PipelineState",
    "PolicySignature",
    "Proprioception",
    "Report",
    "RotationFormat",
    "Sensor",
    "Slice",
    "SourceKind",
    "Split",
    "UnifiedActionSpace",
    "Unpack",
    "VerifyCheck",
    "VerifyReport",
    "check_compatibility",
    "verify",
]
