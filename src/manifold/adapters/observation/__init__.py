"""Observation-space adapters: the adapters on the form the policy consumes."""

from __future__ import annotations

from manifold.adapters.observation.camera_channel_order import SwapChannelOrder
from manifold.adapters.observation.camera_orientation import Rotate180Cameras
from manifold.adapters.observation.camera_resolution import ResizeCameras
from manifold.adapters.observation.camera_vertical_flip import FlipVerticalCameras
from manifold.adapters.observation.dynamic_frame_rebase import DynamicFrameRebaseAdapter
from manifold.adapters.observation.frame_history import StackFrameHistory
from manifold.adapters.observation.frame_rebase import FrameRebaseAdapter
from manifold.adapters.observation.observed_gripper import ObservedGripperAdapter
from manifold.adapters.observation.proprio_rotation import ProprioRotationAdapter

__all__ = [
    "DynamicFrameRebaseAdapter",
    "FlipVerticalCameras",
    "FrameRebaseAdapter",
    "ObservedGripperAdapter",
    "ProprioRotationAdapter",
    "ResizeCameras",
    "Rotate180Cameras",
    "StackFrameHistory",
    "SwapChannelOrder",
]
