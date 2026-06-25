"""Unit tests for the symmetric-seam-bridge convention adapter set.

These adapters are dormant library code (no pairing wires them yet), so each test
exercises the adapter directly: build a synthetic source `ObservationSpace` and a
matching `Observation`, run the adapter's `produce`/`adapt`, and assert the math.
The byte-equivalence test for the non-wrapping quat->axis-angle is the load-bearing
one — it pins the adapter to the frozen reference conversion.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from manifold.adapters.observation import (
    DynamicFrameRebaseAdapter,
    ObservedGripperAdapter,
    ProprioRotationAdapter,
)
from manifold.core.conventions import Frame, GripperFormat, RotationFormat
from manifold.core.embodiment import (
    EEObservationSpec,
    GripperObservationSpec,
    Proprioception,
)
from manifold.core.observation_space import ObservationSpace
from manifold.core.values import Observation
from manifold.lib.rotation import quat_to_axisangle_nonwrapping
from tests._rotation_fixtures import quat_to_axisangle


def _obs_space(
    rotation: RotationFormat,
    *,
    frame: Frame = Frame.WORLD,
    gripper: GripperObservationSpec | None = None,
) -> ObservationSpace:
    return ObservationSpace(
        proprioception=Proprioception(
            ee_pose=EEObservationSpec(rotation=rotation, frame=frame, gripper=gripper)
        )
    )


# --- 1. DynamicFrameRebaseAdapter ------------------------------------------------


def test_dynamic_rebase_matches_inv_base_at_ee_for_position_and_rotation() -> None:
    # World-frame ee pose and a non-trivial world-frame base pose.
    ee_rot = Rotation.from_euler("xyz", [0.3, -0.2, 0.5])
    ee_pos = np.array([0.4, 0.1, 0.7])
    base_rot = Rotation.from_euler("xyz", [0.0, 0.0, 1.1])
    base_pos = np.array([0.2, -0.3, 0.05])

    base_mat = np.eye(4)
    base_mat[:3, :3] = base_rot.as_matrix()
    base_mat[:3, 3] = base_pos
    ee_mat = np.eye(4)
    ee_mat[:3, :3] = ee_rot.as_matrix()
    ee_mat[:3, 3] = ee_pos

    # Expected ee_in_base, the exact SIMPLER runner composition.
    expected = np.linalg.inv(base_mat) @ ee_mat
    expected_pos = expected[:3, 3]
    expected_quat = Rotation.from_matrix(expected[:3, :3]).as_quat()  # xyzw

    source = _obs_space(RotationFormat.QUATERNION, frame=Frame.WORLD)
    adapter = DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE)
    assert adapter.applies(source)
    produced = adapter.produce(source)
    assert produced.proprioception.ee_pose is not None
    assert produced.proprioception.ee_pose.frame == Frame.BASE

    ee_block = np.concatenate([ee_pos, ee_rot.as_quat()]).astype(np.float32)
    observation = Observation(
        state={"ee_pose": ee_block, "base_pose": base_mat.reshape(-1).astype(np.float32)}
    )
    result = adapter.adapt(observation, source=source)
    out = result.state["ee_pose"]

    assert np.allclose(out[:3], expected_pos, atol=1e-5)
    # Quaternion double-cover: compare rotations, not raw floats.
    out_rot = Rotation.from_quat(out[3:7])
    rel = (out_rot * Rotation.from_quat(expected_quat).inv()).magnitude()
    assert rel < 1e-5
    # base_pose channel passes through untouched.
    assert np.allclose(result.state["base_pose"], base_mat.reshape(-1))


def test_dynamic_rebase_accepts_pos_quat_base_pose() -> None:
    base_rot = Rotation.from_euler("xyz", [0.1, 0.2, -0.4])
    base_pos = np.array([0.1, 0.2, 0.3])
    base_mat = np.eye(4)
    base_mat[:3, :3] = base_rot.as_matrix()
    base_mat[:3, 3] = base_pos
    pos_quat = np.concatenate([base_pos, base_rot.as_quat()]).astype(np.float32)

    source = _obs_space(RotationFormat.QUATERNION, frame=Frame.WORLD)
    adapter = DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE)

    ee_pos = np.array([0.5, 0.0, 0.6])
    ee_quat = Rotation.from_euler("xyz", [0.0, 0.3, 0.0]).as_quat()
    observation_mat = Observation(
        state={
            "ee_pose": np.concatenate([ee_pos, ee_quat]).astype(np.float32),
            "base_pose": base_mat.reshape(-1).astype(np.float32),
        }
    )
    observation_pq = Observation(
        state={
            "ee_pose": np.concatenate([ee_pos, ee_quat]).astype(np.float32),
            "base_pose": pos_quat,
        }
    )
    out_mat = adapter.adapt(observation_mat, source=source).state["ee_pose"]
    out_pq = adapter.adapt(observation_pq, source=source).state["ee_pose"]
    assert np.allclose(out_mat, out_pq, atol=1e-5)


def test_dynamic_rebase_does_not_fire_when_already_in_target_frame() -> None:
    source = _obs_space(RotationFormat.QUATERNION, frame=Frame.BASE)
    adapter = DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE)
    assert not adapter.applies(source)


def test_dynamic_rebase_quaternion_output_is_positive_w() -> None:
    # A rebase whose net rotation exceeds 180° triggers scipy's as_quat() to
    # return the negative-w representative.  The adapter must canonicalize to
    # positive-w before writing back — the runners' transforms3d.mat2quat always
    # returns positive-w, and feeding a negative-w quaternion to the non-wrapping
    # quat→axis-angle path diverges by ~2π on every axis.
    #
    # Fixture: base 30° about Z, ee 240° about Z (both world-frame).
    # inv(base) @ ee = 210° about Z, whose quaternion is w ≈ -0.259 (negative-w).
    base_rot = Rotation.from_rotvec([0.0, 0.0, np.radians(30)])
    ee_rot = Rotation.from_rotvec([0.0, 0.0, np.radians(240)])
    base_pos = np.array([0.0, 0.0, 0.0])
    ee_pos = np.array([0.3, 0.1, 0.5])

    base_mat = np.eye(4)
    base_mat[:3, :3] = base_rot.as_matrix()
    base_mat[:3, 3] = base_pos
    ee_mat = np.eye(4)
    ee_mat[:3, :3] = ee_rot.as_matrix()
    ee_mat[:3, 3] = ee_pos

    # Verify the fixture actually produces negative-w from scipy (test precondition).
    expected_combined = np.linalg.inv(base_mat) @ ee_mat
    raw_quat = Rotation.from_matrix(expected_combined[:3, :3]).as_quat()
    assert raw_quat[3] < 0.0, "fixture doesn't produce negative-w; adjust the angles"

    source = _obs_space(RotationFormat.QUATERNION, frame=Frame.WORLD)
    adapter = DynamicFrameRebaseAdapter(Frame.WORLD, Frame.BASE)

    ee_block = np.concatenate([ee_pos, ee_rot.as_quat()]).astype(np.float32)
    observation = Observation(
        state={"ee_pose": ee_block, "base_pose": base_mat.reshape(-1).astype(np.float32)}
    )
    result = adapter.adapt(observation, source=source)
    out_quat = result.state["ee_pose"][3:7]

    # The adapter must canonicalize to positive-w.
    assert out_quat[3] >= 0.0, f"negative w={out_quat[3]:.6f}; canonicalization not applied"

    # The rotation must still be correct: geodesic distance to the expected combined rotation.
    out_rot = Rotation.from_quat(out_quat)
    expected_rot = Rotation.from_matrix(expected_combined[:3, :3])
    geodesic = (out_rot * expected_rot.inv()).magnitude()
    assert geodesic < 1e-5, f"rotation diverged by {geodesic:.2e} rad after canonicalization"


# --- 2. ObservedGripperAdapter ---------------------------------------------------


def test_observed_gripper_flips_closedness_to_width() -> None:
    # SIMPLER reports closedness as open-low (1=closed, 0=open); a width policy reads
    # open-high. Remap UNSIGNED_OPEN_LOW -> UNSIGNED is exactly `1 - closedness`.
    gripper = GripperObservationSpec(dim=1, encoding=GripperFormat.UNSIGNED_OPEN_LOW)
    source = _obs_space(RotationFormat.QUATERNION, gripper=gripper)
    adapter = ObservedGripperAdapter(target=GripperFormat.UNSIGNED)
    assert adapter.applies(source)

    produced = adapter.produce(source)
    assert produced.proprioception.ee_pose is not None
    assert produced.proprioception.ee_pose.gripper is not None
    assert produced.proprioception.ee_pose.gripper.encoding == GripperFormat.UNSIGNED

    # pos(3) + quat(4) + gripper(1): closedness 0.3 -> width 0.7.
    block = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3], dtype=np.float32)
    result = adapter.adapt(Observation(state={"ee_pose": block}), source=source)
    out = result.state["ee_pose"]
    assert np.isclose(out[7], 0.7)
    # pose untouched.
    assert np.allclose(out[:7], block[:7])


def test_observed_gripper_skips_raw_qpos_with_no_declared_encoding() -> None:
    # The default GripperObservationSpec(dim=2) does not declare an encoding -> nothing to bridge.
    gripper = GripperObservationSpec(dim=2)
    source = _obs_space(RotationFormat.AXIS_ANGLE, gripper=gripper)
    adapter = ObservedGripperAdapter(target=GripperFormat.UNSIGNED)
    assert not adapter.applies(source)


# --- 3. Non-wrapping quat->axis-angle (byte-equivalence) -------------------------


def test_nonwrapping_quat_to_axisangle_is_byte_equivalent_to_runner() -> None:
    # Several rotations incl. w<0 (the wrap case scipy gets wrong). Feed float32,
    # the dtype the runner feeds, so the result is byte-identical, not merely close.
    quats = [
        np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32),
        np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.6, 0.0, 0.0, -0.8], dtype=np.float32),
        np.array([-0.2, 0.4, -0.1, -0.7], dtype=np.float32),
    ]
    for quat in quats:
        runner = quat_to_axisangle(quat)
        lib = np.asarray(quat_to_axisangle_nonwrapping(quat), dtype=np.float32)
        assert np.array_equal(runner, lib), f"mismatch for {quat}: {runner} != {lib}"


def test_nonwrapping_differs_from_scipy_wrap_for_negative_w() -> None:
    # w<0: the non-wrapping angle exceeds pi; scipy's as_rotvec wraps into [0, pi].
    quat = np.array([0.6, 0.0, 0.0, -0.8], dtype=np.float32)
    nonwrap = np.asarray(quat_to_axisangle_nonwrapping(quat), dtype=np.float64)
    scipy_wrap = Rotation.from_quat(quat).as_rotvec()
    assert np.linalg.norm(nonwrap) > np.pi
    assert np.linalg.norm(scipy_wrap) <= np.pi
    # They are not close — this is the ~2pi divergence the wrap introduces.
    assert not np.allclose(nonwrap, scipy_wrap)


def test_proprio_rotation_wrap_false_routes_quat_to_nonwrapping() -> None:
    # ProprioRotationAdapter(wrap=False) must reproduce the runner on the ee_pose
    # rotation block for QUATERNION -> AXIS_ANGLE; wrap=True keeps the scipy path.
    quat = np.array([0.6, 0.0, 0.0, -0.8], dtype=np.float32)
    pos = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    source = _obs_space(RotationFormat.QUATERNION)
    observation = Observation(state={"ee_pose": np.concatenate([pos, quat])})

    nonwrap = ProprioRotationAdapter(target=RotationFormat.AXIS_ANGLE, wrap=False)
    out = nonwrap.adapt(observation, source=source).state["ee_pose"]
    expected = quat_to_axisangle(quat)
    assert np.allclose(out[:3], pos)
    # The adapter decodes via [float(v) ...] (float64), so compare in float64.
    assert np.allclose(out[3:6], expected.astype(np.float64), atol=1e-6)
    assert np.linalg.norm(out[3:6]) > np.pi

    wrap = ProprioRotationAdapter(target=RotationFormat.AXIS_ANGLE, wrap=True)
    out_wrap = wrap.adapt(observation, source=source).state["ee_pose"]
    assert np.linalg.norm(out_wrap[3:6]) <= np.pi


def test_proprio_rotation_default_is_unchanged_scipy_path() -> None:
    # The default (wrap=True) must match the pre-existing scipy conversion exactly.
    quat = np.array([0.6, 0.0, 0.0, -0.8], dtype=np.float32)
    pos = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    source = _obs_space(RotationFormat.QUATERNION)
    observation = Observation(state={"ee_pose": np.concatenate([pos, quat])})

    out = ProprioRotationAdapter(target=RotationFormat.AXIS_ANGLE).adapt(observation, source=source)
    expected = Rotation.from_quat(quat.astype(np.float64)).as_rotvec()
    assert np.allclose(out.state["ee_pose"][3:6], expected, atol=1e-6)
