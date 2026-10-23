import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manifold.core.conventions import CameraAxes, GripperFormat, RotationFormat
from manifold.lib.gripper import is_gripper_shut, remap
from manifold.lib.rotation import (
    convert,
    convert_camera_axes,
    from_matrix,
    matrix_from_6d,
    to_matrix,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

# The value that means "fully open" in each gripper encoding.
_OPEN = {
    GripperFormat.SIGNED: 1.0,
    GripperFormat.SIGNED_OPEN_LOW: -1.0,
    GripperFormat.UNSIGNED: 1.0,
    GripperFormat.UNSIGNED_OPEN_LOW: 0.0,
    GripperFormat.BINARY: 1.0,
    GripperFormat.BINARY_OPEN_LOW: 0.0,
}


@pytest.mark.parametrize("source", list(_OPEN))
@pytest.mark.parametrize("target", list(_OPEN))
def test_open_gripper_stays_open_across_every_encoding(
    source: GripperFormat, target: GripperFormat
) -> None:
    # This is the silent-failure axis: a remap must never turn an open gripper closed.
    assert remap(_OPEN[source], source, target) == _OPEN[target]


@pytest.mark.parametrize(
    "target",
    [RotationFormat.QUATERNION, RotationFormat.EULER_XYZ, RotationFormat.ROTATION_6D],
)
def test_rotation_round_trips_through_every_format(target: RotationFormat) -> None:
    axis_angle = [0.3, -0.5, 1.2]
    there = convert(axis_angle, RotationFormat.AXIS_ANGLE, target)
    back = convert(there, target, RotationFormat.AXIS_ANGLE)
    assert np.allclose(axis_angle, back, atol=1e-6)


# --- the matrix-space codec the frame-rebase adapters share -----------------------


@pytest.mark.parametrize(
    "fmt",
    [
        RotationFormat.AXIS_ANGLE,
        RotationFormat.QUATERNION,
        RotationFormat.EULER_XYZ,
        RotationFormat.ROTATION_6D,
    ],
)
def test_to_from_matrix_round_trips_each_format(fmt: RotationFormat) -> None:
    matrix = Rotation.from_euler("xyz", [0.3, -0.6, 1.2]).as_matrix()
    values = from_matrix(matrix, fmt)
    assert np.allclose(to_matrix(values, fmt), matrix, atol=1e-6)


def test_from_matrix_canonicalizes_quaternion_to_positive_w() -> None:
    # A rotation past 180° whose scipy as_quat() emits the negative-w representative;
    # the frame-rebase codec must canonicalize it (a negative-w quat would diverge ~2pi
    # through the downstream non-wrapping quat->axis-angle path).
    matrix = Rotation.from_euler("z", 270, degrees=True).as_matrix()
    assert Rotation.from_matrix(matrix).as_quat()[3] < 0.0  # scipy's raw representative
    quat = from_matrix(matrix, RotationFormat.QUATERNION)
    assert quat[3] >= 0.0  # canonicalized
    assert np.allclose(Rotation.from_quat(quat).as_matrix(), matrix, atol=1e-6)  # same rotation


def test_matrix_from_6d_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        matrix_from_6d(np.zeros(6))  # zero first column
    with pytest.raises(ValueError):
        matrix_from_6d(np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0]))  # second column parallel


def test_safe_int_rejects_bool_and_non_integral_float() -> None:
    from manifold.recipes.lerobot import _safe_int

    assert _safe_int(4) == 4
    assert _safe_int(4.0) == 4
    assert _safe_int("4") == 4
    with pytest.raises(TypeError):
        _safe_int(True)  # a bool is an int subclass but never a real dimension
    with pytest.raises(ValueError):
        _safe_int(4.5)  # non-integral float is malformed, not truncatable


def test_a_quaternion_reorders_between_scalar_first_and_scalar_last():
    wxyz = [0.5, 0.1, 0.2, 0.3]
    xyzw = wxyz_to_xyzw(wxyz)
    np.testing.assert_array_equal(xyzw, [0.1, 0.2, 0.3, 0.5])
    np.testing.assert_array_equal(xyzw_to_wxyz(xyzw), wxyz)


def test_a_camera_rotation_changes_between_opengl_and_opencv_axes():
    rotation = to_matrix([0.0, 0.0, 0.3], RotationFormat.AXIS_ANGLE)
    opencv = convert_camera_axes(rotation, CameraAxes.OPENGL, CameraAxes.OPENCV)
    # The camera looks down -z in OpenGL axes and down +z in OpenCV axes.
    np.testing.assert_allclose(opencv[:, 2], -rotation[:, 2])
    np.testing.assert_allclose(opencv[:, 0], rotation[:, 0])
    back = convert_camera_axes(opencv, CameraAxes.OPENCV, CameraAxes.OPENGL)
    np.testing.assert_allclose(back, rotation)
    same = convert_camera_axes(rotation, CameraAxes.OPENCV, CameraAxes.OPENCV)
    np.testing.assert_allclose(same, rotation)


@pytest.mark.parametrize(
    ("value", "shut"),
    [
        (1.0, True),
        (0.51, True),
        (0.5, None),
        (0.3, None),
        (-0.5, None),
        (-0.51, False),
        (-1.0, False),
    ],
)
def test_a_continuous_gripper_has_a_middle_band_that_commands_neither(value, shut):
    # `signed_open_low` runs from -1 open to +1 shut.
    assert is_gripper_shut(value, GripperFormat.SIGNED_OPEN_LOW) is shut


@pytest.mark.parametrize(("value", "shut"), [(1.0, True), (0.51, True), (0.5, False), (0.0, False)])
def test_a_discrete_gripper_is_shut_in_the_closed_half(value, shut):
    # `binary_open_low` runs from 0 open to 1 shut.
    assert is_gripper_shut(value, GripperFormat.BINARY_OPEN_LOW) is shut
