"""Tests that a simulator draws a body or a camera without failing on a bad rotation.

A simulator can report a rotation that is not a rotation: zeros for a body that has
not been simulated yet, NaN after a physics failure, or a matrix with a zero scale.
Each case here reads such a rotation and checks that the result is the rotation of
no rotation, and that nothing raises.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from manifold.benchmark.simulators.isaacsim import IsaacSimSimulator
from manifold.benchmark.simulators.sapien import SapienSimulator
from manifold.lib.rotation import rotation_matrix_from_xyzw, xyzw_from_rotation_matrix

# Quaternions in `xyzw` that are not rotations.
_BAD_QUATERNIONS = {
    "zeros": [0.0, 0.0, 0.0, 0.0],
    "nan": [np.nan, 0.0, 0.0, 1.0],
    "inf": [np.inf, 0.0, 0.0, 1.0],
    "tiny": [1e-300, 0.0, 0.0, 0.0],
    "short": [0.0, 0.0, 1.0],
}

# 3x3 matrices that are not rotations.
_BAD_MATRICES = {
    "zeros": np.zeros((3, 3)),
    "nan": np.full((3, 3), np.nan),
    "inf": np.full((3, 3), np.inf),
    "mirror": np.diag([1.0, 1.0, -1.0]),
}


@pytest.mark.parametrize("bad", sorted(_BAD_MATRICES))
def test_a_matrix_that_is_not_a_rotation_gives_no_rotation(bad: str) -> None:
    np.testing.assert_array_equal(
        xyzw_from_rotation_matrix(_BAD_MATRICES[bad]), [0.0, 0.0, 0.0, 1.0]
    )


def test_a_rotation_matrix_gives_its_quaternion() -> None:
    quarter_turn_about_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    np.testing.assert_allclose(
        xyzw_from_rotation_matrix(quarter_turn_about_z),
        [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        atol=1e-6,
    )


@pytest.mark.parametrize("bad", sorted(_BAD_QUATERNIONS))
def test_a_quaternion_that_is_not_a_rotation_gives_the_identity_matrix(bad: str) -> None:
    np.testing.assert_array_equal(rotation_matrix_from_xyzw(_BAD_QUATERNIONS[bad]), np.eye(3))


# SAPIEN's poses always have 4 values, so a short quaternion is not a SAPIEN case.
@pytest.mark.parametrize("bad", sorted(set(_BAD_QUATERNIONS) - {"short"}))
def test_a_sapien_shape_with_a_bad_rotation_is_drawn_unrotated(bad: str) -> None:
    xyzw = _BAD_QUATERNIONS[bad]
    local = SimpleNamespace(q=[xyzw[-1], *xyzw[:-1]], p=[0.0, 0.0, 1.0])
    vertices = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    (part,) = SapienSimulator._posed_parts([(vertices, [[0, 1, 2]], None)], None, local)

    np.testing.assert_allclose(part.mesh.vertices, vertices + np.array([0.0, 0.0, 1.0]))


@pytest.mark.parametrize("bad", sorted(_BAD_MATRICES))
def test_an_isaac_sim_body_with_a_bad_transform_is_posed_unrotated(bad: str) -> None:
    matrix = np.eye(4)
    matrix[:3, :3] = _BAD_MATRICES[bad]
    matrix[:3, 3] = [1.0, 2.0, 3.0]

    pose = IsaacSimSimulator._pose(matrix, 1.0)

    np.testing.assert_array_equal(pose.orientation, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(pose.position, [1.0, 2.0, 3.0])
