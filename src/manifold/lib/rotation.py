"""Convert a 3D rotation between encodings.

`convert` maps a rotation given as `source`-encoded floats to `target`-encoded
floats, by way of a scipy `Rotation`. The four encodings (see `RotationFormat`):
axis-angle (rotation vector, 3), quaternion (x, y, z, w, 4), extrinsic XYZ euler
(3), and the Zhou et al. 6D representation (the first two columns of the rotation
matrix, 6). scipy covers the first three directly; the 6D case builds the matrix
by Gram-Schmidt and reads back its first two columns.

scipy's quaternion order is scalar-last (x, y, z, w) and its lowercase euler axes
are extrinsic, matching the conventions here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from manifold.core.conventions import RotationFormat
from manifold.lib.compat import assert_never


def convert(values: Sequence[float], source: RotationFormat, target: RotationFormat) -> list[float]:
    """Convert a rotation from the `source` encoding to the `target` encoding."""
    return from_rotation(to_rotation(values, source), target)


def geodesic_angle(
    a: Sequence[float],
    a_fmt: RotationFormat,
    b: Sequence[float],
    b_fmt: RotationFormat,
) -> float:
    """Angle (radians) of the rotation carrying `a` onto `b`, each in its own encoding.

    Representation-invariant: it compares the decoded rotations, not their float
    encodings, so it is blind to artefacts that preserve the rotation (quaternion
    double-cover, axis-angle wrap, gimbal lock, 6D re-orthonormalisation). Two
    encodings of the same rotation return 0.0; the result is in [0, pi].
    """
    relative = to_rotation(a, a_fmt) * to_rotation(b, b_fmt).inv()
    return float(relative.magnitude())


def to_rotation(values: Sequence[float], fmt: RotationFormat) -> Rotation:
    """Decode `fmt`-encoded floats into a scipy `Rotation`."""
    array = np.asarray(values, dtype=np.float64)
    if fmt is RotationFormat.AXIS_ANGLE:
        return Rotation.from_rotvec(array)
    if fmt is RotationFormat.QUATERNION:
        return Rotation.from_quat(array)
    if fmt is RotationFormat.EULER_XYZ:
        return Rotation.from_euler("xyz", array)
    if fmt is RotationFormat.ROTATION_6D:
        return Rotation.from_matrix(matrix_from_6d(array))
    return assert_never(fmt)


def from_rotation(rotation: Rotation, fmt: RotationFormat) -> list[float]:
    """Encode a scipy `Rotation` into `fmt`-encoded floats."""
    if fmt is RotationFormat.AXIS_ANGLE:
        return rotation.as_rotvec().tolist()
    if fmt is RotationFormat.QUATERNION:
        return rotation.as_quat().tolist()
    if fmt is RotationFormat.EULER_XYZ:
        return rotation.as_euler("xyz").tolist()
    if fmt is RotationFormat.ROTATION_6D:
        return six_d_from_matrix(rotation.as_matrix())
    return assert_never(fmt)


def to_matrix(values: Sequence[float], fmt: RotationFormat) -> np.ndarray:
    """Decode `fmt`-encoded rotation floats into a 3x3 rotation matrix.

    The matrix-space counterpart of `to_rotation`, for callers that compose a change
    of basis in matrix form (the frame-rebase observation adapters). The 6D case
    returns the Gram-Schmidt matrix DIRECTLY rather than round-tripping through scipy,
    so it is bit-for-bit the matrix `matrix_from_6d` produces (`from_matrix` reads its
    first two columns straight back).
    """
    if fmt == RotationFormat.ROTATION_6D:
        return matrix_from_6d(np.asarray(values, dtype=np.float64))
    return to_rotation(values, fmt).as_matrix()


def from_matrix(matrix: np.ndarray, fmt: RotationFormat) -> list[float]:
    """Re-encode a 3x3 rotation matrix into `fmt`-encoded floats.

    Differs from `from_rotation` in one way: the QUATERNION branch canonicalizes to the
    positive-w hemisphere. scipy's `as_quat()` can emit the negative-w representative
    for rotations past 180°, and a frame-rebase adapter may feed its re-encoded block
    into a downstream non-wrapping quat->axis-angle path (`quat_to_axisangle_nonwrapping`),
    where a negative-w quaternion diverges by ~2pi on every axis — enough to push a
    policy off distribution. The runners (transforms3d.mat2quat) always return
    positive-w, so this matches them. `from_rotation` stays
    un-canonicalized for the byte-equivalence-sensitive `convert` path, which re-decodes
    through scipy's double-cover-aware `as_rotvec`.
    """
    if fmt == RotationFormat.ROTATION_6D:
        return six_d_from_matrix(matrix)
    rotation = Rotation.from_matrix(matrix)
    if fmt == RotationFormat.QUATERNION:
        quat = rotation.as_quat()
        if quat[3] < 0.0:
            quat = -quat
        return quat.tolist()
    return from_rotation(rotation, fmt)


def quat_to_axisangle_nonwrapping(values: Sequence[float] | np.ndarray) -> list[float]:
    """Convert an (x, y, z, w) quaternion to a 3D axis-angle WITHOUT the [0, pi] wrap.

    The conversion is angle = 2*acos(w), axis = xyz / sqrt(1 - w**2).
    ``tests/_rotation_fixtures.py:quat_to_axisangle`` is the frozen reference
    implementation, and this function is byte-equivalent to it on purpose.

    It does not canonicalize the quaternion sign, so for w < 0 it
    yields an angle in (pi, 2*pi) rather than wrapping into [0, pi]. scipy's
    ``Rotation.as_rotvec`` (the path `convert`/`to_rotation`/`from_rotation` take)
    wraps into [0, pi] and flips the axis for w < 0, which differs by 2*pi on every
    component and, after a policy's mean/std state normalization, pushes the rotation
    dims ~17 std off distribution — enough to make the policy fail every task. So
    the non-wrapping form is computed by hand here, matching the runner exactly,
    and is what `ProprioRotationAdapter(wrap=False)` routes through.

    The input is not force-upcast to float64; it is used at whatever dtype it
    arrives in, so the per-element arithmetic mirrors the runner's exactly (both
    keep `axis = quat[:3] / den` at the input dtype before the float64 `angle`
    scale). Fed the runner's float32 quaternion, the float32-truncated result is
    therefore byte-identical to the runner's, not merely close.

    The return value is a list of Python floats (float64 via `.tolist()`). The
    byte-equivalence to the runner's ``quat_to_axisangle`` holds **only after the
    caller casts back to float32** — which ``ProprioRotationAdapter`` does when it
    writes the adapted block back into the state array.
    """
    array = np.asarray(values)
    w = float(np.clip(array[3], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den <= 1e-10:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * np.arccos(w)
    axis = array[:3] / den
    return (axis * angle).tolist()


def matrix_from_6d(values: np.ndarray) -> np.ndarray:
    """Rebuild a rotation matrix from its first two columns (Gram-Schmidt).

    Raises ValueError on a degenerate input — a zero/near-zero first column, or a
    second column parallel to the first — rather than dividing by zero and letting
    NaN/inf propagate into `Rotation.from_matrix`.
    """
    a1, a2 = values[0:3], values[3:6]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        raise ValueError("degenerate 6D rotation: first column is zero")
    b1 = a1 / n1
    b2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(b2)
    if n2 < 1e-8:
        raise ValueError("degenerate 6D rotation: columns are parallel")
    b2 = b2 / n2
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


def six_d_from_matrix(matrix: np.ndarray) -> list[float]:
    """The first two columns of a rotation matrix, flattened (the Zhou et al. 6D form)."""
    return matrix[:, 0].tolist() + matrix[:, 1].tolist()


__all__ = [
    "convert",
    "from_matrix",
    "from_rotation",
    "geodesic_angle",
    "matrix_from_6d",
    "quat_to_axisangle_nonwrapping",
    "six_d_from_matrix",
    "to_matrix",
    "to_rotation",
]
