"""Shared rotation fixture for tests pinning the non-wrapping quat→axis-angle convention.

`quat_to_axisangle` is the frozen ground-truth reference the adapter tests assert
byte-equality against. It is a test fixture only; the production conversion is
`manifold.lib.rotation.quat_to_axisangle_nonwrapping`.
"""

from __future__ import annotations

import numpy as np


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert an (x, y, z, w) quaternion to a 3-D axis-angle (rotation vector).

    Matches lerobot's ``LiberoProcessorStep._quat2axisangle`` exactly — angle =
    2*acos(w), axis = xyz / sqrt(1 - w**2) — the convention the GR00T/RLDX state
    modality was trained on, and the same conversion the RoboCasa runner needs.

    It does not canonicalize the quaternion sign, so for w < 0 it
    yields an angle in (pi, 2*pi) rather than wrapping into [0, pi]. scipy's
    ``Rotation.as_rotvec`` wraps into [0, pi] and flips the axis for w < 0, which
    differs by 2*pi on every component and, after the policy's mean/std state
    normalization, pushes the rotation dims ~17 std off distribution — enough to
    make the policy fail every task. So compute it by hand.
    """
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    axis = quat[:3] / den
    return (axis * angle).astype(np.float32)
