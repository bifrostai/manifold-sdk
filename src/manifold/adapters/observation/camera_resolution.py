"""Resize named camera images to a target resolution on the observation side.

A policy trained at one resolution can be served frames at another: unlike
orientation or channel order, a resolution mismatch is loud (a shape check
catches it) or absorbed (the policy resizes anyway), so it is never a hard
incompatibility — but it *is* lossy, since resampling discards or invents pixels.
This adapter resizes the named camera arrays and advances the spec: it sets the
declared `shape` of each named camera to the target. So a resolver may insert it
to bridge a resolution gap, and `check_compatibility` reports the pairing as
COMPATIBLE_VIA_PIPELINE but lossy (ADR-0001, decision 5).

Only the target *shape* is declared. Crop and aspect are not modeled
— the same no-speculative-modeling principle that keeps the action space
monolithic. Interpolation follows the frame's modality rather than being
parameterized, so it is not a spec axis and two instances with the same targets
produce identical specs.

A camera that declares calibration has its intrinsics resampled with it: focal
lengths and principal point scale by the same factors the image does, and `pad=True`
additionally shifts the principal point by the centring offset. Resizing pixels and
leaving the intrinsics behind would silently change the field of view the numbers
claim (ADR 0008). The extrinsic is untouched — resampling moves the pixel grid, not
the camera.

A depth camera resamples differently from a colour one (`_resample_depth` against
`_resample_colour`): interpolating metres is not interpolating colour, so a depth
frame resamples nearest-neighbour and pads with `inf`.

Everything else about the observation — the proprioception, the instruction, and
any cameras not named — passes through.

Parameterized either by a mapping of camera name to target shape, or by a set of
camera names plus one shared target `(H, W)`:

    ResizeCameras(targets={"agentview": (224, 224, 3)})
    ResizeCameras(cameras=("agentview", "wrist"), shape=(224, 224))
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel
from scipy.ndimage import zoom

from manifold.core.adapter import ObservationAdapter
from manifold.core.observation_space import ObservationSpace
from manifold.core.sensor import CameraIntrinsics, Modality
from manifold.core.values import Observation


def _resample_depth(arr: np.ndarray, target_h: int, target_w: int, *, pad: bool) -> np.ndarray:
    """Resample a depth frame: nearest-neighbour, padded with the `inf` no-hit marker."""
    return _resize(arr, target_h, target_w, pad=pad, order=0, pad_value=float("inf"))


def _resample_colour(arr: np.ndarray, target_h: int, target_w: int, *, pad: bool) -> np.ndarray:
    """Resample a colour frame: bilinear, padded black."""
    return _resize(arr, target_h, target_w, pad=pad, order=1, pad_value=0.0)


def _resize(
    arr: np.ndarray, target_h: int, target_w: int, *, pad: bool, order: int, pad_value: float
) -> np.ndarray:
    """Resample the image plane of `arr` to (target_h, target_w).

    Call `_resample_depth` or `_resample_colour` rather than this directly: they are
    the only two pairings of `order` and `pad_value` that are correct. Interpolating
    across a depth edge returns a distance nothing in the scene occupies, and a zero
    pad is a surface at the lens — both plausible values that no shape check can
    reject (ADR 0007). Nearest-neighbour is also the only order that preserves the
    `inf` no-hit marker at all: a combining kernel evaluates `inf - inf` across the
    spline filter and returns NaN, replacing the marker the SDK uses with one it
    does not.

    H, W are the trailing image axes (index -3, -2 in a `(..., H, W, C)` layout);
    leading axes (time/batch) and the trailing channel axis are preserved.

    With `pad`, the image plane scales by a single factor — the smaller of the two
    axis ratios — so it fits the target box without distortion, then centres in a
    canvas of `pad_value`, mirroring openpi's `image_tools.resize_with_pad`. Without
    it, each image axis is scaled alone and `pad_value` is unused.
    """
    zoom_factors = [1.0] * arr.ndim
    if not pad:
        zoom_factors[-3] = target_h / arr.shape[-3]
        zoom_factors[-2] = target_w / arr.shape[-2]
        return np.asarray(zoom(arr, zoom_factors, order=order))

    src_h, src_w = arr.shape[-3], arr.shape[-2]
    scale = min(target_h / src_h, target_w / src_w)
    # Round (not floor) so a perfectly-fitting axis lands exactly on the target.
    new_h = min(target_h, max(1, round(src_h * scale)))
    new_w = min(target_w, max(1, round(src_w * scale)))
    zoom_factors[-3] = new_h / src_h
    zoom_factors[-2] = new_w / src_w
    scaled = np.asarray(zoom(arr, zoom_factors, order=order))
    # Centre the scaled image in a canvas of `pad_value` at the target size: black
    # for a colour frame, `inf` (no hit) for a depth one.
    canvas_shape = (*arr.shape[:-3], target_h, target_w, arr.shape[-1])
    canvas = np.full(canvas_shape, pad_value, dtype=scaled.dtype)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[..., top : top + new_h, left : left + new_w, :] = scaled
    return canvas


def _image_hw(shape: tuple[int, ...]) -> tuple[int, int]:
    """The image (H, W) of a camera shape — the trailing two non-channel axes.

    A frame is `(H, W, C)` and a clip is `(T, H, W, C)`; in both the image plane
    is `shape[-3:-1]`, so resizing keys on the real H, W regardless of any leading
    time or batch axis.
    """
    return (int(shape[-3]), int(shape[-2]))


class ResizeCameras(ObservationAdapter):
    """Resize the named cameras to a target H x W, preserving channel count and dtype.

    Lossy on the seam: resampling is not invertible, so this flags as lossy and is
    opt-in (a resolver tries it only after lossless chains fail). Construct it
    either with a per-camera `targets` mapping, or with `cameras` + a shared
    `shape`; the target shape's leading two axes are taken as (H, W) and the
    channel count and dtype of each frame are preserved.
    """

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = False

    def __init__(
        self,
        targets: Mapping[str, tuple[int, ...]] | None = None,
        *,
        cameras: Sequence[str] | None = None,
        shape: tuple[int, int] | None = None,
        pad: bool = False,
    ) -> None:
        if targets is not None:
            if cameras is not None or shape is not None:
                raise ValueError("pass either `targets`, or `cameras` + `shape`, not both")
            self.targets = {name: tuple(target) for name, target in targets.items()}
        else:
            if cameras is None or shape is None:
                raise ValueError("pass either `targets`, or both `cameras` and `shape`")
            self.targets = {name: tuple(shape) for name in cameras}
        # When True, resize preserves aspect ratio (scale to fit the target box) and
        # pads the remainder to reach the target H x W — matching openpi's
        # `image_tools.resize_with_pad`. When False (the default), each axis is
        # scaled independently (plain anisotropic resample), preserving the prior
        # behaviour. Padding is not a spec axis: the declared output shape is the
        # target either way, so `produce`/`applies` are unaffected.
        self.pad = pad

    def applies(self, source: BaseModel) -> bool:
        """True when any named camera's image (H, W) differs from the target.

        The image plane is the *trailing* two non-channel axes — `shape[-3:-1]` for
        a `(..., H, W, C)` layout — so a rank-4 clip camera `(T, H, W, C)` compares
        on its real H, W, not its leading time axis.
        """
        if not isinstance(source, ObservationSpace):
            return False
        return any(
            (camera := source.camera(name)) is not None
            and _image_hw(camera.shape) != (target[0], target[1])
            for name, target in self.targets.items()
        )

    def produce(self, source: BaseModel) -> ObservationSpace:
        """The same spec with each named camera's image (H, W) set to the target.

        Only the trailing image axes are rewritten, leaving channel count and any
        leading time/batch axis intact (a clip stays rank-4). A camera that declares
        calibration has its intrinsics resampled to match.
        """
        if not isinstance(source, ObservationSpace):
            raise TypeError("ResizeCameras transforms an ObservationSpace source only")
        out = source
        for name, target in self.targets.items():
            camera = source.camera(name)
            if camera is None:
                continue
            shape = camera.shape
            # Replace the H, W axes in place: ..., H, W, C -> ..., target_h, target_w, C.
            new_shape = (*shape[:-3], int(target[0]), int(target[1]), *shape[-1:])
            update: dict[str, Any] = {"shape": new_shape}
            if camera.calibration is not None:
                update["calibration"] = camera.calibration.model_copy(
                    update={
                        "intrinsics": self._resampled_intrinsics(
                            camera.calibration.intrinsics, shape, target
                        )
                    }
                )
            out = out.with_camera(camera.model_copy(update=update))
        return out

    def _resampled_intrinsics(
        self,
        intrinsics: CameraIntrinsics,
        shape: tuple[int, ...],
        target: tuple[int, ...],
    ) -> CameraIntrinsics:
        """The intrinsics of a frame resampled from `shape` to `target`.

        The anisotropic path scales each axis by its own ratio. The padded path scales
        both by the single factor that fits the image in the box, then shifts the
        principal point by the offset the image was centred at — the pinhole is the
        same, and only where it sits in the new pixel grid has moved.
        """
        source_h, source_w = _image_hw(shape)
        target_h, target_w = int(target[0]), int(target[1])
        if not self.pad:
            return intrinsics.scaled(x=target_w / source_w, y=target_h / source_h)
        scale = min(target_h / source_h, target_w / source_w)
        new_h = min(target_h, max(1, round(source_h * scale)))
        new_w = min(target_w, max(1, round(source_w * scale)))
        return intrinsics.scaled(x=scale, y=scale).translated(
            x=(target_w - new_w) // 2, y=(target_h - new_h) // 2
        )

    def adapt(self, observation: Any, *, source: BaseModel) -> Observation:
        """Resize each named camera's image plane to the target H x W; rest passes through.

        A camera absent from `observation.sensors` is skipped. The result keeps the
        frame's dtype: bilinear zoom is a convex combination of the inputs, so it
        cannot exceed the input range, and the cast back to (e.g.) uint8 is safe.

        With `pad=True`, the resize preserves aspect ratio and pads the shrunk image,
        centred, to the full target H x W — matching openpi's `resize_with_pad`, so a
        non-square source keeps its proportions instead of being squashed.
        """
        sensors = dict(observation.sensors)
        for name, target in self.targets.items():
            frame = sensors.get(name)
            if frame is None:
                continue
            arr = np.asarray(frame)
            target_h, target_w = int(target[0]), int(target[1])
            camera = source.camera(name) if isinstance(source, ObservationSpace) else None
            resample = (
                _resample_depth
                if camera is not None and camera.modality is Modality.DEPTH
                else _resample_colour
            )
            resized = resample(arr, target_h, target_w, pad=self.pad)
            sensors[name] = np.ascontiguousarray(resized.astype(arr.dtype))
        return replace(observation, sensors=sensors)


__all__ = ["ResizeCameras"]
