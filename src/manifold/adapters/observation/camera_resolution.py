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
monolithic. The interpolation order is an adapter parameter (not a spec axis),
defaulting to bilinear, so two instances differing only in interpolation produce
identical specs.

Everything else about the observation — the proprioception, the instruction, and
any cameras not named — passes through.

Parameterized either by a mapping of camera name to target shape, or by a set of
camera names plus one shared target `(H, W)`:

    ResizeCameras(targets={"agentview": (224, 224, 3)})
    ResizeCameras(cameras=("agentview", "wrist"), shape=(224, 224))
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel
from scipy.ndimage import zoom

from manifold.core.adapter import ObservationAdapter
from manifold.core.observation_space import ObservationSpace
from manifold.core.values import Observation

# Default interpolation order for scipy.ndimage.zoom: 1 is bilinear — a sensible
# middle ground (order 0 is nearest-neighbour, blocky; order 3 is bicubic, slow
# and prone to ringing on hard edges). It is a parameter, not a spec axis.
_DEFAULT_ORDER = 1


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
        order: int = _DEFAULT_ORDER,
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
        self.order = order
        # When True, resize preserves aspect ratio (scale to fit the target box) and
        # zero-pads the remainder to reach the target H x W — matching openpi's
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
        leading time/batch axis intact (a clip stays rank-4).
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
            out = out.with_camera(camera.model_copy(update={"shape": new_shape}))
        return out

    def adapt(self, observation: Any, *, source: BaseModel) -> Observation:
        """Resize each named camera's image plane to the target H x W; rest passes through.

        A camera absent from `observation.sensors` is skipped. The result keeps the
        frame's dtype: order-1 (bilinear) zoom is a convex combination of the inputs,
        so it cannot exceed the input range, and the cast back to (e.g.) uint8 is safe.

        With `pad=True`, the resize preserves aspect ratio and zero-pads the shrunk
        image, centred, to the full target H x W — matching openpi's `resize_with_pad`,
        so a non-square source keeps its proportions instead of being squashed.
        """
        sensors = dict(observation.sensors)
        for name, target in self.targets.items():
            frame = sensors.get(name)
            if frame is None:
                continue
            arr = np.asarray(frame)
            target_h, target_w = int(target[0]), int(target[1])
            resized = (
                self._pad_resize(arr, target_h, target_w)
                if self.pad
                else self._stretch(arr, target_h, target_w)
            )
            sensors[name] = np.ascontiguousarray(resized.astype(arr.dtype))
        return Observation(
            state=observation.state, sensors=sensors, instruction=observation.instruction
        )

    def _stretch(self, arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Anisotropic resample to (target_h, target_w): each image axis scaled alone."""
        # H, W are the trailing image axes (index -3, -2 in a (..., H, W, C)
        # layout); everything else (leading axes and the channel axis) is 1.0.
        zoom_factors = [1.0] * arr.ndim
        zoom_factors[-3] = target_h / arr.shape[-3]
        zoom_factors[-2] = target_w / arr.shape[-2]
        return np.asarray(zoom(arr, zoom_factors, order=self.order))

    def _pad_resize(self, arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Aspect-preserving resize-with-pad to (target_h, target_w), centred, zero-padded.

        Scales the image plane by a single factor (the smaller of the two axis
        ratios) so it fits inside the target box without distortion, then centres it
        in a zero-filled canvas of the target H x W. Leading axes (time/batch) and
        the trailing channel axis are preserved. Mirrors openpi's
        `image_tools.resize_with_pad`.
        """
        src_h, src_w = arr.shape[-3], arr.shape[-2]
        scale = min(target_h / src_h, target_w / src_w)
        # Round (not floor) so a perfectly-fitting axis lands exactly on the target.
        new_h = min(target_h, max(1, round(src_h * scale)))
        new_w = min(target_w, max(1, round(src_w * scale)))
        zoom_factors = [1.0] * arr.ndim
        zoom_factors[-3] = new_h / src_h
        zoom_factors[-2] = new_w / src_w
        scaled = np.asarray(zoom(arr, zoom_factors, order=self.order))
        # Centre the scaled image in a zero (black) canvas at the target size.
        canvas_shape = (*arr.shape[:-3], target_h, target_w, arr.shape[-1])
        canvas = np.zeros(canvas_shape, dtype=scaled.dtype)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[..., top : top + new_h, left : left + new_w, :] = scaled
        return canvas


__all__ = ["ResizeCameras"]
