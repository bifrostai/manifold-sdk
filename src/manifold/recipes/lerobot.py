"""LeRobot checkpoint introspection helper — populate-and-confirm, not a verifier.

Reads a lerobot checkpoint's `config.json` and returns a `SignatureSuggestion`
whose `.suggested` fields give the author a head start toward a `PolicySignature`.
This is format glue in `recipes/`, not core — it only sees what the format
records. Most checkpoints encode neither frame convention nor gripper polarity,
so those fields land in `.undetermined` with an explanatory note.

ADR-0001, decision 5: "Introspection is a head start, not a verifier. A
`recipes/`-level, format-specific helper reads a checkpoint's own metadata and
*suggests* signature fields for the author to confirm."
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from manifold.core.embodiment.action_space import JointActionSpace
from manifold.core.policy import PolicySignature
from manifold.core.sensor import Camera


@dataclass(frozen=True)
class UndeterminedField:
    """A field the introspection helper could not determine from the checkpoint.

    `name` is the machine-readable field name (e.g. ``"rotation_format"``). `note` is
    a human-readable explanation of why this field cannot be inferred.
    """

    name: str
    note: str


# Always-undetermined fields — lerobot never records these.
_ALWAYS_UNDETERMINED: list[UndeterminedField] = [
    UndeterminedField(
        name="rotation_format",
        note=(
            "lerobot does not record the rotation encoding used by the policy "
            "(AXIS_ANGLE / QUATERNION / EULER_XYZ / ROTATION_6D). "
            "Check the model card or training config."
        ),
    ),
    UndeterminedField(
        name="frame",
        note=(
            "lerobot does not record the reference frame for end-effector actions "
            "(WORLD / BASE / TOOL). Check the dataset README."
        ),
    ),
    UndeterminedField(
        name="gripper_format",
        note=(
            "lerobot does not record gripper polarity or range "
            "(SIGNED / UNSIGNED / BINARY and their _OPEN_LOW variants). "
            "Check the robot's action convention."
        ),
    ),
    UndeterminedField(
        name="channel_order",
        note=(
            "lerobot does not record RGB vs BGR channel order. "
            "Most lerobot datasets use RGB; verify against the dataset loader."
        ),
    ),
    UndeterminedField(
        name="image_orientation",
        note=(
            "lerobot does not record whether images are upright or rotated 180°. "
            "Inspect a sample frame to confirm."
        ),
    ),
]


def _is_language_key(key: str) -> bool:
    """Return True if the feature key looks like a language/instruction feature."""
    lower = key.lower()
    return "language" in lower or "instruction" in lower


def _safe_int(value: object) -> int:
    """Cast *value* to int from an int, an integral float, or a numeric string.

    Rejects `bool` (an `int` subclass, but never a real tensor dimension) and a
    non-integral float (a fractional shape dim is malformed, not silently truncatable).
    """
    if isinstance(value, bool):
        raise TypeError("a bool is not a valid int dimension")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"cannot convert non-integral float {value!r} to int")
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"cannot convert {type(value).__name__!r} to int")


def _parse_input_features(
    features: object,
) -> tuple[list[CameraSuggestion], int | None, bool]:
    """Parse ``input_features`` into a ``(cameras, state_dim, instruction)`` triple."""
    if not isinstance(features, dict):
        return [], None, False

    cameras: list[CameraSuggestion] = []
    state_dim: int | None = None
    instruction: bool = False

    for raw_key, feat in features.items():
        if not isinstance(raw_key, str) or not isinstance(feat, dict):
            continue
        key: str = raw_key
        feat_type = str(feat.get("type", "")).upper()
        shape_raw = feat.get("shape")

        if _is_language_key(key) or feat_type in ("LANGUAGE", "TEXT"):
            instruction = True
        elif feat_type == "VISUAL":
            # Lerobot stores images channels-first: [C, H, W] → (H, W, C).
            if isinstance(shape_raw, (list, tuple)) and len(shape_raw) == 3:
                with contextlib.suppress(TypeError, ValueError):
                    c = _safe_int(shape_raw[0])
                    h = _safe_int(shape_raw[1])
                    w = _safe_int(shape_raw[2])
                    cam_name = key.removeprefix("observation.images.")
                    cameras.append(CameraSuggestion(name=cam_name, shape=(h, w, c)))
        elif feat_type == "STATE" and isinstance(shape_raw, (list, tuple)) and len(shape_raw) >= 1:
            with contextlib.suppress(TypeError, ValueError):
                state_dim = _safe_int(shape_raw[0])

    return cameras, state_dim, instruction


def _parse_output_features(features: object) -> int | None:
    """Parse ``output_features`` and return ``action_dim``, or ``None`` if not found."""
    if not isinstance(features, dict):
        return None

    for feat in features.values():
        if not isinstance(feat, dict):
            continue
        feat_type = str(feat.get("type", "")).upper()
        shape_raw = feat.get("shape")
        if feat_type == "ACTION" and isinstance(shape_raw, (list, tuple)) and len(shape_raw) >= 1:
            with contextlib.suppress(TypeError, ValueError):
                return _safe_int(shape_raw[0])
    return None


@dataclass(frozen=True)
class CameraSuggestion:
    """A camera found in the checkpoint's input features.

    `name` is the camera name, derived by stripping the ``observation.images.``
    prefix from the lerobot feature key. `shape` is the image shape as ``(H, W, C)``,
    converted from lerobot's channels-first ``[C, H, W]`` storage. `note` is a
    reminder that orientation and channel order are assumptions, not recorded by
    lerobot.
    """

    name: str
    shape: tuple[int, int, int]
    note: str = (
        "orientation=UPRIGHT and channel_order=RGB are assumed — lerobot does not record them"
    )


@dataclass(frozen=True)
class SuggestedFields:
    """Fields that could be inferred (with varying confidence) from the checkpoint.

    `cameras` are the camera views found in the input features. `action_dim` is the
    total action vector length from the output feature shape, or ``None`` if not
    found. `state_dim` is the proprioception / state vector length, or ``None`` if not
    found. `instruction` is whether a language feature was detected.
    `normalization_info` is the raw ``normalization_mapping`` from the config,
    surfaced as informational text so the author can see which stats to wire into their
    wrapper — this is not a signature field, normalization is policy-internal (see
    ADR-0001).
    """

    cameras: list[CameraSuggestion]
    action_dim: int | None
    state_dim: int | None
    instruction: bool
    normalization_info: dict[str, str]


@dataclass(frozen=True)
class SignatureSuggestion:
    """The result of introspecting a lerobot checkpoint.

    Carries two disjoint buckets: what was inferred (``suggested``) and what
    the checkpoint does not record (``undetermined``). The author must fill in
    every ``undetermined`` field before the result can become a verified
    ``PolicySignature``.

    `suggested` holds the fields inferred from the checkpoint metadata.
    `undetermined` holds the fields that could not be inferred; each entry gives the
    field and the reason.
    """

    suggested: SuggestedFields
    undetermined: list[UndeterminedField]

    def draft_policy_spec(self) -> object:
        """Build a draft ``PolicySignature`` — a *starting point to edit*, not a verified signature.

        Because the checkpoint does not record the action kind, rotation format,
        gripper format, or reference frame, this method constructs a
        ``JointActionSpace`` as the most conservative fallback: it requires only
        the total action dimension, avoids assuming rotation encoding, and sets
        ``gripper=None``. The author must replace this with the correct spec.

        Returns a ``PolicySignature`` whose ``action_space`` is a ``JointActionSpace``
        with ``dof=action_dim``. Cameras are stubbed with ``UPRIGHT`` orientation and
        ``RGB`` channel order (the same assumptions recorded in each
        ``CameraSuggestion``). Proprioception is left empty — the author must wire the
        correct ``EEObservationSpec`` or ``JointObservationSpec``. Raises
        ``ValueError`` if ``action_dim`` is ``None`` (the output feature was missing
        or malformed), because a ``PolicySignature`` requires an action space.
        """
        if self.suggested.action_dim is None:
            raise ValueError(
                "Cannot build a draft PolicySignature: action_dim is None "
                "(the checkpoint's output feature was missing or malformed). "
                "Inspect the config.json and set the action space manually."
            )

        cameras = [
            Camera(
                name=cam.name,
                shape=cam.shape,
            )
            for cam in self.suggested.cameras
        ]

        action_space = JointActionSpace(
            dof=self.suggested.action_dim,
            gripper=None,
        )

        return PolicySignature(
            action_space=action_space,
            cameras=cameras,
            instruction=self.suggested.instruction,
        )


def from_lerobot_checkpoint(path: str | Path) -> SignatureSuggestion:
    """Read a lerobot checkpoint directory and suggest ``PolicySignature`` fields.

    Reads ``config.json`` from ``path`` (a directory) and extracts what can be
    inferred structurally. Missing or malformed keys are silently skipped — the
    function never raises; instead it populates more items in ``undetermined``.

    `path` is the lerobot checkpoint directory that contains ``config.json``. Returns
    a ``SignatureSuggestion`` with ``suggested`` fields populated where possible and
    ``undetermined`` fields for everything the format does not record.
    """
    path = Path(path)
    config_path = path / "config.json"
    extra_undetermined: list[UndeterminedField] = []

    try:
        with config_path.open() as fh:
            raw: object = json.load(fh)
    except Exception as exc:  # defensive: file missing, bad JSON, permission error
        extra_undetermined.append(
            UndeterminedField(
                name="config_parse_error",
                note=f"Could not read or parse config.json: {exc}. All fields are undetermined.",
            )
        )
        return SignatureSuggestion(
            suggested=SuggestedFields(
                cameras=[],
                action_dim=None,
                state_dim=None,
                instruction=False,
                normalization_info={},
            ),
            undetermined=list(_ALWAYS_UNDETERMINED) + extra_undetermined,
        )

    # `ty`: the JSON parser returns dict[str, Any]; we narrow to dict[str, object] via cast.
    cfg: dict[str, object] = cast("dict[str, object]", raw) if isinstance(raw, dict) else {}

    cameras, state_dim, instruction = _parse_input_features(cfg.get("input_features"))
    action_dim = _parse_output_features(cfg.get("output_features"))

    normalization_info: dict[str, str] = {}
    norm_raw = cfg.get("normalization_mapping")
    if isinstance(norm_raw, dict):
        normalization_info = {str(k): str(v) for k, v in norm_raw.items()}

    if action_dim is not None:
        extra_undetermined.append(
            UndeterminedField(
                name="action_kind",
                note=(
                    f"action_dim={action_dim} was read, but the checkpoint does not say "
                    "whether this is an end-effector (EE) or joint action space. "
                    "Inspect the model card or training environment to decide."
                ),
            )
        )

    return SignatureSuggestion(
        suggested=SuggestedFields(
            cameras=cameras,
            action_dim=action_dim,
            state_dim=state_dim,
            instruction=instruction,
            normalization_info=normalization_info,
        ),
        undetermined=list(_ALWAYS_UNDETERMINED) + extra_undetermined,
    )


__all__ = [
    "CameraSuggestion",
    "SignatureSuggestion",
    "SuggestedFields",
    "UndeterminedField",
    "from_lerobot_checkpoint",
]
