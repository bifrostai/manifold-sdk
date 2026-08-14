"""The replay log: simulator state a benchmark writes for its replay.

A log is one episode's worth of what a rendered replay needs and no policy reads
- the scene's geometry, a pose per body per step, and whatever images, scalars
and text the benchmark chose to publish. It is written to the results directory
the runner already mounts, never onto the bridge (ADR 0003), and the runner turns
it into a Rerun recording. `replay_log_path` returns the path, beside the
rollup `write_rollup` writes.

The framing is the wire's: `pack_stream_frame` puts a four-byte big-endian length
in front of a msgpack body, and arrays are encoded as the codec's `__ndarray__`
and `__image__` wrappers (ADR 0005). Frames are appended as the episode runs - a
header, then the scene, then one per step - so an episode killed halfway leaves a
log readable to its last complete frame. A frame that is *whole* but will not
decode is a different thing and raises, so a corrupted middle cannot pass for a
short tail.

`REPLAY_LOG_VERSION` is the log's own contract version, separate
from `BRIDGE_PROTOCOL_VERSION`: a wire bump leaves stored logs valid, and a
change here leaves the wire alone.

A log describes itself (ADR 0004). The header declares every channel the steps
carry, with a kind the reader dispatches on and a group the runner renders as one
tab, so nothing downstream holds a kinematic model, a joint-name list or a
camera quirk for the benchmark that wrote it. The header also declares one of
those groups the overview group, which the runner draws beside the scene rather
than in a tab of its own: a benchmark declares which of its scalars are worth
watching while the episode plays, and nothing reading the log could derive it.
Geometry is triangles only: `Mesh.box` and `Mesh.cylinder` tessellate the
primitives a tabletop scene holds beside its meshes, which keeps a single
geometry case in every reader.

Conventions the log fixes, so a reader assumes rather than asks: an orientation
is a quaternion in `xyzw` order, a position is metres, and a depth image is
float16 metres. An image channel's dtype is what separates the two kinds of
frame - uint8 is colour, floating-point is depth - so the writer refuses an array
that would make the distinction a guess, rather than storing one kind as the
other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from manifold.lib.compat import StrEnum
from manifold.wire.bridge import MAX_FRAME_BYTES, pack_stream_frame, read_stream_frame
from manifold.wire.codec import (
    ImageFormat,
    find_encoded_image,
    pack_encoded_image,
    pack_ndarray,
    unpack_ndarray_full,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pathlib import Path
    from types import TracebackType

# The log's own contract version, independent of the bridge protocol's.
REPLAY_LOG_VERSION = 2

# The suffix a log file carries, and the directory it goes in relative to the
# output directory a benchmark is given. The runner globs for both.
REPLAY_LOG_SUFFIX = ".replay"
REPLAY_LOG_DIR = "results"

# Vertices and depth are float16, face indices unsigned 32-bit, poses float32.
# Vertices sit in a mesh's local frame at sub-metre magnitudes and a log carries
# only visual geometry, so nothing collides with what float16 rounds (ADR 0004).
_VERTEX_DTYPE = "<f2"
_FACE_DTYPE = "<u4"
_POSE_DTYPE = "<f4"
_DEPTH_DTYPE = "<f2"

# The largest finite depth float16 holds. A sim's "no hit" sentinel is often a
# large finite far plane, which float16 rounds to `inf`, where nothing distinguishes
# it from a measurement, so the writer refuses it and requires `inf` or a clip
# instead (ADR 0004).
_MAX_DEPTH_METRES = float(np.finfo(np.float16).max)


class ReplayFrameType(StrEnum):
    """The kinds of frame a replay log carries.

    `ReplayLogWriter` writes a header, then a scene, then one frame per step; the
    order is the writer's, not this declaration's.
    """

    HEADER = "header"  # the version and the channels every step declares.
    SCENE = "scene"  # the scene bodies and their geometry, once.
    STEP = "step"  # one per step: a pose per body, plus the declared channels.


class ChannelKind(StrEnum):
    """What a channel's per-step value is, and so how a reader renders it."""

    IMAGE = "image"  # a uint8 colour frame or a float16 depth frame.
    SCALAR = "scalar"  # one number per step.
    TEXT = "text"  # one line per step.


@dataclass(frozen=True)
class Channel:
    """One channel a log's steps carry, declared in the header.

    `group` is the tab a scalar is rendered in; a channel that declares none
    is grouped by the first segment of its name.
    """

    name: str
    kind: ChannelKind
    group: str | None = None


@dataclass(frozen=True, eq=False)
class Mesh:
    """A triangle mesh in its body's frame.

    `vertices` is (n, 3) and `faces` is (m, 3) of indices into it. Frozen
    dataclasses with `eq=False` throughout this module, matching `core.values`:
    these hold numpy arrays, so a generated `__eq__` would return an array.
    """

    vertices: np.ndarray
    faces: np.ndarray

    @classmethod
    def box(cls, half_extents: Sequence[float]) -> Mesh:
        """The 12 triangles of an axis-aligned box, centred on its body's origin.

        A benchmark whose scene holds primitives tessellates them through helpers
        like this one rather than declaring a shape a reader would have to
        special-case.
        """
        x, y, z = (float(v) for v in half_extents)
        vertices = np.array(
            [
                [-x, -y, -z],
                [x, -y, -z],
                [x, y, -z],
                [-x, y, -z],
                [-x, -y, z],
                [x, -y, z],
                [x, y, z],
                [-x, y, z],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.uint32,
        )
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def cylinder(cls, radius: float, half_length: float, *, segments: int = 16) -> Mesh:
        """A z-aligned cylinder as triangles, centred on its body's origin.

        `segments` sets how many sides the ring is drawn with, trading vertices
        for roundness.
        """
        angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        vertices = np.vstack(
            [
                np.column_stack([ring, np.full(segments, -half_length)]),
                np.column_stack([ring, np.full(segments, half_length)]),
                np.array([[0.0, 0.0, -half_length], [0.0, 0.0, half_length]]),
            ]
        ).astype(np.float32)
        bottom_hub, top_hub = 2 * segments, 2 * segments + 1
        faces = []
        for lower in range(segments):
            next_lower = (lower + 1) % segments
            upper, next_upper = segments + lower, segments + next_lower
            faces.append([lower, next_lower, next_upper])
            faces.append([lower, next_upper, upper])
            faces.append([bottom_hub, next_lower, lower])
            faces.append([top_hub, upper, next_upper])
        return cls(vertices=vertices, faces=np.array(faces, dtype=np.uint32))


@dataclass(frozen=True, eq=False)
class ScenePart:
    """One piece of a scene body's geometry, posed in that body's frame.

    A body is several parts because a simulator gives it several visual geoms.
    The part's pose is fixed; the body's pose moves per step.
    """

    mesh: Mesh
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    orientation: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)


@dataclass(frozen=True, eq=False)
class SceneBody:
    """One named rigid body in the scene, and the geometry that moves with it."""

    name: str
    parts: tuple[ScenePart, ...]


@dataclass(frozen=True, eq=False)
class Pose:
    """A body's world pose: metres, and a quaternion in `xyzw` order."""

    position: np.ndarray
    orientation: np.ndarray


@dataclass(frozen=True, eq=False)
class ReplayStep:
    """One step of an episode: where every body is, and what each channel said."""

    index: int
    poses: dict[str, Pose] = field(default_factory=dict)
    images: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    text: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class ReplayLog:
    """A log read back: its episode, its channels, its scene and its steps.

    `overview_group` is the scalar group the header asked to have drawn beside the
    scene. If it is `None`, the overview contains the scene, images and text. Each
    scalar group remains in its own tab.
    """

    version: int
    episode_idx: int
    channels: tuple[Channel, ...]
    bodies: tuple[SceneBody, ...]
    steps: tuple[ReplayStep, ...]
    overview_group: str | None


class ReplayLogWriter:
    """Appends one episode's replay log, frame by frame, as the episode runs.

    The header is written when the writer opens, so a log declares its channels even
    if the episode dies before its first step. Steps are numbered by the writer,
    so a caller cannot misreport the index.

    Use it as a context manager:

        with ReplayLogWriter(path, channels=channels) as log:
            log.write_scene(bodies)
            log.write_step(poses, images={"agentview": rgb}, scalars={"reward": r})
    """

    def __init__(
        self,
        path: Path,
        *,
        episode_idx: int,
        channels: Iterable[Channel],
        image_format: ImageFormat = "jpeg",
        overview_group: str | None = None,
    ) -> None:
        """Open `path` for writing and emit the header.

        `episode_idx` is the episode's index in the run, carried in the header so
        a reader identifies the episode from the log alone, without a filename
        convention. `image_format` applies to uint8 channel images; "jpeg" and
        "png" need the `images` extra, and depth is float16 regardless.

        `overview_group` is the scalar group drawn beside the scene rather than
        in a tab of its own. It must match the group of at least one scalar channel.
        Otherwise the overview would omit the requested plots.

        Raises:
            ValueError: If `overview_group` does not identify a scalar group.
        """
        self._channels = tuple(channels)
        self._kinds = {channel.name: channel.kind for channel in self._channels}
        self._image_format = image_format
        _check_overview_group(overview_group, self._channels)
        self._steps = 0
        self._seq = 0
        self._file = path.open("wb")
        self._write(
            ReplayFrameType.HEADER,
            {
                "version": REPLAY_LOG_VERSION,
                "episode_idx": episode_idx,
                "overview_group": overview_group,
                "channels": [
                    {"name": c.name, "kind": str(c.kind), "group": c.group} for c in self._channels
                ],
            },
        )

    def __enter__(self) -> ReplayLogWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def write_scene(self, bodies: Sequence[SceneBody]) -> None:
        """Write the scene frame: every body, its parts and their geometry.

        Raises:
            ValueError: If two bodies share a name, since a step's poses are keyed
                by it and one would shadow the other.
        """
        names = [body.name for body in bodies]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if len(duplicates) > 0:
            raise ValueError(f"scene bodies share a name: {duplicates}")
        self._write(
            ReplayFrameType.SCENE,
            {
                "bodies": [
                    {
                        "name": body.name,
                        "parts": [_pack_part(part) for part in body.parts],
                    }
                    for body in bodies
                ]
            },
        )

    def write_step(
        self,
        poses: Mapping[str, Pose],
        *,
        images: Mapping[str, np.ndarray] | None = None,
        scalars: Mapping[str, float] | None = None,
        text: Mapping[str, str] | None = None,
    ) -> None:
        """Write one step frame, numbering it after the steps already written.

        Raises:
            ValueError: If a value is given for a name the header did not declare,
                or for a name it declared as another kind, or if an image is
                neither a colour nor a depth frame.
        """
        self._check_declared(images, ChannelKind.IMAGE)
        self._check_declared(scalars, ChannelKind.SCALAR)
        self._check_declared(text, ChannelKind.TEXT)
        payload: dict[str, Any] = {
            "index": self._steps,
            "poses": {name: _pack_pose(pose) for name, pose in poses.items()},
            "images": {
                name: _pack_image(array, self._image_format, name=name)
                for name, array in (images or {}).items()
            },
            "scalars": {name: float(value) for name, value in (scalars or {}).items()},
            "text": dict(text or {}),
        }
        self._write(ReplayFrameType.STEP, payload)
        self._steps += 1

    def close(self) -> None:
        """Close the file. Writing after this raises."""
        self._file.close()

    def _write(self, frame_type: ReplayFrameType, payload: dict[str, Any]) -> None:
        """Frame a payload and append it, refusing one larger than any reader accepts.

        `read_stream_frame` rejects a frame declaring more than `MAX_FRAME_BYTES`
        before it reads the body, and it raises rather than stopping, so an
        oversized frame invalidates the whole log rather than its own tail. Refusing
        it here leaves the log readable and reports which frame was too big.
        """
        frame = pack_stream_frame(str(frame_type), payload, seq=self._seq)
        if len(frame) > MAX_FRAME_BYTES:
            raise ValueError(
                f"a {frame_type} frame of {len(frame)} bytes exceeds the {MAX_FRAME_BYTES} a "
                "reader accepts: decimate the scene's geometry or split the channel"
            )
        self._file.write(frame)
        self._file.flush()
        self._seq += 1

    def _check_declared(self, values: Mapping[str, Any] | None, kind: ChannelKind) -> None:
        for name in values or {}:
            declared = self._kinds.get(name)
            if declared is None:
                raise ValueError(f"channel {name!r} is not declared in the header")
            if declared != kind:
                raise ValueError(f"channel {name!r} is declared {declared}, not {kind}")


def replay_log_path(output_dir: Path, episode_idx: int) -> Path:
    """Where episode `episode_idx`'s log goes under a benchmark's output directory.

    Beside the rollup `write_rollup` writes, in `<output-dir>/results/`: that is
    the one directory the runner scans, so a log written anywhere else is a log
    nothing collects. The parent is created.
    """
    directory = output_dir / REPLAY_LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"episode-{episode_idx:05d}{REPLAY_LOG_SUFFIX}"


def read_replay_log(path: Path) -> ReplayLog:
    """Read a log back, stopping at the first incomplete frame.

    A truncated log is not an error: the reader returns the steps that were fully
    written, which is what an episode killed mid-run leaves behind. A frame that
    is complete but malformed is an error, since that is corruption rather than a
    short tail.

    Raises:
        ValueError: If the first frame is not a header, the version is one this
            reader does not accept, or a complete frame is malformed.
    """
    with path.open("rb") as handle:
        frames = _iter_frames(handle)
        header = next(frames, None)
        if header is None or header.get("type") != ReplayFrameType.HEADER:
            raise ValueError("a replay log opens with a header frame")
        version, episode_idx, overview_group, channels = _read_header(header.get("payload", {}))
        bodies: tuple[SceneBody, ...] = ()
        steps: list[ReplayStep] = []
        for frame in frames:
            kind = frame.get("type")
            payload = frame.get("payload", {})
            if kind == ReplayFrameType.SCENE:
                bodies = _read_scene(payload)
            elif kind == ReplayFrameType.STEP:
                steps.append(_read_step(payload))
    return ReplayLog(
        version=version,
        episode_idx=episode_idx,
        channels=channels,
        bodies=bodies,
        steps=tuple(steps),
        overview_group=overview_group,
    )


def _pack_part(part: ScenePart) -> dict[str, Any]:
    return {
        "vertices": _pack_array(part.mesh.vertices, _VERTEX_DTYPE),
        "faces": _pack_array(part.mesh.faces, _FACE_DTYPE),
        "position": _pack_array(part.position, _POSE_DTYPE),
        "orientation": _pack_array(part.orientation, _POSE_DTYPE),
        "color": [float(channel) for channel in part.color],
    }


def _pack_pose(pose: Pose) -> dict[str, Any]:
    return {
        "position": _pack_array(pose.position, _POSE_DTYPE),
        "orientation": _pack_array(pose.orientation, _POSE_DTYPE),
    }


def _pack_image(array: np.ndarray, image_format: ImageFormat, *, name: str) -> dict[str, Any]:
    """Pack a channel image: colour as an encoded image, depth as float16 metres.

    The dtype chooses: uint8 is a colour frame, floating-point is depth in metres.
    Anything else, and any float array shaped like colour, is refused rather than
    stored as the other kind - a log records no per-channel image kind, so a
    reader has only the dtype to go on.

    Raises:
        ValueError: If the array is neither a colour nor a depth frame, or if its
            shape or range is one the chosen encoding cannot hold.
        ImportError: If "jpeg" or "png" is requested but Pillow is not installed.
    """
    if array.dtype == np.uint8:
        return _pack_colour_image(array, image_format, name=name)
    if array.dtype.kind != "f":
        raise ValueError(
            f"image channel {name!r} has dtype {array.dtype}: a colour frame is uint8 "
            "and a depth frame is floating-point metres"
        )
    if array.ndim == 3 and array.shape[2] in (3, 4):
        raise ValueError(
            f"image channel {name!r} is floating-point with shape {list(array.shape)}: a "
            "depth frame has one channel, so convert a colour frame to uint8"
        )
    return _pack_array(_depth_metres(array, name), _DEPTH_DTYPE)


def _pack_colour_image(
    array: np.ndarray, image_format: ImageFormat, *, name: str
) -> dict[str, Any]:
    """Encode a uint8 colour frame, validating the shape the format can hold."""
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (3, 4)):
        raise ValueError(
            f"colour image {name!r} has shape {list(array.shape)}: expected (h, w), "
            "(h, w, 1), (h, w, 3) or (h, w, 4)"
        )
    if image_format == "raw":
        return pack_encoded_image(array.tobytes(), format_="raw", shape=list(array.shape))
    if image_format in ("jpeg", "png"):
        if image_format == "jpeg" and array.ndim == 3 and array.shape[2] == 4:
            raise ValueError(
                f"colour image {name!r} carries an alpha channel, which JPEG cannot hold: "
                "use image_format='png' or drop the alpha channel"
            )
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on the images extra
            raise ImportError(
                f"image_format={image_format!r} needs Pillow: install manifold-sdk[images]"
            ) from exc
        import io

        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format=image_format.upper())
        return pack_encoded_image(buffer.getvalue(), format_=image_format, shape=list(array.shape))
    raise ValueError(f"unsupported image_format {image_format!r}: use 'raw', 'jpeg' or 'png'")


def _depth_metres(array: np.ndarray, name: str) -> np.ndarray:
    """Check a depth frame against what float16 holds, and return it.

    `inf` passes: it is the sentinel a reader can act on. A large finite far
    plane does not, because casting it silently yields `inf` anyway and only the
    benchmark defines what its own sentinel means.
    """
    finite = np.isfinite(array)
    if bool(finite.any()) and float(np.abs(array[finite]).max()) > _MAX_DEPTH_METRES:
        raise ValueError(
            f"depth image {name!r} holds a value beyond {_MAX_DEPTH_METRES} m, which float16 "
            "cannot carry: use `inf` for no hit, or clip to the sensor's range"
        )
    return array


def _check_overview_group(group: str | None, channels: tuple[Channel, ...]) -> None:
    """Check that the overview group identifies a scalar group.

    A typo or a removed group would omit requested plots from the overview. Check
    the name when the writer receives it, before rendering.

    Raises:
        ValueError: If `group` does not identify a scalar group.
    """
    if group is None:
        return
    groups = {_group_of(c) for c in channels if c.kind is ChannelKind.SCALAR}
    if group not in groups:
        known = ", ".join(sorted(groups)) if len(groups) > 0 else "none"
        raise ValueError(
            f"overview group {group!r} does not identify a scalar group: the groups are {known}"
        )


def _group_of(channel: Channel) -> str:
    """The group a scalar channel is plotted in, declared or fallen back to."""
    return channel.group or channel.name.split("/")[0]


def _pack_array(array: np.ndarray, dtype: str) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return pack_ndarray(
        contiguous.reshape(-1).astype(dtype),
        dtype=dtype,
        shape=list(contiguous.shape),
    )


def _iter_frames(handle: Any) -> Iterator[dict[str, Any]]:
    """Yield each frame in the stream, ending where the written bytes end.

    A short read is the end: either the file's clean end, or the partial frame an
    episode killed mid-write left. A frame whose bytes were all there and whose
    body still would not decode is corruption, and raises - otherwise a flipped
    byte mid-log would silently discard every frame after it.

    Raises:
        ValueError: If a complete frame's body is malformed.
    """
    short_read = False

    def read_exactly(count: int) -> bytes | None:
        nonlocal short_read
        data = handle.read(count)
        if len(data) != count:
            short_read = True
            return None
        return data

    while True:
        frame = read_stream_frame(read_exactly)
        if frame is None:
            if short_read:
                return
            raise ValueError("a complete replay log frame is malformed")
        yield frame


def _read_header(payload: dict[str, Any]) -> tuple[int, int, str | None, tuple[Channel, ...]]:
    version = payload.get("version")
    if version != REPLAY_LOG_VERSION:
        raise ValueError(f"unsupported replay log version {version!r}")
    episode_idx = payload.get("episode_idx")
    if not isinstance(episode_idx, int):
        raise ValueError("header 'episode_idx' must be an int")  # noqa: TRY004
    overview = payload.get("overview_group")
    overview_group = overview if isinstance(overview, str) else None
    raw_channels = payload.get("channels")
    if not isinstance(raw_channels, list):
        raise ValueError("header 'channels' must be a list")  # noqa: TRY004
    channels = []
    for node in raw_channels:
        if not isinstance(node, dict):
            raise ValueError("a header channel must be a dict")  # noqa: TRY004
        name = node.get("name")
        kind = node.get("kind")
        group = node.get("group")
        if not isinstance(name, str) or kind not in set(ChannelKind):
            raise ValueError(f"malformed header channel {node!r}")
        channels.append(
            Channel(
                name=name,
                kind=ChannelKind(kind),
                group=group if isinstance(group, str) else None,
            )
        )
    return version, episode_idx, overview_group, tuple(channels)


def _read_scene(payload: dict[str, Any]) -> tuple[SceneBody, ...]:
    raw_bodies = payload.get("bodies")
    if not isinstance(raw_bodies, list):
        raise ValueError("scene 'bodies' must be a list")  # noqa: TRY004
    bodies = []
    for node in raw_bodies:
        if not isinstance(node, dict) or not isinstance(node.get("name"), str):
            raise ValueError(f"malformed scene body {node!r}")  # noqa: TRY004
        raw_parts = node.get("parts")
        if not isinstance(raw_parts, list):
            raise ValueError("a scene body's 'parts' must be a list")  # noqa: TRY004
        bodies.append(
            SceneBody(name=node["name"], parts=tuple(_read_part(part) for part in raw_parts))
        )
    return tuple(bodies)


def _read_part(node: Any) -> ScenePart:
    if not isinstance(node, dict):
        raise ValueError("a scene part must be a dict")  # noqa: TRY004
    vertices = _decoded_array(node.get("vertices"), "part vertices")
    faces = _decoded_array(node.get("faces"), "part faces")
    color = node.get("color")
    if not isinstance(color, list) or len(color) != 4:
        raise ValueError("a scene part's 'color' must be four numbers")
    return ScenePart(
        mesh=Mesh(vertices=vertices, faces=faces),
        position=_decoded_array(node.get("position"), "part position"),
        orientation=_decoded_array(node.get("orientation"), "part orientation"),
        color=(float(color[0]), float(color[1]), float(color[2]), float(color[3])),
    )


def _read_step(payload: dict[str, Any]) -> ReplayStep:
    index = payload.get("index")
    if not isinstance(index, int):
        raise ValueError("step 'index' must be an int")  # noqa: TRY004
    raw_poses = payload.get("poses", {})
    raw_images = payload.get("images", {})
    raw_scalars = payload.get("scalars", {})
    raw_text = payload.get("text", {})
    if not all(isinstance(node, dict) for node in (raw_poses, raw_images, raw_scalars, raw_text)):
        raise ValueError("a step's poses, images, scalars and text must be dicts")
    return ReplayStep(
        index=index,
        poses={name: _read_pose(node) for name, node in raw_poses.items()},
        images={name: _unpack_image(node, name) for name, node in raw_images.items()},
        scalars={name: float(value) for name, value in raw_scalars.items()},
        text={name: str(value) for name, value in raw_text.items()},
    )


def _read_pose(node: Any) -> Pose:
    if not isinstance(node, dict):
        raise ValueError("a step pose must be a dict")  # noqa: TRY004
    return Pose(
        position=_decoded_array(node.get("position"), "pose position"),
        orientation=_decoded_array(node.get("orientation"), "pose orientation"),
    )


def _unpack_image(node: Any, name: str) -> np.ndarray:
    """Decode a channel image back to an array, colour or depth.

    The array is writable, matching `wire.bridge`: a decode returns a value the
    caller owns, not a view over the frame's bytes.

    Raises:
        ValueError: If the node is neither a decodable image nor a decodable array.
        ImportError: If the image is JPEG- or PNG-encoded and Pillow is missing.
    """
    encoded = find_encoded_image(node)
    if encoded is None:
        return _decoded_array(node, f"image {name!r}")
    data, image_format = encoded
    shape = node.get("shape") if isinstance(node, dict) else None
    if image_format == "raw":
        if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
            raise ValueError(f"raw image {name!r} does not declare a shape")
        expected = int(np.prod(shape)) if len(shape) > 0 else 0
        if len(data) != expected:
            raise ValueError(
                f"raw image {name!r} has {len(data)} bytes but shape {shape} needs {expected}"
            )
        return np.array(np.frombuffer(data, dtype=np.uint8), dtype=np.uint8).reshape(shape)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the images extra
        raise ImportError(
            f"image {name!r} is {image_format}-encoded: install manifold-sdk[images]"
        ) from exc
    import io

    try:
        decoded = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"image {name!r} is not decodable {image_format} data") from exc
    return np.array(decoded)


def _decoded_array(node: Any, what: str) -> np.ndarray:
    array = unpack_ndarray_full(node)
    if array is None:
        raise ValueError(f"{what} is not a decodable ndarray wrapper")
    return array


__all__ = [
    "REPLAY_LOG_DIR",
    "REPLAY_LOG_SUFFIX",
    "REPLAY_LOG_VERSION",
    "Channel",
    "ChannelKind",
    "Mesh",
    "Pose",
    "ReplayFrameType",
    "ReplayLog",
    "ReplayLogWriter",
    "ReplayStep",
    "SceneBody",
    "ScenePart",
    "read_replay_log",
    "replay_log_path",
]
