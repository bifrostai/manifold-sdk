"""Tests for the replay log: round-trips, truncation, and the writer's gates."""

from __future__ import annotations

import io

import numpy as np
import pytest

from manifold.replay import (
    REPLAY_LOG_VERSION,
    CameraPinhole,
    Channel,
    ChannelKind,
    Mesh,
    Pose,
    ReplayFrameType,
    ReplayLogWriter,
    SceneBody,
    ScenePart,
    read_replay_log,
    replay_log_path,
)
from manifold.wire.bridge import pack_stream_frame

CHANNELS = (
    Channel(name="agentview", kind=ChannelKind.IMAGE),
    Channel(name="agentview_depth", kind=ChannelKind.IMAGE),
    Channel(name="reward", kind=ChannelKind.SCALAR, group="Task"),
    Channel(name="instruction", kind=ChannelKind.TEXT),
)


def _bodies() -> tuple[SceneBody, ...]:
    return (
        SceneBody(
            name="table",
            parts=(ScenePart(mesh=Mesh.box([0.4, 0.6, 0.02]), color=(0.5, 0.4, 0.3, 1.0)),),
        ),
        SceneBody(
            name="robot0_link0",
            parts=(
                ScenePart(
                    mesh=Mesh(
                        vertices=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
                        faces=np.array([[0, 1, 2]]),
                    ),
                    position=np.array([0.0, 0.0, 0.05]),
                ),
            ),
        ),
    )


def _pose(x: float) -> Pose:
    return Pose(position=np.array([x, 0.0, 0.1]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))


def test_a_log_round_trips_its_header_scene_and_steps(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=7, channels=CHANNELS, image_format="raw") as log:
        log.write_scene(_bodies())
        for step in range(3):
            log.write_step(
                {"table": _pose(0.0), "robot0_link0": _pose(step * 0.01)},
                images={
                    "agentview": np.full((4, 4, 3), step, dtype=np.uint8),
                    "agentview_depth": np.full((4, 4), 1.5, dtype=np.float32),
                },
                scalars={"reward": float(step)},
                text={"instruction": "pick up the mug"},
            )

    read = read_replay_log(path)
    assert read.version == REPLAY_LOG_VERSION
    assert read.episode_idx == 7
    assert read.channels == CHANNELS
    assert [body.name for body in read.bodies] == ["table", "robot0_link0"]
    assert [step.index for step in read.steps] == [0, 1, 2]

    last = read.steps[-1]
    assert last.scalars == {"reward": 2.0}
    assert last.text == {"instruction": "pick up the mug"}
    assert np.array_equal(last.images["agentview"], np.full((4, 4, 3), 2, dtype=np.uint8))
    np.testing.assert_allclose(last.images["agentview_depth"], np.full((4, 4), 1.5))
    np.testing.assert_allclose(last.poses["robot0_link0"].position, [0.02, 0.0, 0.1], atol=1e-6)


def test_an_overview_group_round_trips(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(
        path, episode_idx=0, channels=CHANNELS, image_format="raw", overview_group="Task"
    ) as log:
        log.write_scene(())

    assert read_replay_log(path).overview_group == "Task"


def test_a_log_without_an_overview_group_reads_back_without_one(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="raw") as log:
        log.write_scene(())

    assert read_replay_log(path).overview_group is None


def test_the_writer_rejects_an_overview_group_without_a_scalar_channel(tmp_path):
    with pytest.raises(ValueError, match="does not identify a scalar group"):
        ReplayLogWriter(
            tmp_path / "episode.replay",
            episode_idx=0,
            channels=CHANNELS,
            image_format="raw",
            overview_group="Robot",
        )


def test_an_overview_group_may_use_the_default_channel_group(tmp_path):
    """`Channel.group` defaults to the first segment of the channel name.

    An overview group may use that default.
    """
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(
        path,
        episode_idx=0,
        channels=(Channel(name="joints/0", kind=ChannelKind.SCALAR),),
        image_format="raw",
        overview_group="joints",
    ) as log:
        log.write_scene(())

    assert read_replay_log(path).overview_group == "joints"


def test_a_channel_row_round_trips(tmp_path):
    path = tmp_path / "episode.replay"
    channels = (
        Channel(name="gripper", kind=ChannelKind.SCALAR, group="Action", row=0),
        Channel(name="ee_pos_x", kind=ChannelKind.SCALAR, group="Action", row=1),
    )
    with ReplayLogWriter(path, episode_idx=0, channels=channels, image_format="raw") as log:
        log.write_scene(())

    assert [c.row for c in read_replay_log(path).channels] == [0, 1]


def test_a_channel_without_a_row_reads_back_without_one(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="raw") as log:
        log.write_scene(())

    assert all(channel.row is None for channel in read_replay_log(path).channels)


def test_the_writer_rejects_partially_numbered_channel_rows(tmp_path):
    with pytest.raises(ValueError, match="mix numbered and unnumbered channel rows: Action"):
        ReplayLogWriter(
            tmp_path / "episode.replay",
            episode_idx=0,
            channels=(
                Channel(name="gripper", kind=ChannelKind.SCALAR, group="Action", row=0),
                Channel(name="ee_pos_x", kind=ChannelKind.SCALAR, group="Action"),
            ),
            image_format="raw",
        )


def test_one_group_may_number_its_rows_while_another_does_not(tmp_path):
    """The check is per group, so a benchmark may number the rows of one group and
    leave the rest to the reader."""
    path = tmp_path / "episode.replay"
    channels = (
        Channel(name="gripper", kind=ChannelKind.SCALAR, group="Action", row=0),
        Channel(name="success", kind=ChannelKind.SCALAR, group="Task"),
    )
    with ReplayLogWriter(path, episode_idx=0, channels=channels, image_format="raw") as log:
        log.write_scene(())

    assert [c.row for c in read_replay_log(path).channels] == [0, None]


def test_a_negative_row_is_refused(tmp_path):
    with pytest.raises(ValueError, match="negative row"):
        ReplayLogWriter(
            tmp_path / "episode.replay",
            episode_idx=0,
            channels=(Channel(name="gripper", kind=ChannelKind.SCALAR, group="Action", row=-1),),
            image_format="raw",
        )


def test_a_boxs_geometry_survives_the_float16_vertices(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=(), image_format="raw") as log:
        log.write_scene(_bodies())

    part = read_replay_log(path).bodies[0].parts[0]
    assert part.mesh.faces.shape == (12, 3)
    np.testing.assert_allclose(part.mesh.vertices.max(axis=0), [0.4, 0.6, 0.02], atol=1e-3)
    assert part.color == (0.5, 0.4, 0.3, 1.0)


def test_a_tessellated_cylinder_is_closed_and_bounded():
    mesh = Mesh.cylinder(radius=0.05, half_length=0.2, segments=8)

    assert mesh.faces.shape == (32, 3)
    assert mesh.faces.max() == len(mesh.vertices) - 1
    np.testing.assert_allclose(mesh.vertices[:, 2].min(), -0.2)
    np.testing.assert_allclose(np.abs(mesh.vertices[:, :2]).max(), 0.05)


def test_a_truncated_log_reads_up_to_its_last_whole_frame(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=7, channels=CHANNELS, image_format="raw") as log:
        log.write_scene(_bodies())
        log.write_step({"table": _pose(0.0)}, scalars={"reward": 1.0})
        log.write_step({"table": _pose(0.0)}, scalars={"reward": 2.0})
    whole = path.read_bytes()

    path.write_bytes(whole[:-12])
    read = read_replay_log(path)
    assert [step.scalars["reward"] for step in read.steps] == [1.0]
    assert len(read.bodies) == 2


def test_a_log_with_no_steps_still_declares_its_channels(tmp_path):
    path = tmp_path / "episode.replay"
    ReplayLogWriter(path, episode_idx=3, channels=CHANNELS).close()

    read = read_replay_log(path)
    assert read.episode_idx == 3
    assert read.channels == CHANNELS
    assert read.steps == ()
    assert read.bodies == ()


def test_an_undeclared_channel_is_refused(tmp_path):
    path = tmp_path / "episode.replay"
    with (
        ReplayLogWriter(path, episode_idx=3, channels=CHANNELS) as log,
        pytest.raises(ValueError, match="not declared"),
    ):
        log.write_step({}, scalars={"grasp": 1.0})


def test_a_channel_written_as_the_wrong_kind_is_refused(tmp_path):
    path = tmp_path / "episode.replay"
    with (
        ReplayLogWriter(path, episode_idx=3, channels=CHANNELS) as log,
        pytest.raises(ValueError, match="declared"),
    ):
        log.write_step({}, scalars={"instruction": 1.0})


def test_a_log_that_does_not_open_with_a_header_is_refused(tmp_path):
    path = tmp_path / "episode.replay"
    path.write_bytes(pack_stream_frame(str(ReplayFrameType.STEP), {"index": 0}, seq=0))

    with pytest.raises(ValueError, match="opens with a header"):
        read_replay_log(path)


def test_a_first_frame_that_is_not_a_frame_at_all_is_refused(tmp_path):
    path = tmp_path / "episode.replay"
    path.write_bytes(b"\x00\x00\x00\x04junk")

    with pytest.raises(ValueError, match="malformed"):
        read_replay_log(path)


def test_a_jpeg_channel_image_round_trips(tmp_path):
    pytest.importorskip("PIL")
    path = tmp_path / "episode.replay"
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="jpeg") as log:
        log.write_step({}, images={"agentview": frame})

    decoded = read_replay_log(path).steps[0].images["agentview"]
    assert decoded.shape == (8, 8, 3)
    assert decoded[0, 0, 0] > 200


def test_a_corrupt_whole_frame_is_not_read_as_a_short_tail(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=7, channels=CHANNELS, image_format="raw") as log:
        log.write_scene(_bodies())
        log.write_step({"table": _pose(0.0)}, scalars={"reward": 1.0})
        log.write_step({"table": _pose(0.0)}, scalars={"reward": 2.0})
    whole = bytearray(path.read_bytes())

    # Break a byte inside the last frame's body, leaving its length prefix intact,
    # so every byte the frame declared is present and only the msgpack is wrong.
    whole[-6] ^= 0xFF
    path.write_bytes(bytes(whole))

    with pytest.raises(ValueError, match="complete replay log frame is malformed"):
        read_replay_log(path)


def test_an_image_that_is_neither_colour_nor_depth_is_refused(tmp_path):
    path = tmp_path / "episode.replay"
    with (
        ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="raw") as log,
        pytest.raises(ValueError, match="a colour frame is uint8"),
    ):
        log.write_step({}, images={"agentview": np.zeros((4, 4), dtype=np.uint16)})


def test_a_floating_point_colour_frame_is_refused(tmp_path):
    path = tmp_path / "episode.replay"
    with (
        ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="raw") as log,
        pytest.raises(ValueError, match="convert a colour frame to uint8"),
    ):
        log.write_step({}, images={"agentview": np.zeros((4, 4, 3), dtype=np.float32)})


def test_a_depth_value_float16_cannot_hold_is_refused(tmp_path):
    path = tmp_path / "episode.replay"
    with (
        ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="raw") as log,
        pytest.raises(ValueError, match="float16"),
    ):
        log.write_step({}, images={"agentview_depth": np.full((2, 2), 1e5, dtype=np.float32)})


def test_an_infinite_depth_is_the_sentinel_that_survives(tmp_path):
    path = tmp_path / "episode.replay"
    frame = np.array([[1.5, np.inf], [0.25, np.inf]], dtype=np.float32)
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="raw") as log:
        log.write_step({}, images={"agentview_depth": frame})

    decoded = read_replay_log(path).steps[0].images["agentview_depth"]
    assert np.isinf(decoded[0, 1])
    np.testing.assert_allclose(decoded[0, 0], 1.5)


def test_a_greyscale_colour_frame_drops_its_trailing_axis(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="raw") as log:
        log.write_step({}, images={"agentview": np.full((4, 5, 1), 200, dtype=np.uint8)})

    decoded = read_replay_log(path).steps[0].images["agentview"]
    assert decoded.shape == (4, 5)
    assert decoded[0, 0] == 200


def test_an_alpha_channel_is_refused_for_jpeg(tmp_path):
    pytest.importorskip("PIL")
    path = tmp_path / "episode.replay"
    with (
        ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="jpeg") as log,
        pytest.raises(ValueError, match="alpha channel"),
    ):
        log.write_step({}, images={"agentview": np.zeros((4, 4, 4), dtype=np.uint8)})


def test_a_png_channel_image_round_trips(tmp_path):
    pytest.importorskip("PIL")
    path = tmp_path / "episode.replay"
    frame = np.zeros((8, 8, 4), dtype=np.uint8)
    frame[:, :, 1] = 128
    frame[:, :, 3] = 255
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="png") as log:
        log.write_step({}, images={"agentview": frame})

    decoded = read_replay_log(path).steps[0].images["agentview"]
    assert decoded.shape == (8, 8, 4)
    assert decoded[0, 0, 1] == 128


@pytest.mark.parametrize(("image_format", "shape"), [("jpeg", (8, 8, 3)), ("png", (8, 8, 4))])
def test_a_compressed_frame_reads_back_as_the_bytes_that_were_stored(tmp_path, image_format, shape):
    pillow = pytest.importorskip("PIL.Image")
    path = tmp_path / "episode.replay"
    frame = np.zeros(shape, dtype=np.uint8)
    frame[:, :, 1] = 128
    frame[:, :, -1] = 255
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format=image_format) as log:
        log.write_step({}, images={"agentview": frame})

    stored = read_replay_log(path).steps[0].encoded_images["agentview"]
    buffer = io.BytesIO()
    pillow.fromarray(frame).save(buffer, format=image_format.upper())
    # The writer's own encode, handed back byte for byte: whatever stores the
    # frame downstream stores these rather than re-encoding the decoded pixels.
    assert stored.format == image_format
    assert stored.data == buffer.getvalue()


def test_a_raw_frame_has_no_stored_bytes_to_pass_on(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="raw") as log:
        log.write_step(
            {},
            images={
                "agentview": np.full((4, 4, 3), 200, dtype=np.uint8),
                "agentview_depth": np.full((4, 4), 1.5, dtype=np.float32),
            },
        )

    step = read_replay_log(path).steps[0]
    # A raw frame stored its pixels and a depth frame its float16 array, so
    # neither offers bytes in place of what `images` decoded.
    assert step.encoded_images == {}
    assert step.images["agentview"].shape == (4, 4, 3)
    np.testing.assert_allclose(step.images["agentview_depth"], 1.5)


def test_a_compressed_log_still_decodes_its_depth_and_its_colour(tmp_path):
    pytest.importorskip("PIL")
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="jpeg") as log:
        log.write_step(
            {},
            images={
                "agentview": np.full((8, 8, 3), 120, dtype=np.uint8),
                "agentview_depth": np.full((8, 8), 2.0, dtype=np.float32),
            },
        )

    step = read_replay_log(path).steps[0]
    assert step.images["agentview"].shape == (8, 8, 3)
    np.testing.assert_allclose(step.images["agentview_depth"], 2.0)
    assert set(step.encoded_images) == {"agentview"}


@pytest.mark.parametrize("image_format", ["raw", "jpeg"])
def test_a_decoded_colour_frame_is_the_callers_to_mutate(tmp_path, image_format):
    pytest.importorskip("PIL")
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format=image_format) as log:
        log.write_step({}, images={"agentview": np.zeros((4, 4, 3), dtype=np.uint8)})

    decoded = read_replay_log(path).steps[0].images["agentview"]
    decoded[0, 0, 0] = 7
    assert decoded[0, 0, 0] == 7


def test_two_scene_bodies_may_not_share_a_name(tmp_path):
    path = tmp_path / "episode.replay"
    body = SceneBody(name="table", parts=(ScenePart(mesh=Mesh.box([0.1, 0.1, 0.1])),))
    with (
        ReplayLogWriter(path, episode_idx=0, channels=CHANNELS) as log,
        pytest.raises(ValueError, match="share a name"),
    ):
        log.write_scene((body, body))


def test_a_frame_no_reader_would_accept_is_refused_at_write(tmp_path):
    path = tmp_path / "episode.replay"
    huge = SceneBody(
        name="huge",
        parts=(
            ScenePart(
                mesh=Mesh(
                    vertices=np.zeros((12_000_000, 3), dtype=np.float32),
                    faces=np.zeros((1, 3), dtype=np.uint32),
                ),
            ),
        ),
    )
    with (
        ReplayLogWriter(path, episode_idx=0, channels=CHANNELS) as log,
        pytest.raises(ValueError, match="exceeds"),
    ):
        log.write_scene((huge,))

    # The refused frame was never appended, so the header still reads.
    assert read_replay_log(path).bodies == ()


def test_a_log_goes_beside_the_rollup(tmp_path):
    path = replay_log_path(tmp_path, 4)

    assert path.parent == tmp_path / "results"
    assert path.name == "episode-00004.replay"
    assert path.parent.is_dir()


# --- camera calibration and extrinsics (ADR 0008) -----------------------------

_CAMERAS = (
    CameraPinhole(name="agentview", fx=101.4, fy=101.4, cx=128.0, cy=128.0, width=256, height=256),
    CameraPinhole(name="wrist", fx=101.4, fy=101.4, cx=128.0, cy=128.0, width=256, height=256),
)


def _extrinsic(z: float) -> Pose:
    return Pose(position=np.array([0.5, 0.0, z]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))


def test_header_camera_intrinsics_round_trip(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(
        path, episode_idx=0, channels=CHANNELS, image_format="raw", cameras=_CAMERAS
    ) as log:
        log.write_step({"robot0_link0": _pose(0.0)})
    log = read_replay_log(path)
    assert [c.name for c in log.cameras] == ["agentview", "wrist"]
    assert log.cameras[0].fx == pytest.approx(101.4)
    assert (log.cameras[0].width, log.cameras[0].height) == (256, 256)


def test_a_log_without_cameras_reads_back_without_any(tmp_path):
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="raw") as log:
        log.write_step({"robot0_link0": _pose(0.0)})
    assert read_replay_log(path).cameras == ()


def test_every_step_carries_every_cameras_extrinsic(tmp_path):
    """A pose written once is filled forward, so no step is missing a camera."""
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="raw") as log:
        for i in range(4):
            log.write_step(
                {"robot0_link0": _pose(0.0)},
                # The scene camera never moves; the wrist camera moves every step.
                extrinsics={"agentview": _extrinsic(1.6), "wrist": _extrinsic(1.0 + 0.1 * i)},
            )
    steps = read_replay_log(path).steps
    assert len(steps) == 4
    for i, step in enumerate(steps):
        assert sorted(step.extrinsics) == ["agentview", "wrist"]
        assert step.extrinsics["agentview"].position[2] == pytest.approx(1.6)
        assert step.extrinsics["wrist"].position[2] == pytest.approx(1.0 + 0.1 * i, abs=1e-6)


def test_an_unmoved_camera_is_stored_once(tmp_path):
    """The saving this encoding exists for: a static camera stores one row, not one per step."""
    still = tmp_path / "still.replay"
    with ReplayLogWriter(still, episode_idx=0, channels=CHANNELS, image_format="raw") as log:
        for _ in range(50):
            log.write_step({"robot0_link0": _pose(0.0)}, extrinsics={"agentview": _extrinsic(1.6)})
    moving = tmp_path / "moving.replay"
    with ReplayLogWriter(moving, episode_idx=0, channels=CHANNELS, image_format="raw") as log:
        for i in range(50):
            log.write_step(
                {"robot0_link0": _pose(0.0)}, extrinsics={"agentview": _extrinsic(1.6 + 0.01 * i)}
            )
    # Both read back with a pose on all 50 steps, but only one stored all 50.
    assert all(s.extrinsics["agentview"] is not None for s in read_replay_log(still).steps)
    assert still.stat().st_size < moving.stat().st_size


def test_a_step_with_no_camera_movement_omits_the_key(tmp_path):
    """A publisher that passes no extrinsics writes what it wrote before the field existed."""
    without = tmp_path / "without.replay"
    with ReplayLogWriter(without, episode_idx=0, channels=CHANNELS, image_format="raw") as log:
        log.write_step({"robot0_link0": _pose(0.0)})
    assert read_replay_log(without).steps[0].extrinsics == {}


def test_a_change_float32_cannot_hold_is_not_written(tmp_path):
    """Quantised before comparison: a difference the log cannot store is not a change."""
    path = tmp_path / "episode.replay"
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="raw") as log:
        log.write_step({"robot0_link0": _pose(0.0)}, extrinsics={"agentview": _extrinsic(1.6)})
        first = path.stat().st_size
        # Far below float32's resolution at 1.6, so it rounds to the stored value.
        log.write_step(
            {"robot0_link0": _pose(0.0)}, extrinsics={"agentview": _extrinsic(1.6 + 1e-12)}
        )
        second = path.stat().st_size
    grew = second - first
    with ReplayLogWriter(path, episode_idx=0, channels=CHANNELS, image_format="raw") as log:
        log.write_step({"robot0_link0": _pose(0.0)}, extrinsics={"agentview": _extrinsic(1.6)})
        base = path.stat().st_size
        log.write_step({"robot0_link0": _pose(0.0)}, extrinsics={"agentview": _extrinsic(1.7)})
        real = path.stat().st_size - base
    assert grew < real


def test_a_malformed_header_camera_is_refused(tmp_path):
    """A camera missing its focal length is corruption, not an absent field."""
    path = tmp_path / "episode.replay"
    path.write_bytes(
        pack_stream_frame(
            str(ReplayFrameType.HEADER),
            {
                "version": REPLAY_LOG_VERSION,
                "episode_idx": 0,
                "overview_group": None,
                "channels": [],
                "cameras": [{"name": "agentview"}],
            },
            seq=0,
        )
    )
    with pytest.raises(ValueError, match="malformed header camera"):
        read_replay_log(path)


def test_a_header_camera_list_that_is_not_a_list_is_refused(tmp_path):
    path = tmp_path / "episode.replay"
    path.write_bytes(
        pack_stream_frame(
            str(ReplayFrameType.HEADER),
            {
                "version": REPLAY_LOG_VERSION,
                "episode_idx": 0,
                "overview_group": None,
                "channels": [],
                "cameras": {"agentview": {}},
            },
            seq=0,
        )
    )
    with pytest.raises(ValueError, match="'cameras' must be a list"):
        read_replay_log(path)
