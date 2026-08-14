"""Tests for the replay log: round-trips, truncation, and the writer's gates."""

from __future__ import annotations

import numpy as np
import pytest

from manifold.replay import (
    REPLAY_LOG_VERSION,
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
