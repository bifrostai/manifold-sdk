"""The replay log a benchmark writes beside its rollup, for a replay.

One module: `log.py` owns the format, the writer a benchmark uses, and a reader.
Nothing here touches the bridge - a replay is never part of the stream the two
sides exchange (ADR 0003).
"""

from __future__ import annotations

from manifold.replay.log import (
    REPLAY_LOG_DIR,
    REPLAY_LOG_SUFFIX,
    REPLAY_LOG_VERSION,
    Channel,
    ChannelKind,
    Mesh,
    Pose,
    ReplayFrameType,
    ReplayLog,
    ReplayLogWriter,
    ReplayStep,
    SceneBody,
    ScenePart,
    read_replay_log,
    replay_log_path,
)

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
