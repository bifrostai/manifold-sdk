"""The shared Client, which claims tasks, runs the episode loop and calls the policy.

A Client runs in each benchmark container. It claims `WorkerTask`s until the task
server has none left. For each task, the Client loads the envs through the Service and
gives each waiting episode to the next free env. At each step, it asks the policy for
the action of each env in an episode, in parallel, and steps those envs in a single
Service call. It records a replay of each episode, sends env 0 to the live view, and
reports the results of the task.
"""

from __future__ import annotations

import logging
import socket
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from manifold.adapters.observation import (
    FlipHorizontalCameras,
    FlipVerticalCameras,
    Rotate180Cameras,
    SwapChannelOrder,
)
from manifold.benchmark._replay import _ReplayRecorder
from manifold.benchmark._service import _HooksLogger
from manifold.benchmark._worker import WorkerEpisodeRecord
from manifold.core.pipeline import Pipeline
from manifold.core.sensor import CameraOrientation, ChannelOrder
from manifold.replay import NoteEvent, Severity, replay_log_path
from manifold.wire import (
    BRIDGE_PROTOCOL_VERSION,
    FrameChannel,
    FrameType,
    HelloPayload,
    bridge,
)

if TYPE_CHECKING:
    from manifold.benchmark._live_view import LiveViewPublisher
    from manifold.benchmark._service import ResetResult, Service
    from manifold.benchmark._worker import WorkerSession, WorkerTask
    from manifold.benchmark.config import BenchmarkConfig
    from manifold.core.adapter import ObservationAdapter
    from manifold.core.benchmark import Benchmark
    from manifold.core.values import Action, Observation
    from manifold.replay import ManifoldEvent
    from manifold.wire import ImageFormat


def _upright_image_pipeline(benchmark: Benchmark) -> Pipeline:
    """A pipeline that turns each camera image upright and into RGB.

    The pipeline contains a `FlipVerticalCameras`, `FlipHorizontalCameras` or
    `Rotate180Cameras` adapter for each orientation that is not upright. It contains a
    `SwapChannelOrder` adapter for the cameras that the spec declares as `bgr`.

    The replay recorder and the live view use the result. The policy gets the original
    observation.
    """
    flips = {
        CameraOrientation.FLIPPED_VERTICAL: FlipVerticalCameras,
        CameraOrientation.FLIPPED_HORIZONTAL: FlipHorizontalCameras,
        CameraOrientation.ROTATED_180: Rotate180Cameras,
    }
    adapters: list[ObservationAdapter] = []
    for orientation, flip in flips.items():
        cameras = [str(c.name) for c in benchmark.sensors if c.orientation is orientation]
        if cameras:
            adapters.append(flip(cameras))
    bgr = [str(c.name) for c in benchmark.sensors if c.channel_order is ChannelOrder.BGR]
    if bgr:
        adapters.append(SwapChannelOrder(ChannelOrder.RGB, bgr))
    return Pipeline(observation=adapters)


def _parse_policy_address(url: str) -> tuple[str, int]:
    """The host and port in a policy address such as ``ws://host:port`` or ``host:port``.

    The runner passes a ``ws://`` URL. The bridge sends msgpack frames over a TCP
    socket, so the function uses only the host and the port.
    """
    authority = url.split("://", 1)[-1].rstrip("/")
    host, separator, port = authority.rpartition(":")
    if not separator or not host:
        raise ValueError(f"policy address must be host:port, not {url!r}")
    return host, int(port)


class _ClientPolicyConnection:
    """A connection from an env to the policy server, open for the whole run.

    A batched simulator steps only after the Client has an action for each env. So the
    Client opens a connection to the policy for each env. Each connection sends the
    bridge's messages in the order that `serve` expects.

    The constructor sends the benchmark's spec, and the policy server checks the
    pairing before it answers. `reset` starts an episode, and the policy server then
    drops the actions that remain from the policy's last chunk. `infer` sends an
    observation and returns the action. It checks the action against the benchmark's
    action space.
    """

    def __init__(self, address: str, benchmark: Benchmark, image_format: ImageFormat) -> None:
        host, port = _parse_policy_address(address)
        self._benchmark = benchmark
        self._image_format = image_format
        self._socket = socket.create_connection((host, port))
        self._channel = FrameChannel.from_socket(self._socket)
        self._channel.send(
            FrameType.HELLO,
            HelloPayload(
                protocol_version=BRIDGE_PROTOCOL_VERSION,
                benchmark=benchmark.model_dump(mode="json"),
            ).model_dump(),
        )
        reply = self._channel.recv()
        if reply is None or reply.get("type") != FrameType.READY:
            self._socket.close()
            raise RuntimeError("policy rejected the pairing (no READY)")

    def reset(self) -> None:
        """Start an episode on this connection."""
        self._channel.send(FrameType.RESET, {})

    def infer(self, observation: Observation) -> Action:
        """Send `observation` and return the policy's action."""
        self._channel.send(
            FrameType.OBSERVATION,
            bridge.encode_observation(observation, image_format=self._image_format),
        )
        reply = self._channel.recv()
        if reply is None or reply.get("type") != FrameType.ACTION:
            raise RuntimeError(
                "policy closed the connection or sent a reply other than an action; "
                "check the policy logs for the error"
            )
        action = bridge.decode_action(reply["payload"])
        try:
            self._benchmark.embodiment.action.validate_value(action.values)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"policy returned an action that the benchmark cannot take: {exc}"
            ) from exc
        return action

    def close(self) -> None:
        """Send BYE and close the connection."""
        try:
            self._channel.send(FrameType.BYE, {})
        finally:
            self._socket.close()


class _ClientEpisodeLogger(_HooksLogger):
    """The logger that the hooks get for an episode on an env.

    It writes each message to the run's log, with a prefix that names the env and the
    episode. It also adds the message to the episode's replay as a note. It writes each
    message only once in an episode, so a read that fails at each step gives a single
    line.
    """

    def __init__(self, env_id: int, episode_idx: int) -> None:
        super().__init__(f"env {env_id} episode {episode_idx}: ")
        self._written: set[str] = set()
        self._notes: list[NoteEvent] = []

    def log(self, level: int, msg: object, *args: object, **kwargs: Any) -> None:
        if not self.isEnabledFor(level):
            return
        text = str(msg) % args if args else str(msg)
        if text in self._written:
            return
        self._written.add(text)
        severity = Severity.WARN if level >= logging.WARNING else Severity.INFO
        self._notes.append(NoteEvent(text=text, severity=severity))
        self.logger.log(level, f"{self._prefix}{text}", **kwargs)

    def new_notes(self) -> tuple[NoteEvent, ...]:
        """Return the notes since the last call, for the replay's next frame."""
        notes, self._notes = tuple(self._notes), []
        return notes


@dataclass
class _ClientRunningEpisode:
    """An episode in progress on an env.

    `logger` is the episode's logger, which the hooks get at each step. `instruction` is
    the instruction that the reset returned. The Client puts it on each observation of
    the episode. `initialization_sec` is the time from the reset to the policy's first
    action. It is None until that action arrives. `max_steps` is the step limit of the
    episode: the limit of its task, or else the limit in the config.
    """

    episode_idx: int
    task_id: str
    task_name: str
    instruction: str | None
    logger: _ClientEpisodeLogger
    attributes: dict[str, str]
    observation: Observation
    upright_observation: Observation
    replay: _ReplayRecorder
    started_at: datetime
    initialization_sec: float | None
    steps: int
    max_steps: int | None


class Client:
    """The episode loop of a benchmark container.

    The Client runs the container's tasks on the Service and asks the policy for actions.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        service: Service,
        session: WorkerSession,
        policy_address: str,
        live_view: LiveViewPublisher | None,
    ) -> None:
        self._config = config
        self._service = service
        self._session = session
        self._live_view = live_view
        self._max_steps = config.run.max_steps
        benchmark = config.spec
        self._observation_space = benchmark.observation_space
        self._upright_image_pipeline = _upright_image_pipeline(benchmark)
        # The Client makes a replay recorder for each env and reuses it for the whole run.
        self._replays = [
            _ReplayRecorder(config, service.simulator, env_id) for env_id in range(service.num_envs)
        ]
        # The Client opens a policy connection for each env and reuses it for the whole run.
        self._policies = [
            _ClientPolicyConnection(policy_address, benchmark, config.run.image_format)
            for _ in range(service.num_envs)
        ]
        self._pool = ThreadPoolExecutor(max_workers=service.num_envs)

    def run(self) -> None:
        """Claim and run `WorkerTask`s until none are left, then close the policy connections.

        The caller starts the Service before `run` and stops it after `run`, because some
        simulators end the process when they stop.
        """
        try:
            while (task := self._session.claim()) is not None:
                self._service.load(task)
                try:
                    records, artifacts = self._run_task(task)
                finally:
                    self._service.unload()
                self._session.complete(task, records, artifacts)
                successes = sum(record.success for record in records)
                print(f"task {task.task_id}: {successes}/{len(records)} succeeded", flush=True)
        finally:
            for policy in self._policies:
                policy.close()
            self._pool.shutdown()

    def _run_task(self, task: WorkerTask) -> tuple[list[WorkerEpisodeRecord], dict[int, list]]:
        waiting = deque(task.episode_indices)
        running: dict[int, _ClientRunningEpisode] = {}
        records: list[WorkerEpisodeRecord] = []
        artifacts: dict[int, list] = {}
        try:
            while waiting or running:
                self._start_episodes(task, waiting, running)
                env_ids = sorted(running)
                infer = self._pool.map(
                    lambda i: self._policies[i].infer(running[i].observation), env_ids
                )
                actions = dict(zip(env_ids, infer, strict=True))
                answered = datetime.now(timezone.utc)
                loggers = {env_id: running[env_id].logger for env_id in env_ids}
                for env_id, result in self._service.step(actions, loggers).items():
                    episode = running[env_id]
                    if episode.initialization_sec is None:
                        episode.initialization_sec = (answered - episode.started_at).total_seconds()
                    episode.steps += 1
                    episode.observation = replace(
                        result.observation, instruction=episode.instruction
                    )
                    episode.upright_observation = self._upright_images(episode.observation)
                    out_of_steps = (
                        episode.max_steps is not None and episode.steps >= episode.max_steps
                    )
                    ended = result.success or result.done or out_of_steps
                    episode.replay.record(
                        actions[env_id],
                        episode.upright_observation,
                        self._events(result.events, episode.logger),
                        success=result.success,
                        done=ended,
                    )
                    self._publish_live_view(env_id, episode)
                    if ended:
                        episode.replay.end()
                        records.append(self._episode_record(episode, result.success))
                        artifacts[episode.episode_idx] = [
                            replay_log_path(task.output_dir, episode.episode_idx)
                        ]
                        outcome = "success" if result.success else "failure"
                        print(
                            f"episode {episode.episode_idx}: {outcome} in {episode.steps} steps",
                            flush=True,
                        )
                        del running[env_id]
        finally:
            # After a crash, the Client ends the open episodes. Their replays then contain
            # the steps up to the crash.
            for episode in running.values():
                episode.replay.end()
        records.sort(key=lambda record: record.episode_idx)
        return records, artifacts

    def _start_episodes(
        self, task: WorkerTask, waiting: deque[int], running: dict[int, _ClientRunningEpisode]
    ) -> None:
        """Give the next waiting episodes to free envs.

        If the Service resets all envs together, the method starts episodes only after
        all running episodes have ended.
        """
        free = [env_id for env_id in range(self._service.num_envs) if env_id not in running]
        if not waiting or not free or (self._service.reset_together and running):
            return
        episodes = {env_id: waiting.popleft() for env_id in free[: len(waiting)]}
        began = datetime.now(timezone.utc)
        loggers = {
            env_id: _ClientEpisodeLogger(env_id, episode_idx)
            for env_id, episode_idx in episodes.items()
        }
        results = self._service.reset(episodes, loggers)
        for env_id, episode_idx in episodes.items():
            result = results[env_id]
            # The policy server stores the unused actions of the last chunk. A reset drops
            # them before the new episode.
            self._policies[env_id].reset()
            observation = replace(result.observation, instruction=result.instruction)
            episode = _ClientRunningEpisode(
                episode_idx=episode_idx,
                task_id=result.task_id,
                task_name=result.instruction or self._config.spec.name,
                instruction=result.instruction,
                logger=loggers[env_id],
                attributes=result.attributes,
                observation=observation,
                upright_observation=self._upright_images(observation),
                replay=self._replays[env_id],
                started_at=began,
                initialization_sec=None,
                steps=0,
                max_steps=self._episode_max_steps(result),
            )
            running[env_id] = episode
            episode.replay.begin(
                task.output_dir,
                episode_idx,
                episode.upright_observation,
                result.attributes,
                self._events(result.events, episode.logger),
            )
            self._publish_live_view(env_id, episode)

    @staticmethod
    def _events(
        events: tuple[ManifoldEvent, ...], logger: _ClientEpisodeLogger
    ) -> tuple[ManifoldEvent, ...]:
        """The hooks' events, then the notes that the hooks logged since the last frame."""
        return (*events, *logger.new_notes())

    def _episode_max_steps(self, result: ResetResult) -> int | None:
        """The episode's step limit: the task's limit from the hooks, or else the config's.

        Raises:
            ValueError: If the hooks return a limit and `run.max_steps` is set. The config
                would then state a limit that the episode does not use.
        """
        if result.max_steps is None:
            return self._max_steps
        if self._max_steps is not None:
            raise ValueError(
                f"the hooks returned a step limit of {result.max_steps} for task "
                f"{result.task_id!r}, and run.max_steps is {self._max_steps}; set "
                "run.max_steps to null when the hooks give each task its limit"
            )
        return result.max_steps

    def _upright_images(self, observation: Observation) -> Observation:
        """`observation` with each camera image upright and in RGB."""
        return self._upright_image_pipeline.apply_observation(
            observation, source=self._observation_space
        )

    def _publish_live_view(self, env_id: int, episode: _ClientRunningEpisode) -> None:
        """Send the latest observation of env 0 to the live view.

        The live view shows a single env.
        """
        if self._live_view is None or env_id != 0:
            return
        self._live_view.publish(
            episode.upright_observation,
            episode_idx=episode.episode_idx,
            step=episode.steps,
            task=episode.task_name,
            task_id=episode.task_id,
        )

    def _episode_record(self, episode: _ClientRunningEpisode, success: bool) -> WorkerEpisodeRecord:
        return WorkerEpisodeRecord(
            episode_idx=episode.episode_idx,
            task_name=episode.task_name,
            success=success,
            steps=episode.steps,
            initialization_sec=episode.initialization_sec or 0.0,
            started_at=episode.started_at,
            ended_at=datetime.now(timezone.utc),
            task_id=episode.task_id,
            attributes=episode.attributes,
        )


__all__ = ["Client"]
