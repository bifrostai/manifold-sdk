"""Drive the bridge choreography for both halves of a pairing, backend-agnostic.

`wire/bridge` defines the frames; this recipe owns the conversation over them, so
neither side hand-rolls the accept loop, HELLO/READY handshake, and per-step
send/recv dance. `serve` is the policy side (listen, accept, gate each pairing,
answer observations with actions); `run_benchmark` is the runner side (connect,
advertise, wait for READY, drive episodes); `evaluate` is the zero-hop degenerate
running the identical gate and per-step fold in-process with no server, bridge, or
threads. The policy author supplies an endpoint (a loaded model plus a way to mint
sessions); the runner supplies the environment as plain callables.

One served model, many runners: a `PolicyEndpoint` loads the model once and each
connection gets its own `Session` and `PipelineState` over that shared model.
`serve` runs a daemon thread per connection for concurrent shards; forward
concurrency safety is the endpoint's responsibility (its lock is taken inside
`forward`/`infer`). Each session is `close`d on disconnect to release its scratch.

The transport today is a TCP socket carrying the bridge's length-prefixed frames;
the send/recv surface is named as a minimal `Transport` Protocol (which
`FrameChannel` satisfies structurally), marking the seam where a future transport
plugs in. Compatibility-checking lives on the policy side (ADR-0001): `serve` runs
`check_compatibility` and `verify` before sending READY. Core never imports
`recipes/`; this builds on core.
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from manifold.core.benchmark import Benchmark
from manifold.core.check import check_compatibility
from manifold.core.pipeline import Pipeline
from manifold.core.state import DEFAULT_LANE, PipelineState
from manifold.core.verify import verify
from manifold.recipes.dispatch import assert_shared_profile, multi_pairing_pipeline
from manifold.recipes.sharding import shard_episode_ids
from manifold.wire import BRIDGE_PROTOCOL_VERSION, FrameChannel, FrameType, bridge

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from manifold.core.native_layout import NativeLayout
    from manifold.core.observation_space import ObservationSpace
    from manifold.core.policy import PolicySignature
    from manifold.core.values import Action, Observation
    from manifold.recipes.pairing import Pairing
    from manifold.wire import ImageFormat


def _emit(message: str) -> None:
    """Print one progress line, flushing so it appears under a redirected stdout.

    Python block-buffers a non-TTY stdout by default, which withholds progress
    when output is piped to a file; flushing here keeps a long-running serve loop
    or run from looking hung, in one place rather than at every call site. Callers
    with their own sink pass `on_event`.
    """
    print(message, flush=True)


def _parse_server(server: str) -> tuple[str, int]:
    """Split a ``host:port`` string into its parts (port required).

    Splits on the LAST colon so an IPv6 host (or a host carrying a colon) is handled;
    both halves must be non-empty. Raises ``ValueError`` — a catchable misuse,
    consistent with the rest of this module, so a CLI wrapper translates it to its own
    exit code rather than a library caller getting an uncatchable `SystemExit`.
    """
    host, _, port = server.rpartition(":")
    if not host or not port:
        raise ValueError(f"server must be host:port, got {server!r}")
    return host, int(port)


def _run_episode_in_process(
    session: Session,
    pipeline: Pipeline,
    observation_source: ObservationSpace,
    signature: PolicySignature,
    state: PipelineState,
    reset: Callable[[], Observation],
    step: Callable[[Action], StepResult],
    max_steps: int,
    *,
    episode_idx: int,
    benchmark_name: str,
) -> EpisodeRecord:
    """Drive one in-process episode, returning the record of what it did.

    The in-process twin of `_run_episode`: it resets the session and the lane's
    `PipelineState`, then loops `reset/step` against the shared `_infer_step` fold
    — the same observation->infer->action kernel `_serve_session` runs per frame,
    just without the wire in between.
    """
    state.reset_lane(DEFAULT_LANE)
    session.reset()
    started_at = datetime.now(timezone.utc)
    observation = reset()
    task_name = observation.instruction or benchmark_name
    steps = 0
    success = False
    initialization_sec = 0.0
    for _ in range(max_steps):
        action = _infer_step(session, observation, pipeline, observation_source, signature, state)
        if steps == 0:
            initialization_sec = (datetime.now(timezone.utc) - started_at).total_seconds()
        result = step(action)
        steps += 1
        observation = result.observation
        if result.success or result.done:
            success = result.success
            break
    return EpisodeRecord(
        episode_idx=episode_idx,
        task_name=task_name,
        success=success,
        steps=steps,
        initialization_sec=initialization_sec,
        started_at=started_at,
        ended_at=datetime.now(timezone.utc),
    )


def _serve_connection(
    endpoint: PolicyEndpoint,
    pipeline: Pipeline | Callable[[Benchmark], Pipeline] | None,
    conn: socket.socket,
    addr: Any,
    slots: threading.BoundedSemaphore,
    emit: Callable[[str], None],
) -> None:
    """Own one runner connection end to end on its worker thread.

    Wraps `_serve_session` so any error in a single connection — a rejected
    pairing, a dropped socket, a malformed frame — is caught, logged, and confined
    here: it never propagates to the accept loop and so never kills the server or
    a sibling shard. The semaphore slot taken on accept is always released, and the
    socket always closed, even on error.
    """
    try:
        with conn:
            _serve_session(endpoint, pipeline, conn, emit)
        emit(f"connection from {addr} closed")
    except Exception as exc:  # one shard's failure must not kill the server.
        emit(f"connection from {addr} errored, dropping it: {exc!r}")
    finally:
        slots.release()


def _serve_session(
    endpoint: PolicyEndpoint,
    pipeline: Pipeline | Callable[[Benchmark], Pipeline] | None,
    conn: socket.socket,
    emit: Callable[[str], None],
) -> None:
    """Run one runner connection: handshake, gate the pairing, then answer frames.

    Mints this connection's own `session` over the endpoint's shared model and its
    own `PipelineState`, so concurrent connections never interleave per-session
    state. Inference safety across connections is the endpoint's concern:
    `Session.infer` takes the endpoint's lock around the shared forward. The
    session is `close`d in a `finally` once the connection ends — a dropped shard
    must retire its per-session scratch rather than leak it (see the
    `Session.close` Triton state-leak note).
    """
    channel = FrameChannel.from_socket(conn)
    hello = channel.recv()
    if hello is None or hello.get("type") != FrameType.HELLO:
        emit("expected a hello frame advertising the benchmark; closing")
        return
    benchmark = Benchmark.model_validate(hello["payload"]["benchmark"])

    signature = endpoint.signature
    resolved = _resolve_pipeline(pipeline, benchmark)
    if not _gate(signature, benchmark, resolved, emit):
        return

    # One PipelineState lives with this connection, not the endpoint, so a stateful
    # adapter's per-episode history never interleaves across concurrent connections.
    state = PipelineState()
    observation_source = benchmark.observation_space
    # Mint the session inside the try so its close() is guaranteed even if the
    # READY send itself fails (a peer that drops between gate and READY must not
    # leak the freshly-minted per-session scratch).
    session = endpoint.session()
    try:
        channel.send(FrameType.READY, {})
        while True:
            frame = channel.recv()
            if frame is None or frame.get("type") == FrameType.BYE:
                return
            kind = frame.get("type")
            if kind == FrameType.RESET:
                state.reset_lane(DEFAULT_LANE)
                session.reset()
                continue
            if kind == FrameType.OBSERVATION:
                observation = bridge.decode_observation(frame.get("payload", {}))
                action = _infer_step(
                    session, observation, resolved, observation_source, signature, state
                )
                channel.send(FrameType.ACTION, bridge.encode_action(action))
                continue
            emit(f"ignoring unexpected frame {kind!r}")
    finally:
        # Retire the session's per-session scratch on teardown — see Session.close.
        session.close()


def _infer_step(
    session: Session,
    observation: Observation,
    pipeline: Pipeline,
    observation_source: ObservationSpace,
    signature: PolicySignature,
    state: PipelineState,
) -> Action:
    """One policy-side inference step: the full benchmark->model->benchmark fold.

    The shared kernel `serve` runs per OBSERVATION frame and `evaluate` per in-process
    step (ADR-0001, "the pipeline reaches the model"). The packing phase is optional: a
    pairing with no pack/unpack answers the folded observation directly via
    `session.infer`. Single-lane (DEFAULT_LANE).
    """
    observation = pipeline.apply_observation(
        observation, source=observation_source, state=state, lane=DEFAULT_LANE
    )
    if pipeline.pack is not None and pipeline.unpack is not None:
        # The packing path: pack to the model's native dict, run the open-loop
        # forward, then read one step of the raw chunk back into an Action.
        native = pipeline.apply_pack(observation, source=signature.observation_space)
        raw_chunk, step = session.advance(native)
        action = pipeline.apply_unpack(raw_chunk, step=step)
    else:
        # The no-packing degenerate: the session answers the observation directly.
        action = session.infer(observation)
    return pipeline.apply_action(
        action, source=signature.action_space, state=state, lane=DEFAULT_LANE
    )


def _gate(
    signature: PolicySignature,
    benchmark: Benchmark,
    pipeline: Pipeline,
    emit: Callable[[str], None],
) -> bool:
    """Check and verify the pairing, emitting status; True iff READY may be sent."""
    report = check_compatibility(signature, benchmark, pipeline)
    verified = verify(signature, benchmark, pipeline)
    emit(f"check_compatibility: {report.status.value}")
    if report.lossless is not None:
        n_obs, n_act = len(pipeline.observation), len(pipeline.action)
        emit(
            f"  pipeline: {n_obs} observation + {n_act} action adapter(s), "
            f"lossless={report.lossless}"
        )
    emit(f"verify (synthetic data): {verified.summary()}")
    if report.ok and verified.ok:
        return True
    reasons = report.reasons + list(verified.reasons)
    emit(f"incompatible — rejecting the pairing. reasons: {reasons}")
    return False


def _resolve_pipeline(
    pipeline: Pipeline | Callable[[Benchmark], Pipeline] | None,
    benchmark: Benchmark,
) -> Pipeline:
    """Turn the `pipeline` argument into a concrete Pipeline for this benchmark."""
    if pipeline is None:
        return Pipeline()
    if isinstance(pipeline, Pipeline):
        return pipeline
    return pipeline(benchmark)


def _run_episode(
    channel: Transport,
    reset: Callable[[], Observation],
    step: Callable[[Action], StepResult],
    max_steps: int,
    image_format: ImageFormat,
    *,
    episode_idx: int,
    benchmark_name: str,
) -> EpisodeRecord:
    """Drive one episode over the channel, returning the record of what it did.

    The clock brackets the whole rollout, reset included, since an episode begins
    when the environment is reset.
    """
    channel.send(FrameType.RESET, {})
    started_at = datetime.now(timezone.utc)
    observation = reset()
    task_name = observation.instruction or benchmark_name
    steps = 0
    success = False
    initialization_sec = 0.0
    for _ in range(max_steps):
        channel.send(
            FrameType.OBSERVATION,
            bridge.encode_observation(observation, image_format=image_format),
        )
        reply = channel.recv()
        if reply is None or reply.get("type") != FrameType.ACTION:
            raise PairingRejected("expected an action frame from the policy")
        action = bridge.decode_action(reply["payload"])
        if steps == 0:
            initialization_sec = (datetime.now(timezone.utc) - started_at).total_seconds()
        result = step(action)
        steps += 1
        observation = result.observation
        if result.success or result.done:
            success = result.success
            break
    return EpisodeRecord(
        episode_idx=episode_idx,
        task_name=task_name,
        success=success,
        steps=steps,
        initialization_sec=initialization_sec,
        started_at=started_at,
        ended_at=datetime.now(timezone.utc),
    )


@runtime_checkable
class Transport(Protocol):
    """The minimal frame surface `serve`/`run_benchmark` need from a connection.

    Naming this surface — rather than typing against the concrete `FrameChannel`
    (which satisfies it structurally) — marks the seam where a future shm or iroh
    transport plugs in. Today the only backing is `FrameChannel.from_socket` over TCP.
    """

    def send(self, frame_type: FrameType | str, payload: dict[str, Any]) -> None:
        """Frame and write one typed message with its payload."""
        ...

    def recv(self) -> dict[str, Any] | None:
        """Read the next frame, or None at end of stream."""
        ...


class Session(Protocol):
    """One connection's view of a served policy: a pure open-loop chunk buffer.

    Minted per connection by `PolicyEndpoint.session`; it does not pack (ADR-0001 —
    the pipeline packs/unpacks; the session only decides when to run a fresh forward
    over the shared, lock-guarded model). Per-session scratch must be addressed by a
    stable session key, never call order, so the surface stays liftable to a future
    batched forward over many sessions.
    """

    def advance(self, native: dict[str, Any], /) -> tuple[Any, int]:
        """Return `(raw_chunk, step)`, running a fresh forward when the horizon is exhausted."""
        ...

    def infer(self, observation: Observation, /) -> Action:
        """Answer one observation with an action — the no-packing degenerate path."""
        ...

    def reset(self) -> None:
        """Clear this session's per-episode state (e.g. its action-chunk buffer)."""
        ...

    def close(self) -> None:
        """Retire this session's per-session scratch on disconnect.

        Explicit rather than GC'd: scratch never retired on disconnect accumulates and
        eventually starves new sessions (the served model itself is shared, long-lived).
        """
        ...


class PolicyEndpoint(Protocol):
    """A loaded policy that serves many connections, each its own session.

    Loads the model once and exposes its `signature` (gated against). `session`
    mints a fresh `Session` per connection over the shared model; `forward` runs
    that shared model, taking the endpoint's own lock so concurrent sessions never
    corrupt each other — so `serve` does not add inference serialization of its own.
    """

    signature: PolicySignature

    def session(self) -> Session:
        """Mint a fresh per-connection session over this endpoint's shared model."""
        ...

    def forward(self, native: dict[str, Any], /) -> Any:
        """Run the shared model on a native input dict; return its raw output chunk.

        Takes the endpoint's lock around the loaded model. The returned value is the
        model's raw chunk in its own wire form (a per-key tensor dict, or a single
        indexed array); the pipeline's unpack phase reads one step of it.
        """
        ...


class PolicyProfile(Protocol):
    """The declarative description of a servable policy: how to stand one up.

    One embodiment's worth of server description — the `PolicySignature` gated
    against, the default checkpoint, the input/output `NativeLayout`s, and a `load`.
    A concrete profile (a frozen dataclass plus its own backend fields) satisfies
    this structurally, so a launcher types against this SHAPE rather than probing a
    backend with stringly-typed `hasattr` checks. `load` is the one method a launcher
    must call; the fuller declared shape keeps a generalist mount typed against one
    named contract.
    """

    signature: PolicySignature
    default_weights: str
    input_layout: NativeLayout
    output_layout: NativeLayout

    def load(self, weights: str, device: str | None) -> PolicyEndpoint:
        """Load the checkpoint and return the shared, ready-to-serve endpoint."""
        ...


class ChunkEndpoint(Protocol):
    """Structural view of the endpoint `OpenLoopChunkQueue` holds.

    Typing against this — rather than `PolicyEndpoint`, which does not declare
    `.profile` — keeps the queue backend-agnostic. `profile` is `Any` because a
    backend's profile is its own frozen dataclass; the queue only reads `exec_steps`.
    """

    @property
    def profile(self) -> Any:
        """The backend-specific profile this endpoint was loaded from."""
        ...


class OpenLoopChunkQueue:
    """One connection's open-loop chunk buffer over a shared endpoint.

    A reusable `Session` implementation for a policy backend to subclass: it owns its
    own chunk buffer (no shard shares it) and does not pack. Subclasses implement
    `_forward` (the backend's shared-model call) and override `reset`/`close` only if
    they hold per-session model state.
    """

    def __init__(self, endpoint: ChunkEndpoint) -> None:
        self._endpoint = endpoint
        self._profile = endpoint.profile
        # The buffered raw model chunk and the step pointer into it. None until the
        # first advance; the pointer wraps at exec_steps, triggering a fresh forward.
        self._chunk: Any = None
        self._step = 0

    def _forward(self, native: dict[str, Any], /) -> Any:
        """Run the backend's shared-model forward on `native`; return the raw chunk.

        The one seam between backends. The base raises so a backend that leaves it
        unimplemented fails loudly rather than silently serving a stale buffer.
        """
        raise NotImplementedError("OpenLoopChunkQueue subclass must implement _forward")

    def advance(self, native: dict[str, Any], /) -> tuple[Any, int]:
        """Serve the next open-loop step for `native`: return (raw_chunk, step).

        When the horizon has drained (no buffered chunk, or `exec_steps` served), run
        a fresh `_forward` and reset the step pointer; otherwise serve the next step.
        """
        if self._chunk is None or self._step >= self._profile.exec_steps:
            self._chunk = self._forward(native)
            self._step = 0
        step = self._step
        self._step += 1
        return self._chunk, step

    def infer(self, _observation: Any, /) -> Any:  # pragma: no cover - packing path only
        """Not used: a pairing with a packing phase serves via `advance`, not `infer`."""
        raise NotImplementedError(f"{type(self).__name__} serves the packing path via advance()")

    def reset(self) -> None:
        """Clear this session's open-loop chunk buffer so a new episode starts fresh.

        The base clears only the per-session buffer — a complete per-episode reset for
        a stateless backend. A backend with per-session model state (e.g. recurrent
        memory) overrides this to also arm that state's reset, calling `super().reset()`
        to clear the buffer.
        """
        self._chunk = None
        self._step = 0

    def close(self) -> None:
        """Retire this session — a no-op for a stateless backend.

        The base session holds only its own chunk buffer (Python-managed, freed when
        the session is dropped) and the shared endpoint reference, so there is nothing
        to release. A backend with a per-session model-memory slot overrides this to
        retire that slot (see the `Session.close` note above).
        """


class PairingRejected(RuntimeError):
    """The policy did not confirm the pairing, or the bridge broke mid-run.

    Raised on the runner side when the policy closes without READY (it found the
    pairing incompatible), or when a frame arrives out of the expected order.
    This is a library function, so a rejection is a recoverable exception the
    caller may catch — not a process exit.
    """


@dataclass(frozen=True)
class StepResult:
    """The outcome of stepping the environment with one action.

    `observation` is the next observation the runner ships back to the policy;
    `success` marks the task solved this step; `done` marks the episode otherwise
    finished (truncated, terminated without success). Either flag ends the
    episode; only `success` counts toward the tally.
    """

    observation: Observation
    success: bool
    done: bool


@dataclass(frozen=True)
class EpisodeRecord:
    """One finished episode, as the benchmark observed it.

    `started_at` and `ended_at` bracket the rollout, so `elapsed_sec` derives from
    them rather than being measured separately and free to disagree. `task_name` is
    the reset observation's instruction where the benchmark publishes one, and the
    benchmark's own name otherwise. `episode_idx` is the global episode id, which a
    sharded run partitions disjointly, so two shards never report the same one.
    """

    episode_idx: int
    task_name: str
    success: bool
    steps: int
    initialization_sec: float
    started_at: datetime
    ended_at: datetime

    @property
    def elapsed_sec(self) -> float:
        """The episode's wall-clock duration, in seconds."""
        return (self.ended_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class BenchmarkResult:
    """A finished run: one record per episode it drove.

    `episodes` and `successes` are views over `records`, so a tally cannot drift
    from the episodes it counts.
    """

    records: tuple[EpisodeRecord, ...] = ()

    @property
    def episodes(self) -> int:
        """How many episodes ran."""
        return len(self.records)

    @property
    def successes(self) -> int:
        """How many episodes succeeded."""
        return sum(1 for record in self.records if record.success)

    @property
    def success_rate(self) -> float:
        """The fraction of episodes that succeeded, or 0.0 for an empty run."""
        if self.episodes == 0:
            return 0.0
        return self.successes / self.episodes

    def format_shard(self, shard_index: int, num_shards: int) -> str:
        """The one-line per-shard tally a runner prints at the end of a sharded run.

        Returns the string rather than printing it, so the CLI layer owns stdout;
        the SDK formats its own result type without reaching for `print`.
        """
        return (
            f"[bench shard {shard_index}/{num_shards}] "
            f"{self.successes}/{self.episodes} succeeded ({self.success_rate:.1%})"
        )


def write_rollup(result: BenchmarkResult, output_dir: Path, *, benchmark_name: str) -> Path:
    """Write the run's records to ``<output-dir>/results/<benchmark>.json``, and return the path.

    This is the file a runner scans a benchmark worker's output directory for: a
    flat document keyed ``records``, one entry per episode, instants in ISO 8601.
    The runner derives ``elapsed_sec`` from the two instants, so the rollup does
    not carry a duration free to disagree with them.
    """
    records = [
        {
            "episode_idx": record.episode_idx,
            "task_name": record.task_name,
            "success": record.success,
            "steps": record.steps,
            "initialization_sec": record.initialization_sec,
            "started_at": record.started_at.isoformat(),
            "ended_at": record.ended_at.isoformat(),
        }
        for record in result.records
    ]
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{benchmark_name}.json"
    path.write_text(json.dumps({"records": records}))
    return path


def serve(
    endpoint: PolicyEndpoint,
    *,
    pipeline: Pipeline | Callable[[Benchmark], Pipeline] | None = None,
    host: str = "127.0.0.1",
    port: int,
    max_workers: int = 8,
    on_event: Callable[[str], None] = _emit,
) -> None:
    """Serve one loaded policy to many runner shards concurrently over the bridge.

    Accepts connections, each handled on its own daemon thread with its own session
    and `PipelineState`, so shards sharing the loaded model have isolated per-session
    state. Concurrency is bounded by `max_workers` (a semaphore the accept loop
    acquires before spawning a worker). Per connection the worker reads HELLO, gates
    the pairing (`check_compatibility` + `verify`), and only if both pass sends READY
    and answers each OBSERVATION with an ACTION until BYE. Inference safety is the
    endpoint's concern (its lock is inside the forward), so `serve` does not add
    loop-level serialization. One connection's failure is confined to its worker thread; a
    KeyboardInterrupt stops accepting and waits briefly for in-flight workers.

    `pipeline` may be None (empty pipeline), a callable invoked per connection with
    the benchmark (the dynamic-injection hook — the pipeline arrives with the job),
    or a `Pipeline` used as-is. `on_event` is called from worker threads, so a custom
    sink must be concurrency-safe.
    """
    emit = on_event
    slots = threading.BoundedSemaphore(max_workers)
    workers: list[threading.Thread] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        # A backlog > 1 lets several shards queue at the kernel while the accept
        # loop is between iterations, so a burst of runner connects is not refused.
        listener.listen(max(max_workers, 8))
        emit(f"listening on {host}:{port} (TCP), up to {max_workers} concurrent shard(s)")
        try:
            while True:
                slots.acquire()
                try:
                    conn, addr = listener.accept()
                except BaseException:
                    # Nothing was spawned, so release the slot we took on the way in.
                    slots.release()
                    raise
                emit(f"benchmark connected from {addr}")
                worker = threading.Thread(
                    target=_serve_connection,
                    args=(endpoint, pipeline, conn, addr, slots, emit),
                    daemon=True,
                )
                workers.append(worker)
                worker.start()
                # Drop references to finished workers so the list cannot grow
                # without bound across a long-lived server.
                workers = [w for w in workers if w.is_alive()]
        except KeyboardInterrupt:
            emit("interrupted; shutting down")
    # The listener is closed; give in-flight workers a brief, bounded chance to
    # finish their current frame exchange. They are daemon threads, so the process
    # never hangs on a stuck connection.
    for worker in workers:
        worker.join(timeout=1.0)


def run_benchmark(
    benchmark: Benchmark,
    reset: Callable[[], Observation],
    step: Callable[[Action], StepResult],
    *,
    episodes: int,
    max_steps: int,
    image_format: ImageFormat = "raw",
    host: str = "127.0.0.1",
    port: int,
    on_event: Callable[[str], None] = _emit,
) -> BenchmarkResult:
    """Run a benchmark against a remote policy over the bridge, in native form.

    Connects, advertises the benchmark in HELLO, and blocks on READY before opening
    any episode — so the policy-side gate fences off every rollout. The runner is a
    dumb native-form driver: it never adapts an observation or action, because the
    policy side owns all conversion (ADR-0001). This is one runner shard; several
    against the same served `PolicyEndpoint` run concurrently, each on its own session.
    `image_format` is the sensor encoding for frames ("raw", "jpeg", "png"); "raw"
    does not need an extra dependency.

    Returns:
        A record for each episode run, and the tally over them.

    Raises:
        PairingRejected: If the policy closes without READY, or a frame arrives out
            of the expected order.
    """
    return _run_connected(
        benchmark,
        lambda _episode_id: reset(),
        step,
        episode_ids=range(episodes),
        max_steps=max_steps,
        image_format=image_format,
        host=host,
        port=port,
        on_event=on_event,
    )


def _run_connected(
    benchmark: Benchmark,
    reset_episode: Callable[[int], Observation],
    step: Callable[[Action], StepResult],
    *,
    episode_ids: Sequence[int],
    max_steps: int,
    image_format: ImageFormat,
    host: str,
    port: int,
    on_event: Callable[[str], None],
) -> BenchmarkResult:
    """Hold one connection open and drive the named episodes down it, in order.

    The shared body of `run_benchmark` and `run_episodes`: they differ only in where
    the ids come from, and each episode is stamped with the id it ran rather than
    with its position here, so two shards of one run never report the same
    `episode_idx`. The progress line counts position, since what an operator watching
    one shard wants is how far through its own list it is.
    """
    emit = on_event
    ids = list(episode_ids)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        channel = FrameChannel.from_socket(sock)
        channel.send(
            FrameType.HELLO,
            {
                "protocol_version": BRIDGE_PROTOCOL_VERSION,
                "benchmark": benchmark.model_dump(mode="json"),
            },
        )
        reply = channel.recv()
        if reply is None or reply.get("type") != FrameType.READY:
            raise PairingRejected("policy rejected the pairing (no READY)")
        emit("policy confirmed the pairing (ready)")

        records: list[EpisodeRecord] = []
        for position, episode_id in enumerate(ids):
            record = _run_episode(
                channel,
                lambda episode_id=episode_id: reset_episode(episode_id),
                step,
                max_steps,
                image_format,
                episode_idx=episode_id,
                benchmark_name=benchmark.name,
            )
            records.append(record)
            outcome = "success" if record.success else "failure"
            emit(f"episode {position + 1}/{len(ids)}: {outcome}")
        channel.send(FrameType.BYE, {})
        result = BenchmarkResult(records=tuple(records))
        emit(f"done — {result.successes}/{len(ids)} succeeded")
        return result


def run_episodes(
    benchmark: Benchmark,
    reset_episode: Callable[[int], Observation],
    step: Callable[[Action], StepResult],
    *,
    server: str,
    episode_ids: Sequence[int],
    max_steps: int,
    image_format: ImageFormat = "raw",
    on_event: Callable[[str], None] = _emit,
) -> BenchmarkResult:
    """Run the named global episodes of a benchmark against a remote policy.

    The dispatch contract is a list of episodes: the caller passes the ids to run, and
    this drives exactly those, in the given order. A stride shard is one such list,
    and the set an interrupted attempt never reached is another. This owns the
    run-mechanics around one connection: it parses the server string and hands the
    ids down, so each episode is seeded by the id it runs and each record is stamped
    with it.

    `reset_episode` takes the GLOBAL episode id it begins (to seed the env / pick a
    grid cell) — the only shape difference from `run_benchmark`'s no-argument
    `reset`, since the caller chooses WHICH episode each reset runs. `server` is the
    policy as a ``host:port`` string.

    The ids must be distinct: `episode_idx` identifies an episode within a run, so a
    repeat would report one index twice. An empty list runs nothing, which is the
    result when there is nothing outstanding.

    Returns:
        A record for each episode run, stamped with its global id, and the tally.

    Raises:
        ValueError: If `server` is not ``host:port``, or `episode_ids` holds a
            negative or a repeated id.
        PairingRejected: Propagated from `run_benchmark`.
    """
    host, port = _parse_server(server)
    ids = list(episode_ids)
    if any(episode_id < 0 for episode_id in ids):
        raise ValueError("episode ids must be >= 0")
    if len(set(ids)) != len(ids):
        raise ValueError("episode ids must be distinct")
    return _run_connected(
        benchmark,
        reset_episode,
        step,
        episode_ids=ids,
        max_steps=max_steps,
        image_format=image_format,
        host=host,
        port=port,
        on_event=on_event,
    )


def run_sharded_benchmark(
    benchmark: Benchmark,
    reset_episode: Callable[[int], Observation],
    step: Callable[[Action], StepResult],
    *,
    server: str,
    total_episodes: int,
    num_shards: int = 1,
    shard_index: int = 0,
    max_steps: int,
    image_format: ImageFormat = "raw",
    on_event: Callable[[str], None] = _emit,
) -> BenchmarkResult:
    """Run one shard of a benchmark against a remote policy, owning the run-mechanics.

    The stride-shard spelling of `run_episodes`: it partitions the global episode set
    into this shard's ids (`shard_episode_ids`), dispatches them, and emits the
    per-shard tally.

    `reset_episode` takes the GLOBAL episode id it begins, as `run_episodes` does.
    `total_episodes` is the global count partitioned disjointly across shards; this
    shard runs `len(shard_ids)`. `server` is the policy as a ``host:port`` string;
    `on_event` also receives the final per-shard tally line.

    Returns:
        A record for each episode this shard ran, and the tally over them.

    Raises:
        ValueError: If `server` is not ``host:port``, or the shard flags are invalid.
        PairingRejected: Propagated from `run_benchmark`.
    """
    shard_ids = shard_episode_ids(total_episodes, num_shards, shard_index)
    result = run_episodes(
        benchmark,
        reset_episode,
        step,
        server=server,
        episode_ids=shard_ids,
        max_steps=max_steps,
        image_format=image_format,
        on_event=on_event,
    )
    on_event(result.format_shard(shard_index, num_shards))
    return result


def evaluate(
    endpoint: PolicyEndpoint,
    benchmark: Benchmark,
    reset: Callable[[], Observation],
    step: Callable[[Action], StepResult],
    *,
    pipeline: Pipeline | Callable[[Benchmark], Pipeline] | None = None,
    episodes: int,
    max_steps: int,
    on_event: Callable[[str], None] = _emit,
) -> BenchmarkResult:
    """Run a benchmark against a policy *in process*: no server, no bridge, no threads.

    The zero-hop case of the connector algebra `serve`/`run_benchmark` implement: the
    same gate and the same per-step fold, but the observation and action never leave
    the process. Use it to test a pairing locally before running the served, sharded
    path.

    It mints exactly one `session` (a single in-process lane), resets the session and
    that lane's `PipelineState` at each episode boundary, and `close`s the session in
    a `finally` (mirroring `serve`'s teardown) so even the in-process path retires
    per-session scratch deterministically. `reset`/`step` are the same callables
    `run_benchmark` takes, so a caller swaps recipes without rewriting its environment.
    `pipeline` bridges exactly as `serve` resolves it (None / callable / `Pipeline`).

    Returns:
        A record for each episode run, and the tally over them.

    Raises:
        PairingRejected: If the policy-side gate rejects the pairing. In process there
            is no connection to close, so the rejection surfaces as this exception
            (the served path signals it by closing without READY).
    """
    emit = on_event
    resolved = _resolve_pipeline(pipeline, benchmark)
    signature = endpoint.signature
    if not _gate(signature, benchmark, resolved, emit):
        raise PairingRejected("policy rejected the pairing (incompatible)")
    emit("policy confirmed the pairing (ready)")

    observation_source = benchmark.observation_space
    state = PipelineState()
    session = endpoint.session()
    try:
        records: list[EpisodeRecord] = []
        for episode in range(episodes):
            record = _run_episode_in_process(
                session,
                resolved,
                observation_source,
                signature,
                state,
                reset,
                step,
                max_steps,
                episode_idx=episode,
                benchmark_name=benchmark.name,
            )
            records.append(record)
            outcome = "success" if record.success else "failure"
            emit(f"episode {episode + 1}/{episodes}: {outcome}")
        result = BenchmarkResult(records=tuple(records))
        emit(f"done — {result.successes}/{episodes} succeeded")
        return result
    finally:
        # Retire the session's per-session scratch on teardown — see Session.close.
        session.close()


def launch_server(
    pairings: Sequence[Pairing],
    *,
    weights: str | None = None,
    device: str | None = None,
    host: str = "0.0.0.0",
    port: int,
    max_workers: int = 8,
    on_event: Callable[[str], None] = _emit,
) -> None:
    """Build the shared policy from one-or-more pairings and serve it.

    The reusable launcher body, factored out of any CLI: assert every pairing shares
    one profile (one loaded model serves them all), build the endpoint once (falling
    back to `default_weights` when `weights` is None), resolve the pipeline (a single
    pairing serves its own; several build the benchmark->pipeline matchmaker), and hand
    it to `serve`. Raises `ValueError` (from `assert_shared_profile`) rather than
    calling `sys.exit`, so a CLI can translate it to its own exit code.

    All `pairings` must share the same profile. `weights` falls back to that profile's
    `default_weights` when None; `device` autodetects when None.
    """
    profile = assert_shared_profile(p.profile for p in pairings)
    endpoint = profile.load(weights if weights is not None else profile.default_weights, device)
    if len(pairings) == 1:
        pipeline = pairings[0].pipeline
    else:
        pipeline = multi_pairing_pipeline((p.benchmark, p.pipeline) for p in pairings)
    serve(
        endpoint,
        pipeline=pipeline,
        host=host,
        port=port,
        max_workers=max_workers,
        on_event=on_event,
    )


__all__ = [
    "BenchmarkResult",
    "ChunkEndpoint",
    "EpisodeRecord",
    "OpenLoopChunkQueue",
    "PairingRejected",
    "PolicyEndpoint",
    "PolicyProfile",
    "Session",
    "StepResult",
    "Transport",
    "evaluate",
    "launch_server",
    "run_benchmark",
    "run_episodes",
    "run_sharded_benchmark",
    "serve",
    "write_rollup",
]
