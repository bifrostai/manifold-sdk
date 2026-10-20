"""Serve a policy over the bridge, or run it in process against a stand-in benchmark.

`manifold.wire.bridge` defines the frames, and this module runs the policy's side
of the conversation over them. `serve` listens, accepts each connection, checks the
pairing and answers observations with actions. `evaluate` runs the same check and
the same step in a single process, with no server, bridge or threads. A policy
author supplies an endpoint, which is a loaded model and a way to open sessions.
`manifold.benchmark` runs the benchmark's side.

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
import sys
import sysconfig
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import distribution
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from manifold.benchmark import WorkerEpisodeRecord
from manifold.core.benchmark import Benchmark
from manifold.core.check import check_compatibility
from manifold.core.pipeline import Pipeline
from manifold.core.state import DEFAULT_LANE, PipelineState
from manifold.core.verify import verify
from manifold.recipes.dispatch import assert_shared_profile, multi_pairing_pipeline
from manifold.wire import (
    BRIDGE_PROTOCOL_VERSION,
    FrameChannel,
    FrameType,
    HelloPayload,
    ReadyPayload,
    bridge,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from manifold.benchmark import ResetResult, StepResult
    from manifold.core.native_layout import NativeLayout
    from manifold.core.observation_space import ObservationSpace
    from manifold.core.policy import PolicySignature
    from manifold.core.values import Action, Observation
    from manifold.recipes.pairing import Pairing


def _emit(message: str) -> None:
    """Print one progress line, flushing so it appears under a redirected stdout.

    Python block-buffers a non-TTY stdout by default, which withholds progress
    when output is piped to a file; flushing here keeps a long-running serve loop
    or run from looking hung, in one place rather than at every call site. Callers
    with their own sink pass `on_event`.
    """
    print(message, flush=True)


def _run_episode_in_process(
    session: Session,
    pipeline: Pipeline,
    observation_source: ObservationSpace,
    signature: PolicySignature,
    state: PipelineState,
    reset: Callable[[], ResetResult],
    step: Callable[[Action], StepResult],
    max_steps: int,
    *,
    episode_idx: int,
    benchmark_name: str,
) -> WorkerEpisodeRecord:
    """Run one episode in process, and return its record.

    It resets the session and the lane's `PipelineState`, then calls `reset` and
    `step` around `_infer_step`. `_serve_session` calls `_infer_step` for each frame
    in the same way, over the bridge.
    """
    state.reset_lane(DEFAULT_LANE)
    session.reset()
    started_at = datetime.now(timezone.utc)
    reset_result = reset()
    observation, task_id = reset_result.observation, reset_result.task_id
    task_name = observation.instruction or benchmark_name
    steps = 0
    success = False
    initialization_sec = 0.0
    limit = reset_result.max_steps if reset_result.max_steps is not None else max_steps
    for _ in range(limit):
        action = _infer_step(session, observation, pipeline, observation_source, signature, state)
        if steps == 0:
            initialization_sec = (datetime.now(timezone.utc) - started_at).total_seconds()
        result = step(action)
        steps += 1
        observation = result.observation
        if result.success or result.done:
            success = result.success
            break
    return WorkerEpisodeRecord(
        episode_idx=episode_idx,
        task_name=task_name,
        success=success,
        steps=steps,
        initialization_sec=initialization_sec,
        started_at=started_at,
        ended_at=datetime.now(timezone.utc),
        task_id=task_id,
        attributes=reset_result.attributes,
    )


def _sdk_commit() -> str | None:
    """Return the git commit the installed manifold-sdk was built from.

    Users install the SDK from git pinned to a commit, so the commit identifies the
    SDK where the package version does not. An install from a local directory
    records no commit, and this returns None.
    """
    direct_url = distribution("manifold-sdk").read_text("direct_url.json")
    if direct_url is None:
        return None
    commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    return commit if isinstance(commit, str) else None


# The longest serving script that the Manifold backend stores, in characters.
MAX_SCRIPT_CHARS = 262_144

_SDK_ROOT = Path(__file__).resolve().parents[1]
_STDLIB_ROOT = Path(sysconfig.get_paths()["stdlib"]).resolve()


def _serving_script() -> dict[str, str] | None:
    """Return the path and text of the serving script.

    The serving script is the Python file with the `manifold.serve(...)` call.
    `manifold policy serve` uploads it, so the run page shows the code that
    served the run. This function takes the first frame on the stack outside
    the SDK and the standard library. A `manifold.serve(...)` call in
    `serve.py` gives `serve.py` for any command: `uv run serve.py`,
    `python -m my_policy.serve`, an entry point or a shell script.

    Returns None for code outside a file, as with `python -c`. Also returns
    None when the file is not UTF-8 text or is longer than the backend stores.
    The path is relative to the working directory when the file is inside it.
    """
    frame = sys._getframe(1)
    while frame is not None:
        path = Path(frame.f_code.co_filename)
        if path.is_file() and not _is_inside(path.resolve(), (_SDK_ROOT, _STDLIB_ROOT)):
            break
        frame = frame.f_back
    if frame is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if len(text) > MAX_SCRIPT_CHARS:
        return None
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    shown = resolved.relative_to(cwd) if resolved.is_relative_to(cwd) else resolved
    return {"path": shown.as_posix(), "text": text}


def _is_inside(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def _serve_connection(
    endpoint: PolicyEndpoint,
    pipeline: Pipeline | Callable[[Benchmark], Pipeline] | None,
    script: dict[str, str] | None,
    conn: socket.socket,
    addr: Any,
    emit: Callable[[str], None],
) -> None:
    """Own one runner connection end to end on its worker thread.

    Wraps `_serve_session` so any error in a single connection — a rejected
    pairing, a dropped socket, a malformed frame — is caught, logged, and confined
    here: it never propagates to the accept loop and so never kills the server or
    a sibling shard. The socket is always closed, even on error.

    The server does not log a connection that closes before its first frame,
    or a connection that only asks for the signature. The Manifold CLI opens
    both kinds when serve starts: the first to check that the server is up, and
    the second to read the signature.
    """
    try:
        with conn:
            spoke = _serve_session(endpoint, pipeline, script, conn, addr, emit)
        if spoke:
            emit(f"connection from {addr} closed")
    except Exception as exc:  # one shard's failure must not kill the server.
        emit(f"connection from {addr} errored, dropping it: {exc!r}")


def _received_benchmark(payload: dict[str, Any], emit: Callable[[str], None]) -> Benchmark:
    """Parse the advertised benchmark, logging any camera this SDK does not list."""
    benchmark = Benchmark.from_received(payload)
    unlisted = sorted(
        {str(sensor.get("name")) for sensor in payload.get("sensors", [])}
        - {camera.name for camera in benchmark.sensors}
    )
    if unlisted:
        emit(f"ignoring cameras this SDK does not list: {', '.join(unlisted)}")
    return benchmark


def _serve_session(
    endpoint: PolicyEndpoint,
    pipeline: Pipeline | Callable[[Benchmark], Pipeline] | None,
    script: dict[str, str] | None,
    conn: socket.socket,
    addr: Any,
    emit: Callable[[str], None],
) -> bool:
    """Run one runner connection: handshake, gate the pairing, then answer frames.

    Returns True when the peer sent a frame other than GET_SIGNATURE, and False
    otherwise.

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
    # A peer may send GET_SIGNATURE before HELLO. The CLI closes the connection
    # after the reply, and a benchmark sends HELLO next.
    while hello is not None and hello.get("type") == FrameType.GET_SIGNATURE:
        channel.send(
            FrameType.SIGNATURE,
            {
                "signature": endpoint.signature.model_dump(mode="json"),
                "sdk_version": _sdk_commit(),
                "script": script,
            },
        )
        hello = channel.recv()
    if hello is None:
        return False
    emit(f"benchmark connected from {addr}")
    if hello.get("type") != FrameType.HELLO:
        emit("expected a hello frame advertising the benchmark; closing")
        return True
    # A newer sender may carry fields that this server would drop without any error.
    # Refuse such a sender before parsing, rather than pair against a benchmark that
    # was misread. An older sender is safe, because a field that it omits takes its
    # default value here.
    version = hello["payload"].get("protocol_version", 0)
    if version > BRIDGE_PROTOCOL_VERSION:
        emit(
            f"benchmark uses bridge protocol {version}, newer than this server's "
            f"{BRIDGE_PROTOCOL_VERSION}; closing the connection rather than dropping "
            "fields that this server cannot read"
        )
        return True
    payload = HelloPayload.model_validate(hello["payload"])
    benchmark = _received_benchmark(payload.benchmark, emit)

    signature = endpoint.signature
    resolved = _resolve_pipeline(pipeline, benchmark)
    if not _gate(signature, benchmark, resolved, emit):
        return True

    # One PipelineState lives with this connection, not the endpoint, so a stateful
    # adapter's per-episode history never interleaves across concurrent connections.
    state = PipelineState()
    observation_source = benchmark.observation_space
    # Mint the session inside the try so its close() is guaranteed even if the
    # READY send itself fails (a peer that drops between gate and READY must not
    # leak the freshly-minted per-session scratch).
    session = endpoint.session()
    try:
        channel.send(
            FrameType.READY, ReadyPayload(protocol_version=BRIDGE_PROTOCOL_VERSION).model_dump()
        )
        while True:
            frame = channel.recv()
            if frame is None or frame.get("type") == FrameType.BYE:
                return True
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


@runtime_checkable
class Transport(Protocol):
    """The minimal frame surface that `serve` needs from a connection.

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


class ActionQueue:
    """One connection's open-loop chunk buffer over a shared endpoint.

    A reusable `Session` implementation for a policy backend to subclass: it owns its
    own chunk buffer (no shard shares it) and does not pack. Subclasses implement
    `_forward` (the backend's shared-model call) and override `reset`/`close` only if
    they hold per-session model state.
    """

    def __init__(self, endpoint: PolicyEndpoint) -> None:
        self._endpoint = endpoint
        self._execution_steps = endpoint.signature.execution_steps
        # The buffered raw model chunk and the step pointer into it. None until the
        # first advance; the pointer wraps at execution_steps, triggering a fresh
        # forward.
        self._chunk: Any = None
        self._step = 0

    def _forward(self, native: Any, /) -> Any:
        """Run the backend's shared-model forward on `native`; return the raw chunk.

        `native` is the packed native dict for a container policy, or the
        `Observation` for a prediction function.

        The one seam between backends. The base raises so a backend that leaves it
        unimplemented fails loudly rather than silently serving a stale buffer.
        """
        raise NotImplementedError("ActionQueue subclass must implement _forward")

    def advance(self, native: Any, /) -> tuple[Any, int]:
        """Serve the next open-loop step for `native`: return (raw_chunk, step).

        When the horizon has drained (no buffered chunk, or `execution_steps` served), run
        a fresh `_forward` and reset the step pointer; otherwise serve the next step.
        """
        if self._chunk is None or self._step >= self._execution_steps:
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


@dataclass(frozen=True)
class BenchmarkResult:
    """A finished run: one record per episode it drove.

    `episodes` and `successes` are views over `records`, so a tally cannot drift
    from the episodes it counts.
    """

    records: tuple[WorkerEpisodeRecord, ...] = ()

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


def serve(
    endpoint: PolicyEndpoint,
    *,
    pipeline: Pipeline | Callable[[Benchmark], Pipeline] | None = None,
    host: str = "127.0.0.1",
    port: int,
    # TODO: max_workers is now ignored, but kept for backwards compatibility. Remove it
    # after all existing callers no longer pass max_workers.
    max_workers: int = 8,
    on_event: Callable[[str], None] = _emit,
) -> None:
    """Serve one loaded policy to many runner shards concurrently over the bridge.

    The accept loop has no application-level connection limit. It starts one daemon
    thread per connection, with a separate session and `PipelineState`, so shards
    sharing the loaded model have isolated per-session state. `serve` does not set a
    deadline on the HELLO read, so a peer that sends nothing can block its worker
    indefinitely. Per connection the worker reads HELLO, gates the pairing
    (`check_compatibility` + `verify`), and only if both pass sends READY and answers
    each OBSERVATION with an ACTION until BYE. An endpoint that takes its own lock
    runs one forward at a time across connections; `serve` does not add another lock.
    One connection's failure is confined to its worker thread. A KeyboardInterrupt
    stops accepting and gives all in-flight workers one second in total to finish.

    `pipeline` may be None (empty pipeline), a callable invoked per connection with
    the benchmark (the dynamic-injection hook — the pipeline arrives with the job),
    or a `Pipeline` used as-is. `on_event` is called from worker threads, so a custom
    sink must be concurrency-safe.
    """
    emit = on_event
    script = _serving_script()
    workers: list[threading.Thread] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        # Request the platform's maximum backlog for connections that arrive between
        # accept iterations.
        listener.listen(socket.SOMAXCONN)
        emit(f"listening on {host}:{port} (TCP)")
        try:
            while True:
                conn, addr = listener.accept()
                worker = threading.Thread(
                    target=_serve_connection,
                    args=(endpoint, pipeline, script, conn, addr, emit),
                    daemon=True,
                )
                workers.append(worker)
                worker.start()
                # Drop references to workers that have finished.
                workers = [w for w in workers if w.is_alive()]
        except KeyboardInterrupt:
            emit("interrupted; shutting down")
    # The listener is closed; give all in-flight workers one second in total to
    # finish their current frame exchange. They are daemon threads, so the process
    # never hangs on a stuck connection.
    shutdown_deadline = time.monotonic() + 1.0
    for worker in workers:
        worker.join(timeout=max(0.0, shutdown_deadline - time.monotonic()))


def evaluate(
    endpoint: PolicyEndpoint,
    benchmark: Benchmark,
    reset: Callable[[], ResetResult],
    step: Callable[[Action], StepResult],
    *,
    pipeline: Pipeline | Callable[[Benchmark], Pipeline] | None = None,
    episodes: int,
    max_steps: int,
    on_event: Callable[[str], None] = _emit,
) -> BenchmarkResult:
    """Run a policy against a stand-in benchmark in process, with no server or bridge.

    It runs the same pairing check and the same step as `serve`, but the observation
    and the action stay in the process. A policy author runs it to test a pairing
    before serving the policy.

    It mints exactly one `session` (a single in-process lane), resets the session and
    that lane's `PipelineState` at each episode boundary, and `close`s the session in
    a `finally` (mirroring `serve`'s teardown) so even the in-process path retires
    per-session scratch deterministically. `reset` and `step` return the
    `ResetResult` and `StepResult` that a benchmark's hooks return. `pipeline`
    bridges exactly as `serve` resolves it (None / callable / `Pipeline`).

    Returns:
        A record for each episode run, and the tally over them.

    Raises:
        ValueError: If the policy's pairing check rejects the pairing. `serve` closes
            the connection without READY in the same case.
    """
    emit = on_event
    resolved = _resolve_pipeline(pipeline, benchmark)
    signature = endpoint.signature
    if not _gate(signature, benchmark, resolved, emit):
        raise ValueError("policy rejected the pairing (incompatible)")
    emit("policy confirmed the pairing (ready)")

    observation_source = benchmark.observation_space
    state = PipelineState()
    session = endpoint.session()
    try:
        records: list[WorkerEpisodeRecord] = []
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
    # TODO: max_workers is now ignored, but kept for backwards compatibility. Remove it
    # after all existing callers no longer pass max_workers.
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
        on_event=on_event,
    )


__all__ = [
    "ActionQueue",
    "BenchmarkResult",
    "PolicyEndpoint",
    "PolicyProfile",
    "Session",
    "Transport",
    "evaluate",
    "launch_server",
    "serve",
]
