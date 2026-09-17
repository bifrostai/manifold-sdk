"""The endpoint client: the policy worker of a run against a policy served at a URL.

An endpoint policy is a wrap the customer deployed at a URL and served with
`serve_http`. This module is the other half of that contract: an ordinary
container on a runner that holds the bridge session with each benchmark worker
and turns each observation into a `POST` at that URL. It does not carry a
pairing — the signature, the native layouts and the adapter chains all live with
the model, so an SDK release never rebuilds a customer's pipeline. (`recipes/
serving`'s `PolicyEndpoint` is the loaded model; the endpoint here is the URL.)

The open-loop chunk queue sits on this side. `serve_http` answers one
observation with a whole chunk, and each benchmark worker's connection pops one
action per step and refills the queue when it empties, so the wrap does not keep
per-episode state and a `/forward` that failed is safe to send again.

Configuration is read from the environment, and those names are a contract with
the backend that writes the container plan and with the image that runs this
module:

- `MANIFOLD_ENDPOINT_URL` — the wrap's base URL.
- `MANIFOLD_ENDPOINT_READINESS_WAIT_SECONDS` — how long `/healthz` is polled
  before the side fails.
- `MANIFOLD_ENDPOINT_FORWARD_BUDGET_SECONDS` — how long one observation may be
  retried for.

`MANIFOLD_ENDPOINT_SECRET` is reserved for the credential. The run's config route
redacts an `env` value only where the variable's name contains `secret`, so the
variable that carries a resolved value must be spelled that way. This cut does
not send a credential, and nothing reads that variable yet.

HTTP is `http.client` from the standard library: the SDK's dependencies do not
include an HTTP client, and a policy worker image should not gain one for three
routes.

Run it as `manifold-endpoint-client --port <port>`. `python -m
manifold.recipes.endpoint` reaches the same `main`, but `manifold/recipes/
__init__.py` imports this module before `runpy` executes it, so that form heads
every policy-side log with a `RuntimeWarning`.
"""

from __future__ import annotations

import argparse
import http.client
import os
import socket
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

from manifold.wire import FrameChannel, FrameType, bridge

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# The environment the backend writes into the container plan. The image that
# runs this module and the dispatch that builds the plan must agree on them.
URL_ENV = "MANIFOLD_ENDPOINT_URL"
READINESS_WAIT_ENV = "MANIFOLD_ENDPOINT_READINESS_WAIT_SECONDS"
FORWARD_BUDGET_ENV = "MANIFOLD_ENDPOINT_FORWARD_BUDGET_SECONDS"

# Keep the readiness wait under the runner's own container readiness deadline, so
# that the side ends on this client's non-zero exit and its stderr line rather
# than on the runner's probe, which reports only a port that never opened.
DEFAULT_READINESS_WAIT = timedelta(seconds=300)
DEFAULT_FORWARD_BUDGET = timedelta(seconds=500)

_READINESS_POLL = timedelta(seconds=2)
_INITIAL_RETRY_BACKOFF = timedelta(milliseconds=500)
_MAX_RETRY_BACKOFF = timedelta(seconds=10)

# The accept loop blocks for at most this long, so it reaches its failure check
# between connects rather than only when the next benchmark worker dials.
_ACCEPT_POLL = timedelta(milliseconds=500)

# The 4xx statuses a retry may clear: the wrap is busy, or asks for a later
# attempt. Every other 4xx is the wrap's answer about this pairing.
_RETRY_STATUSES = frozenset({408, 429})


class EndpointFailed(RuntimeError):
    """The endpoint policy did not serve this side, and the side ends here.

    One of three: `/healthz` never answered 200 within the readiness wait, the
    wrap refused the pairing at `/hello`, or a `/forward` ran out of its budget.
    The message is the single line the client writes to stderr before exiting
    non-zero, and the runner then reports the side as `worker_failed`.
    """


def _emit(message: str) -> None:
    """Print one progress line, flushing so it appears under a redirected stdout.

    The same sink `recipes/serving` prints through: a runner collects the
    worker's stdout, and a block-buffered stream would withhold the log of a run
    that is still going.
    """
    print(message, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the endpoint client as one run's policy worker; the process exit code.

    The entry point of the endpoint client image. It reads the URL and the two
    budgets from the environment, binds the bridge address given on the command
    line, and on failure writes one line to stderr and returns 1.
    """
    args = _parse_args(argv)
    url = os.environ.get(URL_ENV)
    if url is None:
        print(f"{URL_ENV} is unset", file=sys.stderr, flush=True)
        return 1
    try:
        serve_endpoint(
            url=url,
            host=args.host,
            port=args.port,
            readiness_wait=_duration(READINESS_WAIT_ENV, DEFAULT_READINESS_WAIT),
            forward_budget=_duration(FORWARD_BUDGET_ENV, DEFAULT_FORWARD_BUDGET),
            max_workers=args.max_workers,
        )
    except (EndpointFailed, ValueError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    return 0


def serve_endpoint(
    *,
    url: str,
    host: str = "0.0.0.0",
    port: int,
    readiness_wait: timedelta = DEFAULT_READINESS_WAIT,
    forward_budget: timedelta = DEFAULT_FORWARD_BUDGET,
    max_workers: int = 8,
    on_event: Callable[[str], None] = _emit,
) -> None:
    """Wait for the endpoint policy, then bridge every benchmark worker to it.

    `/healthz` is polled first, and the bridge address is bound only once it
    answers 200. A benchmark worker therefore never dials a cold endpoint, and
    the benchmark runner's own upstream budget does not have to cover a cold
    start. Each benchmark worker is then served on its own thread with its own
    chunk queue, the way `serve` serves a shard.

    This returns only on a KeyboardInterrupt; the runner stops the container when
    the side ends. `on_event` is called from worker threads, so a custom sink
    must be concurrency-safe.

    Raises:
        EndpointFailed: If `/healthz` never answered 200 within
            `readiness_wait`, if the wrap refused the pairing at `/hello`, or if
            a `/forward` ran out of `forward_budget`.
        ValueError: If `url` does not contain a host.
    """
    client = _EndpointClient(url, forward_budget=forward_budget, emit=on_event)
    client.wait_until_ready(readiness_wait)
    _accept_benchmark_workers(client, host=host, port=port, max_workers=max_workers, emit=on_event)


def _accept_benchmark_workers(
    client: _EndpointClient,
    *,
    host: str,
    port: int,
    max_workers: int,
    emit: Callable[[str], None],
) -> None:
    """Accept benchmark workers, until one of their sessions fails the side.

    `serve`'s accept loop, with one difference: a failure the endpoint caused
    ends the process rather than only the connection, because the wrap has
    answered for the whole side and another attempt would receive the same
    answer. The listener takes a timeout so the loop reaches that check between
    connects.
    """
    failures: list[EndpointFailed] = []
    slots = threading.BoundedSemaphore(max_workers)
    workers: list[threading.Thread] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(max(max_workers, 8))
        listener.settimeout(_ACCEPT_POLL.total_seconds())
        emit(f"listening on {host}:{port} (TCP), up to {max_workers} concurrent shard(s)")
        try:
            while True:
                if len(failures) > 0:
                    raise failures[0]
                slots.acquire()
                try:
                    conn, addr = listener.accept()
                except TimeoutError:
                    slots.release()
                    continue
                except BaseException:
                    # Nothing was spawned, so release the slot we took on the way in.
                    slots.release()
                    raise
                emit(f"benchmark connected from {addr}")
                worker = threading.Thread(
                    target=_serve_worker,
                    args=(client, conn, addr, slots, failures, emit),
                    daemon=True,
                )
                workers.append(worker)
                worker.start()
                # Drop references to finished workers so the list cannot grow
                # without bound across a long-lived run.
                workers = [w for w in workers if w.is_alive()]
        except KeyboardInterrupt:
            emit("interrupted; shutting down")
    for worker in workers:
        worker.join(timeout=1.0)


def _serve_worker(
    client: _EndpointClient,
    conn: socket.socket,
    addr: Any,
    slots: threading.BoundedSemaphore,
    failures: list[EndpointFailed],
    emit: Callable[[str], None],
) -> None:
    """Own one benchmark worker connection end to end on its own thread.

    A bridge-level error is confined here as it is in `serve`: a dropped socket
    or a malformed frame retires this connection alone. An `EndpointFailed` is
    recorded for the accept loop instead, since the wrap answered for the side.
    """
    try:
        with conn:
            _bridge_session(client, conn, emit)
        emit(f"connection from {addr} closed")
    except EndpointFailed as exc:
        # `list.append` is atomic, so the accept loop reads a complete entry.
        failures.append(exc)
        emit(str(exc))
    except Exception as exc:  # one shard's failure must not end the run.
        emit(f"connection from {addr} errored, dropping it: {exc!r}")
    finally:
        slots.release()


def _bridge_session(
    client: _EndpointClient,
    conn: socket.socket,
    emit: Callable[[str], None],
) -> None:
    """Run one benchmark worker's session: handshake, then answer every frame.

    `_serve_session`'s frame choreography, with the pairing gate forwarded to
    `/hello` and the chunk queue held here. An OBSERVATION is answered from the
    queue while it holds an action and refilled by one `/forward` when it is
    empty; RESET empties it at an episode boundary; BYE or a closed socket ends
    the session. The action payloads are sent on as they arrived, so a chunk
    reaches the benchmark in the bytes `serve` would have sent step by step.
    """
    channel = FrameChannel.from_socket(conn)
    hello = channel.recv()
    if hello is None or hello.get("type") != FrameType.HELLO:
        emit("expected a hello frame advertising the benchmark; closing")
        return
    client.hello(hello.get("payload", {}))
    channel.send(FrameType.READY, {})
    queue: list[dict[str, Any]] = []
    while True:
        frame = channel.recv()
        if frame is None or frame.get("type") == FrameType.BYE:
            return
        kind = frame.get("type")
        if kind == FrameType.RESET:
            queue.clear()
            continue
        if kind == FrameType.OBSERVATION:
            if len(queue) == 0:
                queue.extend(client.forward(frame.get("payload", {})))
            channel.send(FrameType.ACTION, queue.pop(0))
            continue
        emit(f"ignoring unexpected frame {kind!r}")


class _EndpointClient:
    """The three routes at the wrap's URL, with the readiness poll and retry rule.

    One instance serves every benchmark worker thread. An `http.client`
    connection is not thread-safe, so each thread opens its own on first use and
    reuses it across requests rather than paying a TLS handshake per step.
    """

    def __init__(
        self,
        url: str,
        *,
        forward_budget: timedelta,
        emit: Callable[[str], None],
    ) -> None:
        split = urllib.parse.urlsplit(url)
        if split.hostname is None:
            raise ValueError(f"endpoint URL does not contain a host: {url!r}")
        self._url = url.rstrip("/")
        self._https = split.scheme == "https"
        self._host = split.hostname
        self._port = split.port
        self._root = split.path.rstrip("/")
        self._budget = forward_budget
        self._emit = emit
        self._connections = threading.local()

    def wait_until_ready(self, readiness_wait: timedelta) -> None:
        """Poll `GET /healthz` until it answers 200, within the readiness wait.

        Each attempt is given the whole remaining wait: a host that is cold
        starting holds the request open until the process answers, and that held
        request is the readiness answer. A refused connection is polled again.

        Raises:
            EndpointFailed: If the wait ran out first. The message says how long
                was waited and what the last attempt returned.
        """
        wait = readiness_wait.total_seconds()
        deadline = time.monotonic() + wait
        reason = "no attempt has finished"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EndpointFailed(
                    f"{self._url}/healthz did not answer 200 within {wait:.0f}s ({reason})"
                )
            try:
                status, _answer, _headers = self._request("GET", "/healthz", None, remaining)
            except (OSError, http.client.HTTPException) as exc:
                reason = repr(exc)
            else:
                if status == 200:
                    self._emit(f"endpoint at {self._url} is ready")
                    return
                reason = f"HTTP {status}"
            time.sleep(min(_READINESS_POLL.total_seconds(), _left(deadline)))

    def hello(self, payload: dict[str, Any]) -> None:
        """Send a benchmark worker's HELLO frame to `/hello`; return once accepted.

        Raises:
            EndpointFailed: If the wrap refused the pairing, or if the route did
                not answer within the forward budget.
        """
        body = bridge.pack_http_frame(FrameType.HELLO, payload)
        status, _answer = self._post("/hello", body)
        if status == 404:
            raise EndpointFailed(
                f"{self._url}/hello is not a route the wrap serves; check the path of {URL_ENV}"
            )
        if status != 200:
            raise EndpointFailed(f"{self._url}/hello refused the pairing (HTTP {status})")

    def forward(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Send one observation to `/forward`; return the chunk's action payloads.

        The payloads stay encoded: the caller sends each one on as the ACTION
        frame of one step.

        Raises:
            EndpointFailed: If the wrap answered a status the retry rule does not
                clear, if the budget ran out, or if the chunk is undecodable or
                empty.
        """
        body = bridge.pack_http_frame(FrameType.OBSERVATION, payload)
        status, answer = self._post("/forward", body)
        if status != 200:
            raise EndpointFailed(f"{self._url}/forward failed (HTTP {status})")
        try:
            chunk = bridge.read_action_chunk(bridge.read_http_frame(answer, FrameType.ACTION))
        except ValueError as exc:
            raise EndpointFailed(
                f"{self._url}/forward answered an undecodable chunk: {exc}"
            ) from exc
        if len(chunk) == 0:
            raise EndpointFailed(f"{self._url}/forward answered an empty chunk")
        return chunk

    def _post(self, path: str, body: bytes) -> tuple[int, bytes]:
        """POST `body` to `path` within the forward budget; the status and answer.

        A connection error, a timeout, a 5xx, a 408 and a 429 are attempted again
        with exponential backoff, and a `Retry-After` header replaces that
        backoff. Every other status is the wrap's own answer and is returned at
        once for the caller to act on. Each attempt is given the whole remaining
        budget, so a serverless host recycling its container mid-run costs one
        cold start inside the budget instead of a failed run.

        Raises:
            EndpointFailed: If the budget ran out while retrying. The message
                says which route and what the last attempt returned.
        """
        budget = self._budget.total_seconds()
        deadline = time.monotonic() + budget
        backoff = _INITIAL_RETRY_BACKOFF
        reason = "no attempt has finished"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EndpointFailed(
                    f"{self._url}{path} did not answer within {budget:.0f}s ({reason})"
                )
            try:
                status, answer, headers = self._request("POST", path, body, remaining)
            except (OSError, http.client.HTTPException) as exc:
                reason = repr(exc)
                delay = backoff
            else:
                if status < 500 and status not in _RETRY_STATUSES:
                    return status, answer
                reason = f"HTTP {status}"
                delay = _retry_after(headers, backoff)
            self._emit(f"retrying {path} after {reason}")
            time.sleep(min(delay.total_seconds(), _left(deadline)))
            backoff = min(backoff * 2, _MAX_RETRY_BACKOFF)

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes, http.client.HTTPMessage]:
        """Send one request on this thread's connection and read the whole answer.

        The connection is closed on any failure, so the next attempt opens a
        fresh one instead of reusing a socket the peer may already have retired.
        """
        connection = self._connection(timeout)
        headers = {"Content-Type": bridge.HTTP_MEDIA_TYPE} if body is not None else {}
        try:
            connection.request(method, f"{self._root}{path}", body=body, headers=headers)
            response = connection.getresponse()
            answer = response.read()
        except BaseException:
            self._drop()
            raise
        return response.status, answer, response.headers

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        """This thread's connection, opened on first use and retimed per attempt.

        `HTTPConnection` reads its timeout when it connects, so an already-open
        socket is retimed directly; otherwise a later attempt with a shorter
        remaining budget would keep the first attempt's deadline.
        """
        connection = getattr(self._connections, "connection", None)
        if connection is None:
            kind = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
            connection = kind(self._host, self._port, timeout=timeout)
            self._connections.connection = connection
        connection.timeout = timeout
        if connection.sock is not None:
            connection.sock.settimeout(timeout)
        return connection

    def _drop(self) -> None:
        """Close and forget this thread's connection after a failed exchange."""
        connection = getattr(self._connections, "connection", None)
        if connection is not None:
            connection.close()
            self._connections.connection = None


def _retry_after(headers: http.client.HTTPMessage, fallback: timedelta) -> timedelta:
    """The delay a `Retry-After` header asks for, or `fallback` where it is absent.

    The header carries either a number of seconds or an HTTP date; a date is
    converted to the delay from now. An unreadable value is ignored in favour of
    the caller's backoff.
    """
    raw = headers.get("Retry-After")
    if raw is None:
        return fallback
    try:
        return max(timedelta(seconds=float(raw)), timedelta(0))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return fallback
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(when - datetime.now(timezone.utc), timedelta(0))


def _left(deadline: float) -> float:
    """The seconds left before `deadline`, floored at zero.

    `time.sleep` is one of the boundaries that takes a number rather than a
    duration, and a monotonic deadline that has passed would otherwise sleep
    backwards.
    """
    return max(deadline - time.monotonic(), 0.0)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the bridge address this client binds and its concurrency cap."""
    parser = argparse.ArgumentParser(
        prog="manifold-endpoint-client",
        description="Forward a benchmark's observations to a policy served at a URL.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Network interface to bind (default: 0.0.0.0).",
    )
    parser.add_argument("--port", type=int, required=True, help="TCP port to bind.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of benchmark workers served concurrently (default: 8).",
    )
    return parser.parse_args(argv)


def _duration(name: str, default: timedelta) -> timedelta:
    """Read one environment variable as a number of seconds, or return `default`.

    The environment is the boundary the seconds are spelled at, and each
    variable's name says so; every reader past this point takes a `timedelta`.

    Raises:
        ValueError: If the variable is set to something other than a number.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return timedelta(seconds=float(raw))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of seconds, got {raw!r}") from exc


__all__ = [
    "DEFAULT_FORWARD_BUDGET",
    "DEFAULT_READINESS_WAIT",
    "FORWARD_BUDGET_ENV",
    "READINESS_WAIT_ENV",
    "URL_ENV",
    "EndpointFailed",
    "main",
    "serve_endpoint",
]


if __name__ == "__main__":
    raise SystemExit(main())
