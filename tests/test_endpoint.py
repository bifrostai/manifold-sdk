"""The endpoint client: the readiness poll, the retry rule, and the failure line.

Each test runs the real `serve_endpoint` against a scripted endpoint and drives
it from a benchmark worker's end of the bridge, so what is asserted is the
client's behaviour rather than the shape of a private helper.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import numpy as np
import pytest

from manifold.core.values import Action
from manifold.recipes import endpoint as endpoint_module
from manifold.recipes import serve_endpoint
from manifold.wire import FrameChannel, FrameType, bridge
from tests._endpoint_fixtures import benchmark, free_port, observation, silent

OK = (200, {}, b"")


@pytest.fixture(autouse=True)
def _short_waits(monkeypatch):
    """Shorten the poll and backoff constants so the suite does not wait on them."""
    monkeypatch.setattr(endpoint_module, "_READINESS_POLL", timedelta(milliseconds=10))
    monkeypatch.setattr(endpoint_module, "_INITIAL_RETRY_BACKOFF", timedelta(milliseconds=10))
    monkeypatch.setattr(endpoint_module, "_MAX_RETRY_BACKOFF", timedelta(milliseconds=50))
    monkeypatch.setattr(endpoint_module, "_ACCEPT_POLL", timedelta(milliseconds=20))


# --- configuration from the environment ---------------------------------------------


def test_the_environment_carries_the_two_budgets_as_seconds(monkeypatch):
    captured = _intercept_serve_endpoint(monkeypatch)
    monkeypatch.setenv(endpoint_module.READINESS_WAIT_ENV, "12")
    monkeypatch.setenv(endpoint_module.FORWARD_BUDGET_ENV, "34.5")

    assert endpoint_module.main(["--port", "0"]) == 0

    assert captured["readiness_wait"] == timedelta(seconds=12)
    assert captured["forward_budget"] == timedelta(seconds=34.5)


def test_the_defaults_apply_where_the_environment_sets_neither(monkeypatch):
    captured = _intercept_serve_endpoint(monkeypatch)
    monkeypatch.delenv(endpoint_module.READINESS_WAIT_ENV, raising=False)
    monkeypatch.delenv(endpoint_module.FORWARD_BUDGET_ENV, raising=False)

    assert endpoint_module.main(["--port", "0"]) == 0

    assert captured["readiness_wait"] == endpoint_module.DEFAULT_READINESS_WAIT
    assert captured["forward_budget"] == endpoint_module.DEFAULT_FORWARD_BUDGET


def test_a_duration_that_is_not_a_number_fails_with_one_line(monkeypatch, capsys):
    _intercept_serve_endpoint(monkeypatch)
    monkeypatch.setenv(endpoint_module.READINESS_WAIT_ENV, "five minutes")

    assert endpoint_module.main(["--port", "0"]) == 1

    assert capsys.readouterr().err.splitlines() == [
        f"{endpoint_module.READINESS_WAIT_ENV} must be a number of seconds, got 'five minutes'"
    ]


def test_an_unset_url_fails_with_one_line(monkeypatch, capsys):
    _intercept_serve_endpoint(monkeypatch)
    monkeypatch.delenv(endpoint_module.URL_ENV, raising=False)

    assert endpoint_module.main(["--port", "0"]) == 1

    assert capsys.readouterr().err.splitlines() == [f"{endpoint_module.URL_ENV} is unset"]


def _intercept_serve_endpoint(monkeypatch) -> dict[str, Any]:
    """Set a usable URL and collect what `main` would have served with."""
    captured: dict[str, Any] = {}
    monkeypatch.setenv(endpoint_module.URL_ENV, "http://127.0.0.1:1")
    monkeypatch.setattr(endpoint_module, "serve_endpoint", lambda **kwargs: captured.update(kwargs))
    return captured


# --- the readiness poll -------------------------------------------------------------


def test_the_bridge_is_bound_only_after_healthz_answers_200():
    script = _Script(healthz=[(503, {}, b""), (503, {}, b""), OK])
    with _client(script) as client:
        _one_step(client)
    assert script.paths[:3] == ["/healthz", "/healthz", "/healthz"]


def test_a_readiness_wait_that_runs_out_fails_the_side_with_one_line():
    script = _Script(healthz=[(503, {}, b"")])
    with _client(script, readiness_wait=timedelta(milliseconds=200)) as client:
        failure = client.await_failure()
    assert "/healthz did not answer 200 within" in str(failure)
    assert "HTTP 503" in str(failure)


def test_no_benchmark_worker_can_dial_before_the_endpoint_is_ready():
    # The port is bound after the poll, so a dial during a cold start is refused
    # rather than accepted and then left waiting.
    script = _Script(healthz=[(503, {}, b"")])
    with _client(script, readiness_wait=timedelta(milliseconds=400)) as client:
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", client.port), timeout=0.2).close()
        client.await_failure()


# --- the retry rule -----------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 502, 503, 408, 429])
def test_a_retryable_status_is_attempted_again_within_the_budget(status):
    script = _Script(forward=[(status, {}, b""), (200, {}, _chunk_body(1))])
    with _client(script) as client:
        action = _one_step(client)
    assert action is not None
    assert script.paths.count("/forward") == 2


@pytest.mark.parametrize("status", [400, 403, 404, 409, 422])
def test_another_4xx_is_the_wraps_answer_and_fails_the_side_at_once(status):
    script = _Script(forward=[(status, {}, b"")])
    with _client(script) as client:
        assert _one_step(client) is None
        failure = client.await_failure()
    assert str(failure) == f"{client.url}/forward failed (HTTP {status})"
    assert script.paths.count("/forward") == 1


def test_a_connection_error_is_attempted_again():
    script = _Script(forward=[_CLOSE, (200, {}, _chunk_body(1))])
    with _client(script) as client:
        assert _one_step(client) is not None
    assert script.paths.count("/forward") == 2


def test_a_forward_budget_that_runs_out_fails_the_side_with_its_last_status():
    script = _Script(forward=[(500, {}, b"")])
    with _client(script, budget=timedelta(milliseconds=200)) as client:
        assert _one_step(client) is None
        failure = client.await_failure()
    assert "/forward did not answer within" in str(failure)
    assert "HTTP 500" in str(failure)


def test_retry_after_replaces_the_backoff():
    script = _Script(forward=[(429, {"Retry-After": "1"}, b""), (200, {}, _chunk_body(1))])
    with _client(script) as client:
        started = time.monotonic()
        assert _one_step(client) is not None
        waited = time.monotonic() - started
    # The backoff is a hundredth of a second here, so only the header explains a
    # wait of about a second.
    assert waited >= 0.9


# --- the handshake and the chunk queue ----------------------------------------------


def test_a_refused_hello_closes_the_connection_and_fails_the_side():
    script = _Script(hello=[(409, {}, b"")])
    with _client(script) as client:
        channel = client.connect()
        channel.send(FrameType.HELLO, _hello_payload())
        assert channel.recv() is None
        failure = client.await_failure()
    assert str(failure) == f"{client.url}/hello refused the pairing (HTTP 409)"


def test_one_forward_fills_the_queue_for_the_whole_chunk():
    script = _Script(forward=[(200, {}, _chunk_body(3))])
    with _client(script) as client:
        channel = client.connect()
        channel.send(FrameType.HELLO, _hello_payload())
        assert channel.recv()["type"] == FrameType.READY
        for _ in range(3):
            assert _observe(channel) is not None
    assert script.paths.count("/forward") == 1


def test_a_reset_empties_the_queue_at_an_episode_boundary():
    script = _Script(forward=[(200, {}, _chunk_body(3))])
    with _client(script) as client:
        channel = client.connect()
        channel.send(FrameType.HELLO, _hello_payload())
        channel.recv()
        assert _observe(channel) is not None
        channel.send(FrameType.RESET, {})
        assert _observe(channel) is not None
    assert script.paths.count("/forward") == 2


def test_an_empty_chunk_fails_the_side():
    script = _Script(forward=[(200, {}, _chunk_body(0))])
    with _client(script) as client:
        assert _one_step(client) is None
        failure = client.await_failure()
    assert "answered an empty chunk" in str(failure)


# --- the scripted endpoint ----------------------------------------------------------

# A scripted answer that closes the connection without replying, which the client
# sees as a connection error.
_CLOSE = (0, {}, b"")


@dataclass
class _Script:
    """The answers a scripted endpoint gives per route, in order.

    Each route's list is consumed one answer per request, and the last answer
    repeats once the list runs out, so a test states only the answers that differ.
    """

    healthz: list[tuple[int, dict[str, str], bytes]] = field(default_factory=lambda: [OK])
    hello: list[tuple[int, dict[str, str], bytes]] = field(default_factory=lambda: [OK])
    forward: list[tuple[int, dict[str, str], bytes]] = field(default_factory=lambda: [OK])
    paths: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self, path: str) -> tuple[int, dict[str, str], bytes]:
        """Record the request and return the answer this route is due."""
        with self._lock:
            self.paths.append(path)
            answers = {
                "/healthz": self.healthz,
                "/hello": self.hello,
                "/forward": self.forward,
            }[path]
            return answers.pop(0) if len(answers) > 1 else answers[0]


class _ScriptedHandler(BaseHTTPRequestHandler):
    """Answer each request from the script, or close without answering."""

    protocol_version = "HTTP/1.1"
    script: ClassVar[_Script]

    def do_GET(self) -> None:
        self._answer()

    def do_POST(self) -> None:
        self._answer()

    def _answer(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        status, headers, body = self.script.take(self.path)
        if status == 0:
            self.close_connection = True
            return
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Discard the access log, which pytest would otherwise capture."""


@dataclass
class _RunningClient:
    """A `serve_endpoint` under test, with the endpoint it was pointed at."""

    url: str
    port: int
    failures: list[BaseException]
    thread: threading.Thread

    def connect(self) -> FrameChannel:
        """Open a benchmark worker's end of the bridge."""
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                return FrameChannel.from_socket(
                    socket.create_connection(("127.0.0.1", self.port), timeout=10.0)
                )
            except OSError:
                time.sleep(0.02)
        pytest.fail(f"nothing accepted on port {self.port}")

    def await_failure(self) -> BaseException:
        """Block until `serve_endpoint` returns, and hand back what ended it."""
        self.thread.join(timeout=15.0)
        assert len(self.failures) == 1, f"serve_endpoint did not fail: {self.failures}"
        return self.failures[0]


@contextmanager
def _client(
    script: _Script,
    *,
    budget: timedelta = timedelta(seconds=5),
    readiness_wait: timedelta = timedelta(seconds=10),
):
    """Host the script and run `serve_endpoint` against it on its own thread."""

    class Handler(_ScriptedHandler):
        pass

    Handler.script = script
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # A short poll keeps `shutdown` from waiting out the 0.5s default.
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    ).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    port = free_port()
    failures: list[BaseException] = []

    def run() -> None:
        try:
            serve_endpoint(
                url=url,
                host="127.0.0.1",
                port=port,
                readiness_wait=readiness_wait,
                forward_budget=budget,
                on_event=silent,
            )
        except BaseException as exc:  # recorded for the test to assert on.
            failures.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield _RunningClient(url=url, port=port, failures=failures, thread=thread)
    finally:
        server.shutdown()
        server.server_close()


# --- driving one benchmark worker ---------------------------------------------------


def _one_step(client: _RunningClient) -> dict[str, Any] | None:
    """Hand shake and send one observation; the ACTION frame, or None on a close."""
    channel = client.connect()
    channel.send(FrameType.HELLO, _hello_payload())
    ready = channel.recv()
    if ready is None or ready.get("type") != FrameType.READY:
        return None
    return _observe(channel)


def _observe(channel: FrameChannel) -> dict[str, Any] | None:
    """Send one observation and read the action frame, or None on a close."""
    channel.send(FrameType.OBSERVATION, bridge.encode_observation(observation(0)))
    frame = channel.recv()
    if frame is None or frame.get("type") != FrameType.ACTION:
        return None
    return frame


def _hello_payload() -> dict[str, Any]:
    """The HELLO payload a benchmark worker advertises."""
    return {"benchmark": benchmark().model_dump(mode="json")}


def _chunk_body(actions: int) -> bytes:
    """A `/forward` answer carrying `actions` steps of one chunk."""
    chunk = [
        Action.from_array(np.full(7, float(step), dtype=np.float32)) for step in range(actions)
    ]
    return bridge.pack_http_frame(FrameType.ACTION, bridge.encode_action_chunk(chunk))
