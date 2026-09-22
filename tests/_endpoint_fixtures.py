"""Shared fixtures for the HTTP-serving and endpoint-client tests.

The pairing is deliberately small — one camera, one proprioception vector, an
end-effector action space both sides already agree on — so the observation and
action chains are empty and what the tests exercise is the packing phase and the
chunk cut, which is where the two serving paths could diverge.

The rest is the plumbing a test needs to call an ASGI app: one direct caller for
a single route, and one `http.server` host for a client that opens a real
connection.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pytest

from manifold.adapters.packing import PackToNativeLayout, UnpackFromNativeLayout
from manifold.core.benchmark import Benchmark
from manifold.core.conventions import GripperFormat, RotationFormat
from manifold.core.embodiment import (
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    GripperObservationSpec,
    Proprioception,
)
from manifold.core.native_layout import (
    DtypeCast,
    LayoutEntry,
    NativeLayout,
    Slice,
    SourceKind,
)
from manifold.core.pipeline import Pipeline
from manifold.core.policy import PolicySignature
from manifold.core.sensor import Camera
from manifold.core.values import Observation
from manifold.recipes import OpenLoopChunkQueue, Pairing, StepResult, read_pairing

# How many actions one forward answers with, on both serving paths.
EXEC_STEPS = 4

CAMERA_SHAPE = (8, 8, 3)


def action_space() -> EEActionSpace:
    """3 position + 3 axis-angle + 1 gripper = 7 dims per step."""
    return EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED, delta=True
    )


def proprioception() -> Proprioception:
    """3 position + 3 axis-angle + 2 gripper joints = 8 dims."""
    return Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.AXIS_ANGLE, gripper=GripperObservationSpec(dim=2)
        )
    )


def benchmark() -> Benchmark:
    """The benchmark both serving paths are driven against."""
    return Benchmark(
        name="suite",
        embodiment=Embodiment(name="arm", action=action_space(), proprioception=proprioception()),
        sensors=[Camera(name="agentview", shape=CAMERA_SHAPE)],
        instruction=True,
    )


def signature() -> PolicySignature:
    """A policy consuming exactly what the benchmark publishes."""
    return PolicySignature(
        action_space=action_space(),
        proprioception=proprioception(),
        cameras=[Camera(name="agentview", shape=CAMERA_SHAPE)],
        instruction=True,
    )


def input_layout() -> NativeLayout:
    """The model's native input dict: one image, one state vector, the prompt."""
    return NativeLayout(
        entries=(
            LayoutEntry(
                key="image",
                source=SourceKind.CAMERA,
                source_name="agentview",
                ops=(DtypeCast(dtype="uint8", contiguous=True),),
            ),
            LayoutEntry(
                key="state",
                source=SourceKind.STATE,
                source_name="ee_pose",
                ops=(DtypeCast(dtype="float32"),),
            ),
            LayoutEntry(key="prompt", source=SourceKind.INSTRUCTION),
        )
    )


def output_layout() -> NativeLayout:
    """A single-array output: the chunk indexed by step, windowed to 7 dims."""
    return NativeLayout(
        entries=(
            LayoutEntry(
                key="action", source=SourceKind.STATE, source_name=None, ops=(Slice(stop=7),)
            ),
        )
    )


def pipeline() -> Pipeline:
    """The packing phase alone: both convention chains are empty."""
    return Pipeline(
        pack=PackToNativeLayout(input_layout()),
        unpack=UnpackFromNativeLayout(output_layout(), action_space()),
    )


class FakeSession(OpenLoopChunkQueue):
    """The chunk queue `serve` mints per connection over `FakeEndpoint`."""

    _endpoint: FakeEndpoint

    def _forward(self, native: dict[str, Any], /) -> Any:
        return self._endpoint.forward(native)


class FakeEndpoint:
    """A loaded model stood in for by arithmetic over the native input.

    `forward` depends only on its argument, so the two serving paths compute the
    same chunk from the same observation and a comparison between them is about
    the paths rather than about the model.
    """

    def __init__(self) -> None:
        self.signature = signature()
        self.profile = SimpleNamespace(exec_steps=EXEC_STEPS, default_weights="DEFAULT")
        self.forwards = 0
        self.loaded_weights: list[str] = []

    def session(self) -> FakeSession:
        """Mint this connection's chunk queue over the shared model."""
        return FakeSession(self)

    def forward(self, native: dict[str, Any], /) -> Any:
        """Return an (EXEC_STEPS, 7) chunk derived from the native state vector."""
        self.forwards += 1
        seed = float(np.asarray(native["state"], dtype=np.float32).sum())
        base = np.arange(EXEC_STEPS * 7, dtype=np.float32).reshape(EXEC_STEPS, 7)
        return base + seed


def fake_pairing(endpoint: FakeEndpoint) -> Pairing:
    """A pairing whose profile loads `endpoint`, read through `read_pairing`.

    Built through `read_pairing` so the stand-in profile passes through the
    module's untyped attributes rather than being declared as a `PolicyProfile`.
    """
    profile = SimpleNamespace(
        signature=endpoint.signature,
        default_weights="DEFAULT",
        input_layout=input_layout(),
        output_layout=output_layout(),
        load=lambda weights, _device: endpoint.loaded_weights.append(weights) or endpoint,
    )
    module = cast(
        ModuleType,
        SimpleNamespace(
            __name__="tests.endpoint_pairing",
            PROFILE=profile,
            BENCHMARK=benchmark(),
            PIPELINE=pipeline(),
        ),
    )
    return read_pairing(module)


def observation(step: int) -> Observation:
    """The observation the fake environment publishes at `step`.

    Every channel varies with the step, so an action folded from the wrong
    observation differs from one folded from the right one.
    """
    return Observation(
        state={"ee_pose": np.full(8, float(step), dtype=np.float32)},
        sensors={"agentview": np.full(CAMERA_SHAPE, step % 256, dtype=np.uint8)},
        instruction="pick up the block",
    )


class FakeEnvironment:
    """A benchmark whose observations depend on the step alone, not on the action.

    Both serving paths therefore see the identical observation sequence, and the
    actions they answer with are comparable step by step.
    """

    def __init__(self) -> None:
        self.step_index = 0
        self.actions: list[np.ndarray] = []

    def reset(self) -> Observation:
        """Begin an episode at step 0."""
        self.step_index = 0
        return observation(0)

    def step(self, action: Any) -> StepResult:
        """Record the action and publish the next observation."""
        self.actions.append(np.asarray(action.values, dtype=np.float64).copy())
        self.step_index += 1
        return StepResult(observation=observation(self.step_index), success=False, done=False)


def silent(_message: str) -> None:
    """Discard a progress line, so a test's output stays its assertions."""


# --- calling an ASGI app ------------------------------------------------------------


def call_route(app: Any, method: str, path: str, body: bytes) -> tuple[int, bytes]:
    """Call one route directly, without a server in between."""
    events = [{"type": "http.request", "body": body, "more_body": False}]
    return asyncio.run(call_asgi(app, method, path, events))


async def call_asgi(
    app: Any,
    method: str,
    path: str,
    events: list[dict[str, Any]],
    root_path: str = "",
) -> tuple[int, bytes]:
    """Drive one HTTP scope through the app and collect its status and body.

    `root_path` is the mount prefix a host sets, which ASGI also leaves on
    `path`.
    """
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "root_path": root_path,
        "headers": [],
    }
    pending = list(events)
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return pending.pop(0)

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    await app(scope, receive, send)
    status = next(e["status"] for e in sent if e["type"] == "http.response.start")
    body = b"".join(e.get("body", b"") for e in sent if e["type"] == "http.response.body")
    return status, body


class ASGIRequestHandler(BaseHTTPRequestHandler):
    """Serve one ASGI app over `http.server`, one request at a time.

    Modal or uvicorn hosts the published app; this is the smallest stand-in that
    keeps a connection alive between requests, which is what the endpoint client
    reuses across steps.
    """

    protocol_version = "HTTP/1.1"
    asgi_app: ClassVar[Any]

    def do_GET(self) -> None:
        self._answer("GET")

    def do_POST(self) -> None:
        self._answer("POST")

    def _answer(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        events = [{"type": "http.request", "body": body, "more_body": False}]
        status, payload = asyncio.run(call_asgi(self.asgi_app, method, self.path, events))
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        """Discard the access log, which pytest would otherwise capture."""


@contextmanager
def asgi_server(app: Any):
    """Host `app` on a loopback port and yield its base URL."""

    class Handler(ASGIRequestHandler):
        asgi_app = staticmethod(app)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # A short poll keeps `shutdown` from waiting out the 0.5s default.
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    ).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def free_port() -> int:
    """A loopback port free at this moment, for a server about to bind it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def await_port(port: int, timeout: float = 10.0) -> None:
    """Block until something accepts on `port`, so a driver does not race the bind."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.02)
    pytest.fail(f"nothing accepted on port {port} within {timeout:.0f}s")
