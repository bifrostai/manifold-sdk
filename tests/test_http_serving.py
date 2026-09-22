"""`serve_http`'s three routes, called directly as an ASGI app.

Each test drives one scope through the app the way a host would, so what is
asserted is the status and body a caller reads, plus the reasons a refusal
leaves in the server's own log.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, ClassVar

from pydantic import BaseModel

from manifold.core.adapter import ObservationAdapter
from manifold.core.observation_space import ObservationSpace
from manifold.core.pipeline import Pipeline
from manifold.recipes import serve_http
from manifold.wire import FrameType, bridge
from tests._endpoint_fixtures import (
    EXEC_STEPS,
    FakeEndpoint,
    benchmark,
    call_asgi,
    call_route,
    fake_pairing,
    observation,
    silent,
)


def test_the_model_is_loaded_before_the_app_is_returned():
    endpoint = FakeEndpoint()
    serve_http(fake_pairing(endpoint), weights="CHECKPOINT", on_event=silent)
    assert endpoint.loaded_weights == ["CHECKPOINT"]


def test_the_profiles_default_weights_are_loaded_when_none_are_given():
    endpoint = FakeEndpoint()
    serve_http(fake_pairing(endpoint), on_event=silent)
    assert endpoint.loaded_weights == ["DEFAULT"]


def test_healthz_answers_200_because_the_app_exists_only_once_loaded():
    app = serve_http(fake_pairing(FakeEndpoint()), on_event=silent)
    assert call_route(app, "GET", "/healthz", b"") == (200, b"")


def test_hello_accepts_the_pairing_the_wrap_was_built_for():
    app = serve_http(fake_pairing(FakeEndpoint()), on_event=silent)
    assert call_route(app, "POST", "/hello", _hello_body())[0] == 200


def test_hello_refuses_an_incompatible_benchmark_without_a_body_to_read():
    lines: list[str] = []
    app = serve_http(fake_pairing(FakeEndpoint()), on_event=lines.append)
    other = benchmark().model_copy(update={"instruction": False})

    status, body = call_route(app, "POST", "/hello", _hello_body(other))

    assert (status, body) == (409, b"")
    assert any("rejecting the pairing" in line for line in lines)


def test_hello_refuses_a_stateful_adapter_it_cannot_carry_across_requests():
    lines: list[str] = []
    pairing = fake_pairing(FakeEndpoint())
    stateful = Pipeline(
        observation=(_StatefulPassthrough(),),
        pack=pairing.pipeline.pack,
        unpack=pairing.pipeline.unpack,
    )
    app = serve_http(replace(pairing, pipeline=stateful), on_event=lines.append)

    status, body = call_route(app, "POST", "/hello", _hello_body())

    assert (status, body) == (409, b"")
    assert any("_StatefulPassthrough" in line for line in lines)


def test_hello_refuses_an_undecodable_body_as_the_callers_own_bug():
    app = serve_http(fake_pairing(FakeEndpoint()), on_event=silent)
    assert call_route(app, "POST", "/hello", b"not a frame")[0] == 400


def test_forward_answers_with_a_whole_chunk():
    app = serve_http(fake_pairing(FakeEndpoint()), on_event=silent)

    status, body = call_route(app, "POST", "/forward", _observation_body())

    assert status == 200
    chunk = bridge.read_action_chunk(bridge.read_http_frame(body, FrameType.ACTION))
    assert len(chunk) == EXEC_STEPS
    assert bridge.decode_action(chunk[0]).values.shape == (7,)


def test_forward_answers_a_repeated_request_with_the_same_chunk():
    # Nothing per-episode is kept between requests, which is what makes a
    # /forward that failed safe for the caller to send again.
    app = serve_http(fake_pairing(FakeEndpoint()), on_event=silent)
    body = _observation_body()

    first = _chunk_of(call_route(app, "POST", "/forward", body)[1])
    again = _chunk_of(call_route(app, "POST", "/forward", body)[1])

    assert first == again


def test_forward_refuses_an_undecodable_body():
    app = serve_http(fake_pairing(FakeEndpoint()), on_event=silent)
    assert call_route(app, "POST", "/forward", b"not a frame")[0] == 400


def test_a_failed_forward_answers_500_so_the_caller_may_repeat_it():
    app = serve_http(fake_pairing(_BrokenEndpoint()), on_event=silent)
    assert call_route(app, "POST", "/forward", _observation_body())[0] == 500


def test_an_unknown_route_answers_404():
    app = serve_http(fake_pairing(FakeEndpoint()), on_event=silent)
    assert call_route(app, "GET", "/metrics", b"")[0] == 404


def test_every_route_answers_under_a_mount_prefix():
    # A host that mounts the app leaves its prefix on `path` as well as on
    # `root_path`, and the client dials the path of its own URL.
    app = serve_http(fake_pairing(FakeEndpoint()), on_event=silent)

    healthz = asyncio.run(call_asgi(app, "GET", "/wrap/healthz", [_one(b"")], root_path="/wrap"))
    forward = asyncio.run(
        call_asgi(
            app,
            "POST",
            "/wrap/forward",
            [_one(_observation_body())],
            root_path="/wrap",
        )
    )

    assert healthz == (200, b"")
    assert forward[0] == 200


def test_forward_refuses_a_stateful_adapter_as_hello_does():
    # The refusal belongs to the loaded pipeline, and a stateless /forward
    # carries nothing that says whether this caller is the one hello refused.
    lines: list[str] = []
    pairing = fake_pairing(FakeEndpoint())
    stateful = Pipeline(
        observation=(_StatefulPassthrough(),),
        pack=pairing.pipeline.pack,
        unpack=pairing.pipeline.unpack,
    )
    app = serve_http(replace(pairing, pipeline=stateful), on_event=lines.append)

    status, body = call_route(app, "POST", "/forward", _observation_body())

    assert (status, body) == (409, b"")
    assert any("_StatefulPassthrough" in line for line in lines)


def test_a_body_over_the_frame_cap_is_refused_before_it_is_held():
    lines: list[str] = []
    app = serve_http(fake_pairing(FakeEndpoint()), on_event=lines.append)
    events = [
        {"type": "http.request", "body": b"x" * (1024 * 1024), "more_body": True}
        for _ in range(bridge.MAX_FRAME_BYTES // (1024 * 1024) + 1)
    ]

    status, body = asyncio.run(call_asgi(app, "POST", "/forward", events))

    assert (status, body) == (400, b"")
    assert any("request body is over" in line for line in lines)


def _one(body: bytes) -> dict[str, Any]:
    return {"type": "http.request", "body": body, "more_body": False}


def test_a_body_split_across_chunks_is_read_whole():
    app = serve_http(fake_pairing(FakeEndpoint()), on_event=silent)
    body = _observation_body()
    events = [
        {"type": "http.request", "body": body[:10], "more_body": True},
        {"type": "http.request", "body": body[10:], "more_body": False},
    ]
    assert asyncio.run(call_asgi(app, "POST", "/forward", events))[0] == 200


def test_the_lifespan_scope_is_acknowledged_so_a_host_routes_requests():
    app = serve_http(fake_pairing(FakeEndpoint()), on_event=silent)
    assert asyncio.run(_run_lifespan(app)) == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]


# --- helpers ------------------------------------------------------------------------


async def _run_lifespan(app: Any) -> list[str]:
    """Drive a lifespan scope through the app and collect what it acknowledged."""
    pending = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
    sent: list[str] = []

    async def receive() -> dict[str, Any]:
        return pending.pop(0)

    async def send(event: dict[str, Any]) -> None:
        sent.append(event["type"])

    await app({"type": "lifespan"}, receive, send)
    return sent


class _StatefulPassthrough(ObservationAdapter):
    """A stateful adapter that changes nothing, to exercise the `/hello` refusal."""

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True
    state_key: ClassVar[str | None] = "history"

    def applies(self, source: BaseModel) -> bool:
        """Never resolved into a chain; the pairing's author inserts it."""
        return False

    def produce(self, source: BaseModel) -> BaseModel:
        """The observation space is unchanged."""
        return source

    def adapt(self, data: Any, /, *, source: BaseModel, state: dict[str, Any] | None = None) -> Any:
        """Return the observation untouched."""
        return data


class _BrokenEndpoint(FakeEndpoint):
    """An endpoint whose model raises, to exercise the 500 answer."""

    def forward(self, native: dict[str, Any], /) -> Any:
        """Fail the way a model out of device memory would."""
        raise RuntimeError("the GPU fell over")


def _chunk_of(body: bytes) -> list[dict[str, Any]]:
    """The action payloads of a `/forward` answer.

    Read rather than compared byte for byte: `pack_frame` stamps every frame with
    the moment it was written, so two identical chunks differ in that field.
    """
    return bridge.read_action_chunk(bridge.read_http_frame(body, FrameType.ACTION))


def _hello_body(bench: Any = None) -> bytes:
    """A HELLO frame body advertising `bench`, defaulting to the fixture."""
    return bridge.pack_http_frame(
        FrameType.HELLO, {"benchmark": (bench or benchmark()).model_dump(mode="json")}
    )


def _observation_body(step: int = 0) -> bytes:
    """An OBSERVATION frame body carrying the fixture's step-`step` observation."""
    return bridge.pack_http_frame(
        FrameType.OBSERVATION, bridge.encode_observation(observation(step))
    )
