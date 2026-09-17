"""The two serving paths answer one observation sequence with identical actions.

The check the customer-hosted endpoint plan requires: the same pairing served by
`launch_server` and by `serve_http` must produce the same actions, step for step.
It is the only check that catches a codec or chunk-cut divergence between the
paths, so it drives the real endpoint client against the real ASGI app rather
than calling the fold twice in process.
"""

from __future__ import annotations

import threading

import numpy as np

from manifold.core.benchmark import Benchmark
from manifold.recipes import launch_server, run_benchmark, serve_endpoint, serve_http
from tests._endpoint_fixtures import (
    EXEC_STEPS,
    FakeEndpoint,
    FakeEnvironment,
    asgi_server,
    await_port,
    benchmark,
    fake_pairing,
    free_port,
    silent,
)

EPISODES = 2
MAX_STEPS = 6


def test_the_two_serving_paths_answer_with_identical_actions():
    bench = benchmark()

    over_bridge = _drive_over_bridge(bench, FakeEndpoint())
    over_http = _drive_over_http(bench, FakeEndpoint())

    assert len(over_bridge) == EPISODES * MAX_STEPS
    assert len(over_http) == len(over_bridge)
    for step, (served, endpointed) in enumerate(zip(over_bridge, over_http, strict=True)):
        np.testing.assert_array_equal(served, endpointed, err_msg=f"step {step}")


def test_each_path_runs_the_model_once_per_chunk():
    # The chunk cut is what makes the actions above comparable: the paths must
    # forward on the same steps, not merely end up with the same values.
    bench = benchmark()
    over_bridge = FakeEndpoint()
    over_http = FakeEndpoint()

    _drive_over_bridge(bench, over_bridge)
    _drive_over_http(bench, over_http)

    chunks_per_episode = -(-MAX_STEPS // EXEC_STEPS)
    assert over_bridge.forwards == EPISODES * chunks_per_episode
    assert over_http.forwards == over_bridge.forwards


def _drive_over_bridge(bench: Benchmark, endpoint: FakeEndpoint) -> list[np.ndarray]:
    """Run the benchmark against `launch_server`: the bridge serving path."""
    port = free_port()
    threading.Thread(
        target=launch_server,
        args=([fake_pairing(endpoint)],),
        kwargs={"host": "127.0.0.1", "port": port, "on_event": silent},
        daemon=True,
    ).start()
    await_port(port)
    return _run(bench, port)


def _drive_over_http(bench: Benchmark, endpoint: FakeEndpoint) -> list[np.ndarray]:
    """Run the benchmark against `serve_http` through the endpoint client."""
    app = serve_http(fake_pairing(endpoint), on_event=silent)
    with asgi_server(app) as url:
        port = free_port()
        threading.Thread(
            target=serve_endpoint,
            kwargs={"url": url, "host": "127.0.0.1", "port": port, "on_event": silent},
            daemon=True,
        ).start()
        await_port(port)
        return _run(bench, port)


def _run(bench: Benchmark, port: int) -> list[np.ndarray]:
    """Drive the fake environment against whatever is listening on `port`."""
    environment = FakeEnvironment()
    run_benchmark(
        bench,
        environment.reset,
        environment.step,
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        host="127.0.0.1",
        port=port,
        on_event=silent,
    )
    return environment.actions
