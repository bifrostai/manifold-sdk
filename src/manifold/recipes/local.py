"""Start a local policy server."""

from __future__ import annotations

import os
from collections.abc import Callable

from manifold.adapters.observation.camera_resolution import ResizeCameras
from manifold.adapters.observation.camera_rotate_180 import Rotate180Cameras
from manifold.adapters.observation.camera_vertical_flip import FlipVerticalCameras
from manifold.adapters.observation.proprio_rotation import ProprioRotationAdapter
from manifold.core.adapter import ActionAdapter, ObservationAdapter
from manifold.core.benchmark import Benchmark
from manifold.core.pipeline import Pipeline
from manifold.core.policy import PolicySignature
from manifold.core.values import Action, Observation
from manifold.recipes.function import FunctionEndpoint
from manifold.recipes.resolve import resolve
from manifold.recipes.serving import serve as serve_endpoint


def _server_address(server: str) -> tuple[str, int]:
    value = server.removeprefix("tcp:")
    host, separator, port = value.rpartition(":")
    if not host or not separator or not port:
        raise ValueError("MANIFOLD_SERVER_URL must use tcp:HOST:PORT")
    return host, int(port)


def _default_pool(
    signature: PolicySignature,
    benchmark: Benchmark,
) -> list[ActionAdapter | ObservationAdapter]:
    """Build the built-in adapters that could bridge `benchmark` to `signature`.

    Only cameras that both sides declare get adapters. A camera whose
    orientation differs gets its own flip and rotation, so the resolver can fix
    each camera separately. A camera whose shape differs gets a resize. The
    proprioception rotation is added when the two `ee_pose` formats differ.
    """
    pool: list[ActionAdapter | ObservationAdapter] = []
    published = {camera.name: camera for camera in benchmark.sensors}
    resize_targets: dict[str, tuple[int, ...]] = {}
    for camera in signature.cameras:
        source = published.get(camera.name)
        if source is None:
            continue
        if source.orientation != camera.orientation:
            pool.append(FlipVerticalCameras(cameras=(camera.name,)))
            pool.append(Rotate180Cameras(cameras=(camera.name,)))
        if source.shape != camera.shape:
            resize_targets[camera.name] = camera.shape
    if resize_targets:
        pool.append(ResizeCameras(targets=resize_targets))
    policy_ee_pose = signature.proprioception.ee_pose
    benchmark_ee_pose = benchmark.embodiment.proprioception.ee_pose
    if (
        policy_ee_pose is not None
        and benchmark_ee_pose is not None
        and policy_ee_pose.rotation != benchmark_ee_pose.rotation
    ):
        pool.append(ProprioRotationAdapter(target=policy_ee_pose.rotation))
    return pool


def serve(
    predict: Callable[[Observation], Action],
    signature: PolicySignature,
    *,
    pipeline: Pipeline | Callable[[Benchmark], Pipeline] | None = None,
    server: str | None = None,
) -> None:
    """Serve `predict` on the local address from `MANIFOLD_SERVER_URL`.

    Pass `pipeline` to choose the adapters yourself, as a `Pipeline` or as a
    function that builds one for each benchmark. Without it, the resolver
    builds a pipeline for each benchmark from the adapters that
    `_default_pool` picks. The server checks the pipeline either way, before
    it accepts a benchmark.
    """
    address = server if server is not None else os.environ.get("MANIFOLD_SERVER_URL")
    if address is None:
        raise ValueError("MANIFOLD_SERVER_URL is required")
    host, port = _server_address(address)
    serve_endpoint(
        FunctionEndpoint(predict, signature),
        pipeline=pipeline if pipeline is not None else _resolving_pipeline(signature),
        host=host,
        port=port,
    )


def _resolving_pipeline(signature: PolicySignature) -> Callable[[Benchmark], Pipeline]:
    """Return a function that resolves a pipeline for each benchmark."""

    def pipeline(benchmark: Benchmark) -> Pipeline:
        resolved = resolve(signature, benchmark, _default_pool(signature, benchmark))
        if resolved is None:
            raise ValueError(
                f"the built-in adapters cannot bridge {benchmark.name} to this "
                "policy signature. Pass `pipeline` to choose the adapters yourself."
            )
        return resolved

    return pipeline


__all__ = ["serve"]
