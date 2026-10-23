"""The start command of a benchmark image.

    python -m manifold.benchmark /opt/manifold/config.json

The command reads the config and imports `hooks.py` from the config's directory. It
then builds the Service and runs the Client. The runner passes the addresses of the
task server, the policy and the live view in environment variables.

An image build runs the command with `--check`. The check reads the config and the
hooks file, writes the spec to `spec.json` beside the config, and exits:

    python -m manifold.benchmark --check /opt/manifold/config.json
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from manifold.benchmark._client import Client
from manifold.benchmark._live_view import live_view_publisher
from manifold.benchmark._service import make_service
from manifold.benchmark._worker import WorkerSession
from manifold.benchmark.config import load_config
from manifold.benchmark.hooks import load_hooks

# The runner sets these environment variables when it starts the container.
_SERVER_URL_ENV = "MANIFOLD_SERVER_URL"
_TASK_SERVER_ADDRESS_ENV = "MANIFOLD_TASK_SERVER_ADDRESS"
_WORKER_ID_ENV = "MANIFOLD_WORKER_ID"
# The runner sets this variable only when a user watches the live view.
_LIVE_VIEW_SERVER_ADDRESS_ENV = "MANIFOLD_LIVE_VIEW_SERVER_ADDRESS"


def main() -> None:
    """Run the benchmark that a config file describes."""
    parser = argparse.ArgumentParser(prog="python -m manifold.benchmark", description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the config and the hooks file, write spec.json beside the config, and exit",
    )
    parser.add_argument("config", type=Path, help="the benchmark's config file")
    args = parser.parse_args()

    config = load_config(args.config)
    benchmark = config.spec
    hooks = load_hooks(config, args.config.parent)
    if args.check:
        spec = args.config.parent / "spec.json"
        spec.write_text(benchmark.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"checked {args.config} and its hooks file, and wrote {spec}", flush=True)
        return
    # Other packages log each request at INFO, so only the `manifold` loggers show INFO.
    # The hooks write to `manifold.benchmark.hooks`.
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("manifold").setLevel(logging.INFO)
    service = make_service(config, hooks)
    policy_address = os.environ[_SERVER_URL_ENV]
    # Some simulators, such as Isaac Sim, end the process when they stop. So the
    # Service stops after the live view and the task server session close.
    try:
        service.start()
        with (
            live_view_publisher(
                benchmark,
                os.environ.get(_LIVE_VIEW_SERVER_ADDRESS_ENV),
                config.replay.cameras[0],
            ) as live_view,
            WorkerSession(
                os.environ[_TASK_SERVER_ADDRESS_ENV],
                worker_id=os.environ[_WORKER_ID_ENV],
            ) as session,
        ):
            Client(config, service, session, policy_address, live_view).run()
    finally:
        service.stop()


if __name__ == "__main__":
    main()
