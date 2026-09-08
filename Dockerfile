# syntax=docker/dockerfile:1
# The SDK as a container image, published as ghcr.io/bifrostai/manifold-sdk:<version>
# by .github/workflows/publish.yml. Build from the repository root, which is the context:
#
#     docker build -t ghcr.io/bifrostai/manifold-sdk:0.1.0 .
#
# A workload image uses it one of two ways, always pinned by tag AND digest:
#
#   FROM ghcr.io/bifrostai/manifold-sdk:<version>@sha256:...
#       when the SDK owns the interpreter.
#
#   COPY --from=ghcr.io/bifrostai/manifold-sdk:<version>@sha256:... /dist /tmp/sdk
#   RUN <that interpreter> -m pip install /tmp/sdk/*.whl
#       when another environment owns it (a simulator's bundled Python, a model
#       stack's own lock). /dist carries the wheel and the exported lock for that.
#
# No CMD: this is a base, never registered as a workload.

# Only uv reads the lock and builds the wheel, and the image does not need uv
# afterwards, so both happen in a stage that is discarded. `--frozen` fails the build
# on a lock that no longer matches pyproject, so a forgotten `uv lock` cannot ship as
# a quietly different dependency set.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS build
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
# `images` is the extra that pulls Pillow, which encodes jpeg and png at the wire edge;
# a policy or benchmark image that sends compressed frames needs it, so the default
# environment carries it.
RUN uv build --wheel --out-dir /dist \
 && uv export --frozen --no-emit-project --no-dev --extra images --no-hashes \
        --output-file /dist/requirements.txt

# Debian, not alpine: a runner probes a policy container's readiness with
# `docker exec <name> bash -c 'exec 3<>/dev/tcp/127.0.0.1/8000'`, and /dev/tcp is a
# bash feature busybox ash does not have.
FROM python:3.12-slim-bookworm

# GitHub links a container package to its repository through this label; without it
# the package is unlinked and does not inherit the repository's permissions.
LABEL org.opencontainers.image.source=https://github.com/bifrostai/manifold-sdk \
      org.opencontainers.image.licenses=MPL-2.0

WORKDIR /opt/manifold

# Otherwise a crashing container's last lines sit in a block buffer that the exit
# discards, and the runner reads container output to explain an exit.
ENV PYTHONUNBUFFERED=1

# The dependencies install in their own layer above the wheel: they are most of the
# image (numpy and scipy) and change far less often than the SDK, so only the lock
# export comes in before them and the wheel, which changes every build, after.
COPY --from=build /dist/requirements.txt /dist/
RUN pip install --no-cache-dir -r /dist/requirements.txt
# --no-deps because the requirements above are the whole dependency set, and
# resolving again here would let pip reach past the lock.
COPY --from=build /dist /dist
RUN pip install --no-cache-dir --no-deps /dist/*.whl
