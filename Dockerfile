# syntax=docker/dockerfile:1.7
# Containerised, reproducible environment for the Linux/CUDA cluster.
# Build (from Code/):   docker build -t grokking-tda .
# Train:                docker run --gpus all grokking-tda +experiment=tf_mod97_grok
#
# Dependencies install in their own layer (keyed on the lockfile) so source edits
# don't re-resolve the environment — the uv + lockfile payoff for Docker.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Bring in the uv binary from its official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 1) Resolve + install dependencies only (cached unless the lock changes).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2) Install the project itself.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"
ENTRYPOINT ["gtda-train"]
CMD ["+experiment=smoke"]
