# syntax=docker/dockerfile:1.7
# Build context: the pinned software-agent-sdk checkout.
ARG BASE_IMAGE=debian:bookworm-slim
FROM python:3.13-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/agent-server/.venv \
    UV_PYTHON_INSTALL_DIR=/agent-server/uv-managed-python
WORKDIR /agent-server
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY openhands-sdk ./openhands-sdk
COPY openhands-tools ./openhands-tools
COPY openhands-workspace ./openhands-workspace
COPY openhands-agent-server ./openhands-agent-server
RUN uv python install 3.13 && \
    uv venv --python-preference only-managed --python 3.13 .venv && \
    uv sync --frozen --no-dev --no-editable --managed-python --all-packages --extra sail --extra modal && \
    uv cache clean

FROM ${BASE_IMAGE} AS agent-server
ARG SDK_SHA
LABEL org.opencontainers.image.revision=${SDK_SHA}
ENV OPENHANDS_BUILD_GIT_SHA=${SDK_SHA} \
    OPENHANDS_BUILD_GIT_REF=sail-workspace-hardened \
    PATH=/agent-server/.venv/bin:$PATH \
    LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git ripgrep tmux procps sudo libatomic1 && \
    rm -rf /var/lib/apt/lists/* && \
    (getent group 10001 || groupadd -g 10001 openhands) && \
    (id openhands || useradd -m -u 10001 -g 10001 -s /bin/bash openhands) && \
    printf 'openhands ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/openhands && \
    mkdir -p /workspace && chown 10001:10001 /workspace
COPY --from=builder --chown=10001:10001 /agent-server /agent-server
USER 10001:10001
WORKDIR /workspace
EXPOSE 8000
CMD ["/agent-server/.venv/bin/python", "-m", "openhands.agent_server", "--host", "0.0.0.0", "--port", "8000"]

FROM agent-server AS codescout
COPY examples/02_remote_agent_server/06_custom_tool/custom_tools /app/custom_tools
COPY examples/02_remote_agent_server/06_custom_tool/prompts_codescout /app/prompts_codescout
ENV OH_EXTRA_PYTHON_PATH=/app
