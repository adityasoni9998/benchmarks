#!/usr/bin/env python3
"""Build SWE-rebench agent-server images in parallel Modal VM Sandboxes.

Each VM Sandbox runs Docker and BuildKit, builds one or more dataset images with
the SWE-rebench testbed reinstall layer, and optionally pushes them to a
registry. Successful registry images can then be published as Modal named
Images.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    as_completed,
)
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import modal
from modal.container_process import ContainerProcess

from benchmarks.swerebench.apptainer_build import (
    SwerebenchImageSpec,
    collect_unique_image_specs,
)
from benchmarks.swerebench.docker_build import docker_custom_tag


APP_NAME = "swerebench-modal-image-builder"
BUILDER_MODAL_IMAGE_NAME = "swerebench-modal-vm-builder:latest"
BENCHMARKS_REPOSITORY = "https://github.com/adityasoni9998/benchmarks.git"
BENCHMARKS_SHA = "1c09d45a5689715bc0f53cd31ae091be23cbb19c"
SDK_SHA = "5acdf05ae2fc6224f92114db6f557b181cd0f280"
SDK_SHORT_SHA = SDK_SHA[:7]
BUILDKIT_IMAGE = "moby/buildkit:v0.32.2"
DOCKERHUB_SECRET_NAME = "dockerhub-adityasoni8"
DOCKERHUB_USERNAME = "adityasoni8"
LOCAL_TEST_IMAGE = "local/swerebench-modal-build-test"
REGISTRY_IMAGE = "docker.io/adityasoni8/eval-agent-server"
BUILD_TIMEOUT_SECONDS = 45 * 60
SANDBOX_TIMEOUT_SECONDS = 60 * 60
PUBLISH_TIMEOUT_SECONDS = 60 * 60
MAX_MODAL_IMAGE_NAME_LENGTH = 64
MAX_IMAGES_PER_VM = 8
MAX_PLAN_BATCH_DETAILS = 20


@dataclass(frozen=True)
class BuildRequest:
    base_image: str
    custom_tag: str
    install_command: str
    destination_image: str
    push: bool = False
    force_build: bool = True
    repo: str | None = None
    instance_id: str | None = None


@dataclass
class BuildResult:
    method: Literal["vm-sandbox"]
    request: BuildRequest
    started_at: str
    duration_seconds: float
    success: bool
    tags: list[str]
    error: str | None = None
    docker_info: str | None = None
    sandbox_id: str | None = None


@dataclass(frozen=True)
class SandboxBuildConfig:
    cpu: float = 2.0
    memory: int = 8192
    build_timeout_seconds: int = BUILD_TIMEOUT_SECONDS
    sandbox_timeout_seconds: int = SANDBOX_TIMEOUT_SECONDS
    stream_build_logs: bool = False


def build_requests(
    image_specs: Iterable[SwerebenchImageSpec],
    *,
    push: bool = False,
    force_build: bool = True,
    destination_image: str | None = None,
) -> list[BuildRequest]:
    destination = destination_image or (REGISTRY_IMAGE if push else LOCAL_TEST_IMAGE)
    requests = []
    for spec in image_specs:
        if spec.install_command is None:
            raise ValueError(f"{spec.base_image} has no install command")
        requests.append(
            BuildRequest(
                base_image=spec.base_image,
                custom_tag=docker_custom_tag(spec),
                install_command=spec.install_command,
                destination_image=destination,
                push=push,
                force_build=force_build,
                repo=spec.repo,
                instance_id=spec.instance_id,
            )
        )
    return requests


def parse_repos(value: str) -> list[str]:
    """Parse a comma-separated repository filter while preserving user order."""
    return list(
        dict.fromkeys(repo.strip() for repo in value.split(",") if repo.strip())
    )


def select_image_specs_by_repo(
    image_specs: list[SwerebenchImageSpec],
    *,
    repos: list[str],
    images_per_repo: int,
) -> list[SwerebenchImageSpec]:
    """Select exactly ``images_per_repo`` image specs for every requested repo."""
    if images_per_repo < 1:
        raise ValueError("images_per_repo must be at least 1")
    if images_per_repo > MAX_IMAGES_PER_VM:
        raise ValueError(
            f"images_per_repo cannot exceed the per-VM cap of {MAX_IMAGES_PER_VM}"
        )

    by_repo: dict[str, list[SwerebenchImageSpec]] = {}
    for spec in image_specs:
        if spec.repo is not None:
            by_repo.setdefault(spec.repo, []).append(spec)

    selected: list[SwerebenchImageSpec] = []
    for repo in repos:
        repo_specs = sorted(by_repo.get(repo, []), key=lambda spec: spec.base_image)
        if len(repo_specs) < images_per_repo:
            raise ValueError(
                f"Repository {repo!r} has {len(repo_specs)} selected images; "
                f"need {images_per_repo}"
            )
        selected.extend(repo_specs[:images_per_repo])
    return selected


def expected_registry_tag(request: BuildRequest) -> str:
    return (
        f"{request.destination_image}:{SDK_SHORT_SHA}-{request.custom_tag}"
        "-source-minimal"
    )


def modal_image_name(image_ref: str) -> str:
    """Apply SWE-Smith's naming rules, then shorten deterministically."""
    repository, tag = image_ref.rsplit(":", 1)
    name = repository.replace("/", "__")
    if len(name) > MAX_MODAL_IMAGE_NAME_LENGTH:
        digest = hashlib.sha256(name.encode()).hexdigest()[:8]
        name = f"{name[: MAX_MODAL_IMAGE_NAME_LENGTH - 9]}-{digest}"

    if len(tag) > MAX_MODAL_IMAGE_NAME_LENGTH:
        tag = tag.replace(".x86_64", "")
    if len(tag) > MAX_MODAL_IMAGE_NAME_LENGTH:
        tag = tag.replace("-source-minimal", "-src-min")
    if len(tag) > MAX_MODAL_IMAGE_NAME_LENGTH:
        digest = hashlib.sha256(tag.encode()).hexdigest()[:8]
        tag = f"{tag[: MAX_MODAL_IMAGE_NAME_LENGTH - 9]}-{digest}"
    return f"{name}:{tag}"


app = modal.App(APP_NAME)
registry_secret = modal.Secret.from_name(
    DOCKERHUB_SECRET_NAME,
    required_keys=["REGISTRY_USERNAME", "REGISTRY_PASSWORD"],
)
builder_image_definition = (
    modal.Image.from_registry("ubuntu:24.04", secret=registry_secret, add_python="3.12")
    .entrypoint([])
    .env(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "OPENHANDS_SUPPRESS_BANNER": "1",
            "UV_LINK_MODE": "copy",
        }
    )
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "docker-buildx",
        "docker.io",
        "git",
        "make",
    )
    .run_commands(
        "curl -LsSf https://astral.sh/uv/0.11.8/install.sh "
        "| env UV_INSTALL_DIR=/usr/local/bin sh",
        (
            f"git clone {BENCHMARKS_REPOSITORY} /opt/benchmarks "
            f"&& cd /opt/benchmarks && git checkout {BENCHMARKS_SHA} "
            "&& git submodule update --init --recursive "
            f'&& test "$(git -C vendor/software-agent-sdk rev-parse HEAD)" = {SDK_SHA} '
            "&& uv sync --dev --frozen"
        ),
    )
    .add_local_file(
        Path(__file__).with_name("Dockerfile.testbed-reinstall"),
        "/opt/benchmarks/benchmarks/swerebench/Dockerfile.testbed-reinstall",
        copy=True,
    )
    .add_local_file(
        Path(__file__).with_name("reinstall_testbed.sh"),
        "/opt/benchmarks/benchmarks/swerebench/reinstall_testbed.sh",
        copy=True,
    )
    .add_local_file(
        Path(__file__).with_name("docker_build.py"),
        "/opt/benchmarks/benchmarks/swerebench/docker_build.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parents[1] / "utils" / "image_utils.py",
        "/opt/benchmarks/benchmarks/utils/image_utils.py",
        copy=True,
    )
    .workdir("/opt/benchmarks")
)
builder_image = modal.Image.from_name(BUILDER_MODAL_IMAGE_NAME)


def publish_builder_runtime_image(
    *, app_handle: modal.App | None = None, announce: bool = True
) -> str:
    """Build and publish the shared VM runtime before concurrent launches."""
    target_app = app if app_handle is None else app_handle
    built_image = builder_image_definition.build(target_app)
    built_image.publish(BUILDER_MODAL_IMAGE_NAME)
    if announce:
        print(f"Published VM builder image {BUILDER_MODAL_IMAGE_NAME}")
    return BUILDER_MODAL_IMAGE_NAME


WORKER_CODE = """
import json
import os
import sys

from benchmarks.swerebench.apptainer_build import SwerebenchImageSpec
from benchmarks.swerebench.docker_build import build_docker_agent_image

request = json.loads(sys.argv[1])
os.environ["BUILDKIT_PROGRESS"] = "plain"
os.environ["BUILDKIT_RESET_ON_FAILURE"] = "1"
os.environ["EXTENSIONS_REF"] = "main"
os.environ["OPENHANDS_BUILDKIT_CACHE_MODE"] = "off"
if request["push"]:
    os.environ["OPENHANDS_IMAGE_COMPRESSION"] = "estargz"

spec = SwerebenchImageSpec(
    base_image=request["base_image"],
    custom_tag=request["custom_tag"],
    install_command=request["install_command"],
)
output = build_docker_agent_image(
    spec,
    target_image=request["destination_image"],
    push=request["push"],
    force_build=request.get("force_build", True),
)
print(json.dumps({"tags": output.tags, "error": output.error}, sort_keys=True))
raise SystemExit(1 if output.error or not output.tags else 0)
"""


def _stream_process(
    process: ContainerProcess[str], *, stream_output: bool = True
) -> tuple[str, str]:
    output: dict[str, list[str]] = {"stdout": [], "stderr": []}

    def consume(name: str) -> None:
        stream = process.stdout if name == "stdout" else process.stderr
        for line in stream:
            output[name].append(line)
            if stream_output:
                print(line, end="", file=sys.stdout if name == "stdout" else sys.stderr)

    stdout_thread = threading.Thread(target=consume, args=("stdout",))
    stderr_thread = threading.Thread(target=consume, args=("stderr",))
    stdout_thread.start()
    stderr_thread.start()
    process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return "".join(output["stdout"]), "".join(output["stderr"])


async def _stream_process_async(
    process: ContainerProcess[str], *, stream_output: bool = True
) -> tuple[str, str]:
    """Drain both process streams without dedicating local threads to them."""
    stdout_task = asyncio.create_task(process.stdout.read.aio())
    stderr_task = asyncio.create_task(process.stderr.read.aio())
    await process.wait.aio()
    stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
    if stream_output:
        if stdout:
            print(stdout, end="", file=sys.stdout)
        if stderr:
            print(stderr, end="", file=sys.stderr)
    return stdout, stderr


def _cleanup_sandbox(sandbox: modal.Sandbox) -> str | None:
    """Best-effort cleanup that never converts completed work into a run crash."""
    errors: list[str] = []
    try:
        sandbox.terminate(wait=True)
    except Exception as exc:
        errors.append(f"terminate: {type(exc).__name__}: {exc}")
    try:
        sandbox.detach()
    except Exception as exc:
        errors.append(f"detach: {type(exc).__name__}: {exc}")
    return "; ".join(errors) or None


async def _cleanup_sandbox_async(sandbox: modal.Sandbox) -> str | None:
    """Asynchronous counterpart to :func:`_cleanup_sandbox`."""
    errors: list[str] = []
    try:
        await sandbox.terminate.aio(wait=True)
    except Exception as exc:
        errors.append(f"terminate: {type(exc).__name__}: {exc}")
    try:
        await sandbox.detach.aio()
    except Exception as exc:
        errors.append(f"detach: {type(exc).__name__}: {exc}")
    return "; ".join(errors) or None


def _build_in_vm_sandbox(
    request: BuildRequest, config: SandboxBuildConfig
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    sandbox: modal.Sandbox | None = None
    sandbox_id: str | None = None
    tags: list[str] = []
    error: str | None = None
    cleanup_error: str | None = None
    docker_info: str | None = None
    try:
        sandbox = modal.Sandbox.create(
            "/usr/bin/dockerd",
            "--host=unix:///var/run/docker.sock",
            "--log-level=error",
            timeout=config.sandbox_timeout_seconds,
            app=app,
            image=builder_image,
            secrets=[registry_secret],
            cpu=config.cpu,
            memory=config.memory,
            experimental_options={"vm_runtime": True},
            tags={"method": "vm-sandbox", "benchmark": "swerebench"},
        )
        sandbox_id = sandbox.object_id
        setup = sandbox.exec(
            "bash",
            "-lc",
            (
                "set -euo pipefail; "
                "for i in $(seq 1 120); do "
                "docker info >/dev/null 2>&1 && break; sleep 1; done; "
                "docker info --format 'storage_driver={{.Driver}}'; "
                f'test "$REGISTRY_USERNAME" = "{DOCKERHUB_USERNAME}"; '
                "printf '%s' \"$REGISTRY_PASSWORD\" | docker login "
                '--username "$REGISTRY_USERNAME" --password-stdin >/dev/null; '
                "docker buildx create --name swerebench-modal-sdk "
                "--driver docker-container "
                f"--driver-opt image={BUILDKIT_IMAGE} --use; "
                "docker buildx inspect --bootstrap"
            ),
            timeout=180,
        )
        setup_stdout, setup_stderr = _stream_process(
            setup, stream_output=config.stream_build_logs
        )
        if setup.returncode != 0:
            raise RuntimeError(
                f"Docker setup failed: {setup_stderr[-4000:] or setup_stdout[-4000:]}"
            )
        docker_info = setup_stdout

        process = sandbox.exec(
            "/opt/benchmarks/.venv/bin/python",
            "-c",
            WORKER_CODE,
            json.dumps(asdict(request)),
            workdir="/opt/benchmarks",
            timeout=config.build_timeout_seconds,
            env={"OPENHANDS_SUPPRESS_BANNER": "1", "UV_LINK_MODE": "copy"},
        )
        stdout, stderr = _stream_process(
            process, stream_output=config.stream_build_logs
        )
        result_line = next(
            (line for line in reversed(stdout.splitlines()) if line.startswith("{")),
            None,
        )
        if result_line is not None:
            payload = json.loads(result_line)
            tags = list(payload.get("tags", []))
            error = payload.get("error")
        if process.returncode != 0 and error is None:
            error = stderr[-4000:] or stdout[-4000:] or "Build process failed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if sandbox is not None:
            cleanup_error = _cleanup_sandbox(sandbox)

    result = asdict(
        BuildResult(
            method="vm-sandbox",
            request=request,
            started_at=started_at,
            duration_seconds=round(time.monotonic() - started, 3),
            success=error is None and bool(tags),
            tags=tags,
            error=error,
            docker_info=docker_info,
            sandbox_id=sandbox_id,
        )
    )
    if cleanup_error is not None:
        result["sandbox_cleanup_error"] = cleanup_error
    return result


def group_build_requests_by_repo(
    requests: list[BuildRequest], *, max_images_per_vm: int
) -> list[list[BuildRequest]]:
    """Pack repo-ordered requests into full VM batches on a best-effort basis."""
    if max_images_per_vm < 1:
        raise ValueError("max_images_per_vm must be at least 1")
    if max_images_per_vm > MAX_IMAGES_PER_VM:
        raise ValueError(
            f"max_images_per_vm cannot exceed the safety cap of {MAX_IMAGES_PER_VM}"
        )

    by_repo: dict[str, list[BuildRequest]] = {}
    for request in requests:
        # Metadata-free requests are kept isolated instead of accidentally
        # combining unrelated repositories in one sandbox.
        group_key = request.repo or f"__base_image__:{request.base_image}"
        by_repo.setdefault(group_key, []).append(request)

    batches: list[list[BuildRequest]] = []
    current_batch: list[BuildRequest] = []
    for group_key in sorted(by_repo):
        repo_requests = sorted(
            by_repo[group_key], key=lambda request: request.base_image
        )
        offset = 0
        while offset < len(repo_requests):
            remaining_capacity = max_images_per_vm - len(current_batch)
            take = min(remaining_capacity, len(repo_requests) - offset)
            current_batch.extend(repo_requests[offset : offset + take])
            offset += take
            if len(current_batch) == max_images_per_vm:
                batches.append(current_batch)
                current_batch = []
    if current_batch:
        batches.append(current_batch)
    return batches


def print_repo_batch_plan(
    requests: list[BuildRequest], *, max_images_per_vm: int, details: bool = True
) -> None:
    """Print the exact VM-to-image allocation before any sandboxes are created."""
    batches = group_build_requests_by_repo(
        requests, max_images_per_vm=max_images_per_vm
    )
    visible_batches = batches[:MAX_PLAN_BATCH_DETAILS] if details else []
    print(
        json.dumps(
            {
                "image_count": len(requests),
                "vm_count": len(batches),
                "max_images_per_vm": max_images_per_vm,
                "omitted_vm_batches": len(batches) - len(visible_batches),
                "batches": [
                    {
                        "vm_batch": index,
                        "repos": list(dict.fromkeys(request.repo for request in batch)),
                        "image_count": len(batch),
                        "instances": [request.instance_id for request in batch],
                        "base_images": [request.base_image for request in batch],
                    }
                    for index, batch in enumerate(visible_batches, start=1)
                ],
            },
            indent=2,
        )
    )


def _build_repo_batch_in_vm_sandbox(
    requests: list[BuildRequest],
    config: SandboxBuildConfig,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Build a same-repository request batch using one Docker/BuildKit sandbox."""
    if not requests:
        raise ValueError("A VM batch must contain at least one build request")

    batch_started = time.monotonic()
    sandbox: modal.Sandbox | None = None
    sandbox_id: str | None = None
    docker_info: str | None = None
    cleanup_error: str | None = None
    results: list[dict[str, Any]] = []
    batch_repos = list(dict.fromkeys(request.repo for request in requests))
    try:
        sandbox = modal.Sandbox.create(
            "/usr/bin/dockerd",
            "--host=unix:///var/run/docker.sock",
            "--log-level=error",
            timeout=config.sandbox_timeout_seconds,
            app=app,
            image=builder_image,
            secrets=[registry_secret],
            cpu=config.cpu,
            memory=config.memory,
            experimental_options={"vm_runtime": True},
            tags={
                "method": "vm-repo-batch",
                "benchmark": "swerebench",
                "repo": (
                    (requests[0].repo or "unknown").replace("/", "__")[:64]
                    if len(batch_repos) == 1
                    else "mixed"
                ),
                "repo_count": str(len(batch_repos)),
            },
        )
        sandbox_id = sandbox.object_id
        setup = sandbox.exec(
            "bash",
            "-lc",
            (
                "set -euo pipefail; "
                "for i in $(seq 1 120); do "
                "docker info >/dev/null 2>&1 && break; sleep 1; done; "
                "docker info --format 'storage_driver={{.Driver}}'; "
                f'test "$REGISTRY_USERNAME" = "{DOCKERHUB_USERNAME}"; '
                "printf '%s' \"$REGISTRY_PASSWORD\" | docker login "
                '--username "$REGISTRY_USERNAME" --password-stdin >/dev/null; '
                "docker buildx create --name swerebench-modal-sdk "
                "--driver docker-container "
                f"--driver-opt image={BUILDKIT_IMAGE} --use; "
                "docker buildx inspect --bootstrap"
            ),
            timeout=180,
        )
        setup_stdout, setup_stderr = _stream_process(
            setup, stream_output=config.stream_build_logs
        )
        if setup.returncode != 0:
            raise RuntimeError(
                f"Docker setup failed: {setup_stderr[-4000:] or setup_stdout[-4000:]}"
            )
        docker_info = setup_stdout

        for request in requests:
            started_at = datetime.now(UTC).isoformat()
            started = time.monotonic()
            tags: list[str] = []
            error: str | None = None
            try:
                process = sandbox.exec(
                    "/opt/benchmarks/.venv/bin/python",
                    "-c",
                    WORKER_CODE,
                    json.dumps(asdict(request)),
                    workdir="/opt/benchmarks",
                    timeout=config.build_timeout_seconds,
                    env={"OPENHANDS_SUPPRESS_BANNER": "1", "UV_LINK_MODE": "copy"},
                )
                stdout, stderr = _stream_process(
                    process, stream_output=config.stream_build_logs
                )
                result_line = next(
                    (
                        line
                        for line in reversed(stdout.splitlines())
                        if line.startswith("{")
                    ),
                    None,
                )
                if result_line is not None:
                    payload = json.loads(result_line)
                    tags = list(payload.get("tags", []))
                    error = payload.get("error")
                if process.returncode != 0 and error is None:
                    error = stderr[-4000:] or stdout[-4000:] or "Build process failed"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            result = asdict(
                BuildResult(
                    method="vm-sandbox",
                    request=request,
                    started_at=started_at,
                    duration_seconds=round(time.monotonic() - started, 3),
                    success=error is None and bool(tags),
                    tags=tags,
                    error=error,
                    docker_info=docker_info,
                    sandbox_id=sandbox_id,
                )
            )
            results.append(result)
            if on_result is not None:
                try:
                    on_result(result)
                except Exception as exc:
                    result["result_callback_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        batch_error = f"{type(exc).__name__}: {exc}"
        for request in requests[len(results) :]:
            result = asdict(
                BuildResult(
                    method="vm-sandbox",
                    request=request,
                    started_at=datetime.now(UTC).isoformat(),
                    duration_seconds=round(time.monotonic() - batch_started, 3),
                    success=False,
                    tags=[],
                    error=batch_error,
                    docker_info=docker_info,
                    sandbox_id=sandbox_id,
                )
            )
            results.append(result)
            if on_result is not None:
                try:
                    on_result(result)
                except Exception as callback_exc:
                    result["result_callback_error"] = (
                        f"{type(callback_exc).__name__}: {callback_exc}"
                    )
    finally:
        if sandbox is not None:
            cleanup_error = _cleanup_sandbox(sandbox)

    batch_duration_seconds = round(time.monotonic() - batch_started, 3)
    for result in results:
        result.update(
            {
                "batch_repo": batch_repos[0] if len(batch_repos) == 1 else None,
                "batch_repos": batch_repos,
                "batch_size": len(requests),
                "batch_duration_seconds": batch_duration_seconds,
            }
        )
        if cleanup_error is not None:
            result["sandbox_cleanup_error"] = cleanup_error
    return results


async def _build_repo_batch_in_vm_sandbox_async(
    requests: list[BuildRequest],
    config: SandboxBuildConfig,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Build one VM batch using Modal's native asynchronous Sandbox API."""
    if not requests:
        raise ValueError("A VM batch must contain at least one build request")

    batch_started = time.monotonic()
    sandbox: modal.Sandbox | None = None
    sandbox_id: str | None = None
    docker_info: str | None = None
    cleanup_error: str | None = None
    results: list[dict[str, Any]] = []
    batch_repos = list(dict.fromkeys(request.repo for request in requests))
    try:
        sandbox = await modal.Sandbox._experimental_create.aio(
            "/usr/bin/dockerd",
            "--host=unix:///var/run/docker.sock",
            "--log-level=error",
            timeout=config.sandbox_timeout_seconds,
            app=app,
            image=builder_image,
            secrets=[registry_secret],
            cpu=config.cpu,
            memory=config.memory,
            experimental_options={"vm_runtime": True},
            tags={
                "method": "vm-repo-batch",
                "benchmark": "swerebench",
                "repo": (
                    (requests[0].repo or "unknown").replace("/", "__")[:64]
                    if len(batch_repos) == 1
                    else "mixed"
                ),
                "repo_count": str(len(batch_repos)),
            },
        )
        sandbox_id = sandbox.object_id
        setup = await sandbox.exec.aio(
            "bash",
            "-lc",
            (
                "set -euo pipefail; "
                "for i in $(seq 1 120); do "
                "docker info >/dev/null 2>&1 && break; sleep 1; done; "
                "docker info --format 'storage_driver={{.Driver}}'; "
                f'test "$REGISTRY_USERNAME" = "{DOCKERHUB_USERNAME}"; '
                "printf '%s' \"$REGISTRY_PASSWORD\" | docker login "
                '--username "$REGISTRY_USERNAME" --password-stdin >/dev/null; '
                "docker buildx create --name swerebench-modal-sdk "
                "--driver docker-container "
                f"--driver-opt image={BUILDKIT_IMAGE} --use; "
                "docker buildx inspect --bootstrap"
            ),
            timeout=180,
        )
        setup_stdout, setup_stderr = await _stream_process_async(
            setup, stream_output=config.stream_build_logs
        )
        if setup.returncode != 0:
            raise RuntimeError(
                f"Docker setup failed: {setup_stderr[-4000:] or setup_stdout[-4000:]}"
            )
        docker_info = setup_stdout

        for request in requests:
            started_at = datetime.now(UTC).isoformat()
            started = time.monotonic()
            tags: list[str] = []
            error: str | None = None
            try:
                process = await sandbox.exec.aio(
                    "/opt/benchmarks/.venv/bin/python",
                    "-c",
                    WORKER_CODE,
                    json.dumps(asdict(request)),
                    workdir="/opt/benchmarks",
                    timeout=config.build_timeout_seconds,
                    env={"OPENHANDS_SUPPRESS_BANNER": "1", "UV_LINK_MODE": "copy"},
                )
                stdout, stderr = await _stream_process_async(
                    process, stream_output=config.stream_build_logs
                )
                result_line = next(
                    (
                        line
                        for line in reversed(stdout.splitlines())
                        if line.startswith("{")
                    ),
                    None,
                )
                if result_line is not None:
                    payload = json.loads(result_line)
                    tags = list(payload.get("tags", []))
                    error = payload.get("error")
                if process.returncode != 0 and error is None:
                    error = stderr[-4000:] or stdout[-4000:] or "Build process failed"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            result = asdict(
                BuildResult(
                    method="vm-sandbox",
                    request=request,
                    started_at=started_at,
                    duration_seconds=round(time.monotonic() - started, 3),
                    success=error is None and bool(tags),
                    tags=tags,
                    error=error,
                    docker_info=docker_info,
                    sandbox_id=sandbox_id,
                )
            )
            results.append(result)
            if on_result is not None:
                try:
                    on_result(result)
                except Exception as exc:
                    result["result_callback_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        batch_error = f"{type(exc).__name__}: {exc}"
        for request in requests[len(results) :]:
            result = asdict(
                BuildResult(
                    method="vm-sandbox",
                    request=request,
                    started_at=datetime.now(UTC).isoformat(),
                    duration_seconds=round(time.monotonic() - batch_started, 3),
                    success=False,
                    tags=[],
                    error=batch_error,
                    docker_info=docker_info,
                    sandbox_id=sandbox_id,
                )
            )
            results.append(result)
            if on_result is not None:
                try:
                    on_result(result)
                except Exception as callback_exc:
                    result["result_callback_error"] = (
                        f"{type(callback_exc).__name__}: {callback_exc}"
                    )
    finally:
        if sandbox is not None:
            cleanup_error = await _cleanup_sandbox_async(sandbox)

    batch_duration_seconds = round(time.monotonic() - batch_started, 3)
    for result in results:
        result.update(
            {
                "batch_repo": batch_repos[0] if len(batch_repos) == 1 else None,
                "batch_repos": batch_repos,
                "batch_size": len(requests),
                "batch_duration_seconds": batch_duration_seconds,
            }
        )
        if cleanup_error is not None:
            result["sandbox_cleanup_error"] = cleanup_error
    return results


def _create_results_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(".agent_tmp") / f"modal-build-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_results(
    method: str,
    results: list[dict[str, Any]],
    *,
    announce: bool = True,
    output_dir: Path | None = None,
) -> Path:
    output_dir = output_dir or _create_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method}.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    if announce:
        print(f"Wrote {method} results to {output_path}")
    return output_path


def run_vm_builds(
    requests: list[BuildRequest],
    *,
    max_workers: int,
    config: SandboxBuildConfig,
    show_progress: bool = True,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if not requests:
        raise ValueError("No image build requests selected")
    results: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, len(requests))) as executor:
        futures = {
            executor.submit(_build_in_vm_sandbox, request, config): request
            for request in requests
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if on_result is not None:
                try:
                    on_result(result)
                except Exception as exc:
                    result["result_callback_error"] = f"{type(exc).__name__}: {exc}"
            completed += 1
            if show_progress:
                successful = sum(bool(item["success"]) for item in results)
                print(
                    f"Progress: instances={completed}/{len(requests)} "
                    f"sandboxes={completed}/{len(requests)} "
                    f"successful={successful} failed={completed - successful}"
                )
    results.sort(key=lambda result: str(result["request"]["base_image"]))
    _write_results("vm-sandbox", results, announce=show_progress, output_dir=output_dir)
    return results


async def _run_repo_batched_vm_builds_async(
    requests: list[BuildRequest],
    *,
    max_workers: int,
    max_images_per_vm: int,
    config: SandboxBuildConfig,
    show_progress: bool = True,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Build repo-local chunks with native async Sandbox concurrency."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    batches = group_build_requests_by_repo(
        requests, max_images_per_vm=max_images_per_vm
    )
    if not batches:
        raise ValueError("No image build requests selected")

    results: list[dict[str, Any]] = []
    completed_instances = 0
    completed_sandboxes = 0
    semaphore = asyncio.Semaphore(min(max_workers, len(batches)))

    async def run_batch(
        batch: list[BuildRequest],
    ) -> tuple[list[BuildRequest], list[dict[str, Any]]]:
        async with semaphore:
            batch_results = await _build_repo_batch_in_vm_sandbox_async(
                batch, config, on_result
            )
            return batch, batch_results

    tasks = [asyncio.create_task(run_batch(batch)) for batch in batches]
    for completed in asyncio.as_completed(tasks):
        batch, batch_results = await completed
        results.extend(batch_results)
        completed_instances += len(batch)
        completed_sandboxes += 1
        if show_progress:
            successful = sum(bool(result["success"]) for result in results)
            print(
                f"Progress: instances={completed_instances}/{len(requests)} "
                f"sandboxes={completed_sandboxes}/{len(batches)} "
                f"successful={successful} "
                f"failed={completed_instances - successful}"
            )

    results.sort(key=lambda result: str(result["request"]["base_image"]))
    _write_results(
        "vm-repo-batch", results, announce=show_progress, output_dir=output_dir
    )
    return results


def run_repo_batched_vm_builds(
    requests: list[BuildRequest],
    *,
    max_workers: int,
    max_images_per_vm: int,
    config: SandboxBuildConfig,
    show_progress: bool = True,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the asynchronous repo-batched builder from the local entrypoint."""
    return asyncio.run(
        _run_repo_batched_vm_builds_async(
            requests,
            max_workers=max_workers,
            max_images_per_vm=max_images_per_vm,
            config=config,
            show_progress=show_progress,
            on_result=on_result,
            output_dir=output_dir,
        )
    )


def publish_registry_image(
    image_ref: str,
    *,
    stream_output: bool = True,
    announce: bool = True,
    timeout_seconds: int = PUBLISH_TIMEOUT_SECONDS,
) -> str:
    name = modal_image_name(image_ref)
    image = modal.Image.from_registry(
        image_ref,
        secret=registry_secret,
        force_build=True,
    ).entrypoint([])
    app_handle = modal.App.lookup(APP_NAME, create_if_missing=True)

    def wait_for_modal(future: Future[Any], operation: str) -> Any:
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Timed out after {timeout_seconds}s while {operation} {image_ref}"
            ) from exc

    def build_and_publish() -> None:
        start_build = cast(Callable[..., Future[Any]], image.build)
        built_image = wait_for_modal(start_build(app_handle, _future=True), "importing")
        start_publish = cast(Callable[..., Future[Any]], built_image.publish)
        wait_for_modal(start_publish(name, _future=True), "publishing")

    if stream_output:
        with modal.enable_output():
            build_and_publish()
    else:
        build_and_publish()
    if announce:
        print(f"Published {image_ref} as Modal Image {name}")
    return name


def registry_refs_from_build_results(results: Iterable[dict[str, Any]]) -> list[str]:
    refs = []
    for result in results:
        if not isinstance(result, dict) or not result.get("success"):
            continue
        request_data = result.get("request")
        if not isinstance(request_data, dict):
            raise ValueError("Successful result is missing its request")
        request = BuildRequest(**request_data)
        if request.push:
            refs.append(expected_registry_tag(request))
    return list(dict.fromkeys(refs))


def registry_refs_from_results(path: Path) -> list[str]:
    results = json.loads(path.read_text())
    if not isinstance(results, list):
        raise ValueError("Build results must contain a JSON list")
    return registry_refs_from_build_results(results)


def _publish_registry_image_result(
    image_ref: str,
    *,
    stream_output: bool,
    timeout_seconds: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    context = metadata or {}
    try:
        modal_image = publish_registry_image(
            image_ref,
            stream_output=stream_output,
            announce=False,
            timeout_seconds=timeout_seconds,
        )
        return {
            **context,
            "duration_seconds": round(time.monotonic() - started, 3),
            "image_ref": image_ref,
            "modal_image": modal_image,
            "success": True,
        }
    except Exception as exc:
        return {
            **context,
            "duration_seconds": round(time.monotonic() - started, 3),
            "image_ref": image_ref,
            "error": f"{type(exc).__name__}: {exc}",
            "success": False,
        }


class ParallelNamedImagePublisher:
    """Publish registry images using a pool independent of VM build workers."""

    def __init__(
        self,
        *,
        max_workers: int,
        show_progress: bool = True,
        stream_output: bool = False,
        timeout_seconds: int = PUBLISH_TIMEOUT_SECONDS,
    ) -> None:
        if max_workers < 1:
            raise ValueError("publish max_workers must be at least 1")
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._show_progress = show_progress
        self._stream_output = stream_output
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._futures: list[Future[dict[str, Any]]] = []
        self._submitted_refs: set[str] = set()
        self._completed = 0
        self._successful = 0

    def submit_image_ref(
        self, image_ref: str, *, metadata: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            if image_ref in self._submitted_refs:
                return
            self._submitted_refs.add(image_ref)
            future = self._executor.submit(
                _publish_registry_image_result,
                image_ref,
                stream_output=self._stream_output,
                timeout_seconds=self._timeout_seconds,
                metadata=metadata,
            )
            self._futures.append(future)
        future.add_done_callback(self._on_done)

    def submit_build_result(self, result: dict[str, Any]) -> None:
        request = result.get("request", {})
        metadata = {
            "instance_id": request.get("instance_id"),
            "repo": request.get("repo"),
            "base_image": request.get("base_image"),
            "sandbox_id": result.get("sandbox_id"),
        }
        for image_ref in registry_refs_from_build_results([result]):
            self.submit_image_ref(image_ref, metadata=metadata)

    def _on_done(self, future: Future[dict[str, Any]]) -> None:
        result = future.result()
        with self._lock:
            self._completed += 1
            self._successful += int(bool(result["success"]))
            if self._show_progress:
                print(
                    f"Named images: completed={self._completed} "
                    f"submitted={len(self._submitted_refs)} "
                    f"successful={self._successful} "
                    f"failed={self._completed - self._successful}"
                )

    def finish(self) -> list[dict[str, Any]]:
        self._executor.shutdown(wait=True)
        return sorted(
            (future.result() for future in self._futures),
            key=lambda result: str(result["image_ref"]),
        )


def publish_registry_images(
    image_refs: Iterable[str],
    *,
    max_workers: int,
    show_progress: bool = True,
    stream_output: bool = False,
    timeout_seconds: int = PUBLISH_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    publisher = ParallelNamedImagePublisher(
        max_workers=max_workers,
        show_progress=show_progress,
        stream_output=stream_output,
        timeout_seconds=timeout_seconds,
    )
    for image_ref in image_refs:
        publisher.submit_image_ref(image_ref)
    return publisher.finish()


def _print_publish_summary(
    results: list[dict[str, Any]], *, wall_clock_seconds: float
) -> None:
    successful = sum(bool(result["success"]) for result in results)
    print(
        json.dumps(
            {
                "named_images_total": len(results),
                "named_images_successful": successful,
                "named_images_failed": len(results) - successful,
                "named_images_wall_clock_seconds": wall_clock_seconds,
            },
            indent=2,
        )
    )


def _require_complete_run(
    build_results: list[dict[str, Any]],
    publish_results: list[dict[str, Any]] | None = None,
) -> None:
    build_failures = sum(not bool(result["success"]) for result in build_results)
    publish_failures = sum(
        not bool(result["success"]) for result in (publish_results or [])
    )
    expected_publishes = sum(
        bool(result["success"] and result["request"]["push"])
        for result in build_results
    )
    missing_publishes = (
        max(0, expected_publishes - len(publish_results or []))
        if publish_results is not None
        else 0
    )
    if build_failures or publish_failures or missing_publishes:
        raise RuntimeError(
            "Image run incomplete: "
            f"build_failures={build_failures}, "
            f"publish_failures={publish_failures}, "
            f"missing_publishes={missing_publishes}"
        )


def collect_run_failures(
    build_results: list[dict[str, Any]],
    publish_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for result in build_results:
        request = result["request"]
        common = {
            "instance_id": request.get("instance_id"),
            "repo": request.get("repo"),
            "base_image": request.get("base_image"),
            "registry_image": expected_registry_tag(BuildRequest(**request)),
            "sandbox_id": result.get("sandbox_id"),
        }
        if not result["success"]:
            failures.append(
                {
                    **common,
                    "stage": "build",
                    "error": result.get("error") or "Unknown build failure",
                }
            )
        if result.get("result_callback_error"):
            failures.append(
                {
                    **common,
                    "stage": "publish-submit",
                    "error": result["result_callback_error"],
                }
            )

    for result in publish_results or []:
        if result["success"]:
            continue
        failures.append(
            {
                "stage": "publish",
                "instance_id": result.get("instance_id"),
                "repo": result.get("repo"),
                "base_image": result.get("base_image"),
                "registry_image": result["image_ref"],
                "modal_image": modal_image_name(result["image_ref"]),
                "sandbox_id": result.get("sandbox_id"),
                "error": result.get("error") or "Unknown publish failure",
            }
        )
    return failures


def _print_summary(
    results: list[dict[str, Any]], *, wall_clock_seconds: float | None = None
) -> None:
    durations = [float(result["duration_seconds"]) for result in results]
    successful = sum(bool(result["success"]) for result in results)
    vm_durations: dict[str, float] = {}
    for index, result in enumerate(results):
        sandbox_id = str(result.get("sandbox_id") or f"unknown-{index}")
        vm_durations[sandbox_id] = float(
            result.get("batch_duration_seconds", result["duration_seconds"])
        )
    print(
        json.dumps(
            {
                "total": len(results),
                "successful": successful,
                "failed": len(results) - successful,
                "vm_count": len(vm_durations),
                "wall_clock_seconds": wall_clock_seconds,
                "wall_clock_proxy_seconds": (
                    wall_clock_seconds
                    if wall_clock_seconds is not None
                    else max(vm_durations.values(), default=0)
                ),
                "cumulative_seconds": round(sum(durations), 3),
                "cumulative_image_build_seconds": round(sum(durations), 3),
                "cumulative_vm_seconds": round(sum(vm_durations.values()), 3),
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def main(
    method: str = "push-vm-batch",
    dataset: str = "nebius/SWE-rebench",
    split: str = "filtered",
    n_limit: int = 0,
    image_limit: int = 8,
    select: str = "",
    repos: str = "",
    images_per_repo: int = 0,
    destination_image: str = REGISTRY_IMAGE,
    results_file: str = "",
    publish_ref: str = "",
    random_seed: int = 20260830,
    max_workers: int = 8,
    publish_max_workers: int = 8,
    publish_timeout_seconds: int = PUBLISH_TIMEOUT_SECONDS,
    max_images_per_vm: int = MAX_IMAGES_PER_VM,
    cpu: float = 2.0,
    memory: int = 8192,
    build_timeout_seconds: int = BUILD_TIMEOUT_SECONDS,
    sandbox_timeout_seconds: int = SANDBOX_TIMEOUT_SECONDS,
    sandbox_v2: bool = False,
    dry_run: bool = False,
    force_build: bool = True,
    publish_named_images: bool = True,
    stream_build_logs: bool = False,
    stream_publish_logs: bool = False,
    silent: bool = False,
) -> None:
    if sandbox_v2:
        os.environ["MODAL_SANDBOX_V2"] = "1"

    if method in {"vm", "push-vm-batch", "push-vm-repo-batch"}:
        selected_repos = parse_repos(repos)
        specs = collect_unique_image_specs(
            dataset=dataset,
            split=split,
            n_limit=n_limit or None,
            selected_instances_file=select or None,
            image_limit=None if selected_repos else image_limit or None,
        )
        if selected_repos:
            specs = select_image_specs_by_repo(
                specs,
                repos=selected_repos,
                images_per_repo=images_per_repo or max_images_per_vm,
            )
        requests = build_requests(
            specs,
            push=method in {"push-vm-batch", "push-vm-repo-batch"},
            force_build=force_build,
            destination_image=destination_image,
        )
        config = SandboxBuildConfig(
            cpu=cpu,
            memory=memory,
            build_timeout_seconds=build_timeout_seconds,
            sandbox_timeout_seconds=sandbox_timeout_seconds,
            stream_build_logs=stream_build_logs,
        )
        publisher: ParallelNamedImagePublisher | None = None
        if method == "push-vm-repo-batch":
            if not silent:
                print_repo_batch_plan(
                    requests,
                    max_images_per_vm=max_images_per_vm,
                    details=dry_run,
                )
            if dry_run:
                return
        publish_builder_runtime_image(announce=not silent)
        if publish_named_images and any(request.push for request in requests):
            publisher = ParallelNamedImagePublisher(
                max_workers=publish_max_workers,
                show_progress=not silent,
                stream_output=stream_publish_logs,
                timeout_seconds=publish_timeout_seconds,
            )

        run_output_dir = _create_results_dir()
        publish_started = time.monotonic()
        builds_started = time.monotonic()
        if method == "push-vm-repo-batch":
            results = run_repo_batched_vm_builds(
                requests,
                max_workers=max_workers,
                max_images_per_vm=max_images_per_vm,
                config=config,
                show_progress=not silent,
                on_result=(publisher.submit_build_result if publisher else None),
                output_dir=run_output_dir,
            )
        else:
            results = run_vm_builds(
                requests,
                max_workers=max_workers,
                config=config,
                show_progress=not silent,
                on_result=(publisher.submit_build_result if publisher else None),
                output_dir=run_output_dir,
            )
        build_wall_clock_seconds = round(time.monotonic() - builds_started, 3)
        if not silent:
            _print_summary(
                results,
                wall_clock_seconds=build_wall_clock_seconds,
            )
        publish_results: list[dict[str, Any]] | None = None
        if publisher is not None:
            publish_results = publisher.finish()
            publish_wall_clock_seconds = round(time.monotonic() - publish_started, 3)
            _write_results(
                "modal-publish",
                publish_results,
                announce=not silent,
                output_dir=run_output_dir,
            )
            if not silent:
                _print_publish_summary(
                    publish_results,
                    wall_clock_seconds=publish_wall_clock_seconds,
                )
        failures = collect_run_failures(results, publish_results)
        if failures:
            _write_results(
                "failures", failures, announce=not silent, output_dir=run_output_dir
            )
        _require_complete_run(results, publish_results)
    elif method == "publish":
        if not publish_ref:
            raise ValueError("--publish-ref is required for method=publish")
        publish_registry_image(publish_ref, timeout_seconds=publish_timeout_seconds)
    elif method == "publish-random-result":
        if not results_file:
            raise ValueError(
                "--results-file is required for method=publish-random-result"
            )
        refs = registry_refs_from_results(Path(results_file))
        run_output_dir = _create_results_dir()
        if not refs:
            raise ValueError("No successful pushed images found")
        publish_registry_image(
            random.Random(random_seed).choice(refs),
            timeout_seconds=publish_timeout_seconds,
        )
    elif method in {"publish-results", "publish-dataset"}:
        if method == "publish-results":
            if not results_file:
                raise ValueError(
                    "--results-file is required for method=publish-results"
                )
            refs = registry_refs_from_results(Path(results_file))
        else:
            specs = collect_unique_image_specs(
                dataset=dataset,
                split=split,
                n_limit=n_limit or None,
                selected_instances_file=select or None,
                image_limit=image_limit or None,
            )
            refs = [
                expected_registry_tag(request)
                for request in build_requests(
                    specs,
                    push=True,
                    force_build=False,
                    destination_image=destination_image,
                )
            ]
        run_output_dir = _create_results_dir()
        publish_started = time.monotonic()
        publish_results = publish_registry_images(
            refs,
            max_workers=publish_max_workers,
            show_progress=not silent,
            stream_output=stream_publish_logs,
            timeout_seconds=publish_timeout_seconds,
        )
        _write_results(
            "modal-publish",
            publish_results,
            announce=not silent,
            output_dir=run_output_dir,
        )
        failures = collect_run_failures([], publish_results)
        if failures:
            _write_results(
                "failures", failures, announce=not silent, output_dir=run_output_dir
            )
        if not silent:
            _print_publish_summary(
                publish_results,
                wall_clock_seconds=round(time.monotonic() - publish_started, 3),
            )
        if any(not bool(result["success"]) for result in publish_results):
            raise RuntimeError("One or more Modal named-image publications failed")
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    raise SystemExit("Run this script with `modal run`, not directly with Python.")
