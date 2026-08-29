#!/usr/bin/env python3
"""Build SWE-Smith agent-server images in Modal VM Sandboxes.

Use ``--method push-vm-batch --base-images-file <path>`` for a registry-backed
batch. Manifests may be newline-delimited image names, JSONL records with an
``image_name`` field, or a JSON list containing either form. Publishing the
successful registry images into Modal is a separate ``publish-results`` step.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import modal
from modal.container_process import ContainerProcess


APP_NAME = "swesmith-modal-image-builder"
BENCHMARKS_REPOSITORY = "https://github.com/adityasoni9998/benchmarks.git"
BENCHMARKS_SHA = "42bbf91cb5270fd4305a0dcaf225b728bfd62ca6"
SDK_SHA = "5acdf05ae2fc6224f92114db6f557b181cd0f280"
SDK_SHORT_SHA = SDK_SHA[:7]
BUILDKIT_IMAGE = "moby/buildkit:v0.32.2"
DOCKERHUB_SECRET_NAME = "dockerhub-adityasoni8"
LOCAL_TEST_IMAGE = "local/swesmith-modal-build-test"
REGISTRY_IMAGE = "docker.io/adityasoni8/eval-agent-server"
BUILD_TIMEOUT_SECONDS = 45 * 60
SANDBOX_TIMEOUT_SECONDS = 60 * 60
MAX_MODAL_IMAGE_NAME_LENGTH = 64


# Deterministic sample from the first two published dataset shards.
VM_BASE_IMAGES = (
    "docker.io/swebench/swesmith.x86_64.agronholm_1776_typeguard.b6a7e438",
    "docker.io/swebench/swesmith.x86_64.cantools_1776_cantools.0c6a7871",
    "docker.io/swebench/swesmith.x86_64.cog-creators_1776_red-discordbot.33e0eac7",
    "docker.io/swebench/swesmith.x86_64.jd_1776_tenacity.0d40e76f",
    "docker.io/swebench/swesmith.x86_64.knio_1776_dominate.9082227e",
    "docker.io/swebench/swesmith.x86_64.madzak_1776_python-json-logger.5f85723f",
    "docker.io/swebench/swesmith.x86_64.marshmallow-code_1776_webargs.dbde72fe",
    "docker.io/swebench/swesmith.x86_64.pdfminer_1776_pdfminer.six.1a8bd2f7",
)


@dataclass(frozen=True)
class BuildRequest:
    base_image: str
    custom_tag: str
    destination_image: str
    push: bool = False


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


def custom_tag_for_base_image(base_image: str) -> str:
    return base_image.rsplit("/", 1)[-1].split(":", 1)[0]


def build_requests(
    base_images: Iterable[str],
    *,
    push: bool = False,
    destination_image: str | None = None,
) -> list[BuildRequest]:
    destination = destination_image or (REGISTRY_IMAGE if push else LOCAL_TEST_IMAGE)
    return [
        BuildRequest(
            base_image=base_image,
            custom_tag=custom_tag_for_base_image(base_image),
            destination_image=destination,
            push=push,
        )
        for base_image in base_images
    ]


def load_base_images(path: Path) -> list[str]:
    def image_name(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("image_name"), str):
            return value["image_name"]
        raise ValueError("Each image entry must be a string or contain image_name")

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("images"), list):
            entries = payload["images"]
        elif isinstance(payload, list):
            entries = payload
        else:
            raise ValueError("JSON manifests must be a list or contain an images list")
    elif path.suffix == ".jsonl":
        entries = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        entries = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    normalized = []
    for entry in entries:
        name = image_name(entry).lower().strip()
        if not name.startswith("docker.io/"):
            name = f"docker.io/{name}"
        normalized.append(name)
    return list(dict.fromkeys(normalized))


def expected_registry_tag(request: BuildRequest) -> str:
    return (
        f"{request.destination_image}:{SDK_SHORT_SHA}-{request.custom_tag}"
        "-source-minimal"
    )


def modal_image_name(image_ref: str) -> str:
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
registry_secret = modal.Secret.from_name(DOCKERHUB_SECRET_NAME)
builder_image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
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
    .workdir("/opt/benchmarks")
)


WORKER_CODE = """
import json
import os
import sys

from benchmarks.utils.build_utils import build_image

request = json.loads(sys.argv[1])
os.environ["BUILDKIT_PROGRESS"] = "plain"
os.environ["BUILDKIT_RESET_ON_FAILURE"] = "1"
os.environ["EXTENSIONS_REF"] = "main"
os.environ["OPENHANDS_BUILDKIT_CACHE_MODE"] = "off"
if request["push"]:
    os.environ["OPENHANDS_IMAGE_COMPRESSION"] = "estargz"

output = build_image(
    base_image=request["base_image"],
    target_image=request["destination_image"],
    custom_tag=request["custom_tag"],
    target="source-minimal",
    push=request["push"],
    force_build=True,
)
print(json.dumps({"tags": output.tags, "error": output.error}, sort_keys=True))
raise SystemExit(1 if output.error or not output.tags else 0)
"""


def _stream_process(
    process: ContainerProcess[str],
) -> tuple[str, str]:
    output: dict[str, list[str]] = {"stdout": [], "stderr": []}

    def consume(name: str) -> None:
        stream = process.stdout if name == "stdout" else process.stderr
        for line in stream:
            output[name].append(line)
            print(line, end="", file=sys.stdout if name == "stdout" else sys.stderr)

    stdout_thread = threading.Thread(target=consume, args=("stdout",))
    stderr_thread = threading.Thread(target=consume, args=("stderr",))
    stdout_thread.start()
    stderr_thread.start()
    process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return "".join(output["stdout"]), "".join(output["stderr"])


def _build_in_vm_sandbox(
    request: BuildRequest, config: SandboxBuildConfig
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    sandbox: modal.Sandbox | None = None
    sandbox_id: str | None = None
    tags: list[str] = []
    error: str | None = None
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
            tags={"method": "vm-sandbox", "sdk": SDK_SHORT_SHA},
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
                "printf '%s' \"$REGISTRY_PASSWORD\" | docker login "
                '--username "$REGISTRY_USERNAME" --password-stdin >/dev/null; '
                "docker buildx create --name swesmith-modal-sdk "
                "--driver docker-container "
                f"--driver-opt image={BUILDKIT_IMAGE} --use; "
                "docker buildx inspect --bootstrap"
            ),
            timeout=180,
        )
        setup_stdout, setup_stderr = _stream_process(setup)
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
            env={
                "OPENHANDS_SUPPRESS_BANNER": "1",
                "UV_LINK_MODE": "copy",
            },
        )
        stdout, stderr = _stream_process(process)
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
            try:
                sandbox.terminate(wait=True)
            finally:
                sandbox.detach()

    result = BuildResult(
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
    return asdict(result)


def _write_results(method: str, results: list[dict[str, Any]]) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(".agent_tmp") / f"modal-build-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method}.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {method} results to {output_path}")
    return output_path


def run_vm_builds(
    requests: list[BuildRequest],
    *,
    max_workers: int | None = None,
    config: SandboxBuildConfig | None = None,
) -> list[dict[str, Any]]:
    if not requests:
        return []
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    worker_count = min(max_workers or len(requests), len(requests))
    sandbox_config = config or SandboxBuildConfig()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_build_in_vm_sandbox, request, sandbox_config): request
            for request in requests
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"VM build completed success={result['success']} "
                f"base={futures[future].base_image}"
            )
    results.sort(key=lambda result: str(result["request"]))
    _write_results("vm-sandbox", results)
    return results


def publish_registry_image(image_ref: str) -> str:
    name = modal_image_name(image_ref)
    image = modal.Image.from_registry(image_ref, secret=registry_secret).entrypoint([])
    app_handle = modal.App.lookup(APP_NAME, create_if_missing=True)
    with modal.enable_output():
        image.build(app_handle).publish(name)
    print(f"Published {image_ref} as Modal Image {name}")
    return name


def registry_refs_from_results(path: Path) -> list[str]:
    results = json.loads(path.read_text())
    if not isinstance(results, list):
        raise ValueError("Build results must contain a JSON list")

    refs = []
    for result in results:
        if not isinstance(result, dict) or not result.get("success"):
            continue
        request_data = result.get("request")
        if not isinstance(request_data, dict):
            raise ValueError("Successful build result is missing its request")
        request = BuildRequest(**request_data)
        if not request.push:
            continue
        refs.append(expected_registry_tag(request))
    return list(dict.fromkeys(refs))


def publish_registry_images(
    image_refs: list[str], *, max_workers: int
) -> list[dict[str, Any]]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if not image_refs:
        return []

    def publish(image_ref: str) -> dict[str, Any]:
        started = time.monotonic()
        try:
            modal_image = publish_registry_image(image_ref)
            return {
                "duration_seconds": round(time.monotonic() - started, 3),
                "image_ref": image_ref,
                "modal_image": modal_image,
                "success": True,
            }
        except Exception as exc:
            return {
                "duration_seconds": round(time.monotonic() - started, 3),
                "image_ref": image_ref,
                "error": f"{type(exc).__name__}: {exc}",
                "success": False,
            }

    with ThreadPoolExecutor(max_workers=min(max_workers, len(image_refs))) as executor:
        return list(executor.map(publish, image_refs))


def _print_summary(results: list[dict[str, Any]]) -> None:
    successful = sum(bool(result["success"]) for result in results)
    durations = [float(result["duration_seconds"]) for result in results]
    print(
        json.dumps(
            {
                "total": len(results),
                "successful": successful,
                "failed": len(results) - successful,
                "wall_clock_proxy_seconds": max(durations, default=0),
                "cumulative_seconds": round(sum(durations), 3),
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def main(
    method: str = "vm",
    push_index: int = 0,
    publish_ref: str = "",
    base_images_file: str = "",
    results_file: str = "",
    destination_image: str = REGISTRY_IMAGE,
    max_workers: int = 8,
    cpu: float = 2.0,
    memory: int = 8192,
    build_timeout_seconds: int = BUILD_TIMEOUT_SECONDS,
    sandbox_timeout_seconds: int = SANDBOX_TIMEOUT_SECONDS,
    sandbox_v2: bool = False,
) -> None:
    if sandbox_v2:
        os.environ["MODAL_SANDBOX_V2"] = "1"

    sandbox_config = SandboxBuildConfig(
        cpu=cpu,
        memory=memory,
        build_timeout_seconds=build_timeout_seconds,
        sandbox_timeout_seconds=sandbox_timeout_seconds,
    )
    if method == "vm":
        _print_summary(
            run_vm_builds(
                build_requests(VM_BASE_IMAGES),
                max_workers=max_workers,
                config=sandbox_config,
            )
        )
    elif method == "push-vm":
        request = build_requests(VM_BASE_IMAGES, push=True)[push_index]
        results = run_vm_builds(
            [request], max_workers=max_workers, config=sandbox_config
        )
        _print_summary(results)
        if results[0]["success"]:
            publish_registry_image(expected_registry_tag(request))
    elif method == "push-vm-batch":
        if not base_images_file:
            raise ValueError("--base-images-file is required for method=push-vm-batch")
        requests = build_requests(
            load_base_images(Path(base_images_file)),
            push=True,
            destination_image=destination_image,
        )
        _print_summary(
            run_vm_builds(
                requests,
                max_workers=max_workers,
                config=sandbox_config,
            )
        )
    elif method == "publish":
        if not publish_ref:
            raise ValueError("--publish-ref is required for method=publish")
        publish_registry_image(publish_ref)
    elif method == "publish-results":
        if not results_file:
            raise ValueError("--results-file is required for method=publish-results")
        publish_results = publish_registry_images(
            registry_refs_from_results(Path(results_file)),
            max_workers=max_workers,
        )
        _write_results("publish", publish_results)
        _print_summary(publish_results)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    raise SystemExit("Run this script with `modal run`, not directly with Python.")
