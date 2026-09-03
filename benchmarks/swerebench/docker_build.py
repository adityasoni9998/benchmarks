"""Build Docker agent-server images for SWE-rebench base images.

Each SWE-rebench row supplies both a base image and the command that installs
the repository from ``/testbed``. The standard SDK image is built first, then a
small final layer repeats that install command so editable installs point at the
repository contained in the final image.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from benchmarks.swerebench.apptainer_build import (
    DEFAULT_BUILD_TARGET,
    SwerebenchImageSpec,
    collect_unique_image_specs,
)
from benchmarks.utils.build_utils import (
    BuildOutput,
    _get_sdk_submodule_info,
    _prepare_cached_sdist,
    build_image,
    default_build_output_dir,
    run_docker_build_layer,
)
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.image_utils import local_image_exists, remote_image_exists
from openhands.sdk import get_logger


logger = get_logger(__name__)

DOCKERFILE = Path(__file__).with_name("Dockerfile.testbed-reinstall")
INTERMEDIATE_IMAGE_SUFFIX = "-swerebench-unrepaired"
MAX_DOCKER_CUSTOM_TAG_LENGTH = 64


def docker_custom_tag(image_spec: SwerebenchImageSpec) -> str:
    """Return a repair-aware key short enough for every SDK Docker tag alias."""
    custom_tag = image_spec.custom_tag
    if len(custom_tag) <= MAX_DOCKER_CUSTOM_TAG_LENGTH:
        return custom_tag
    digest = hashlib.sha256(custom_tag.encode()).hexdigest()[:12]
    prefix_length = MAX_DOCKER_CUSTOM_TAG_LENGTH - len(digest) - 1
    return f"{custom_tag[:prefix_length]}-{digest}"


def primary_agent_image_tag(
    target_image: str,
    image_spec: SwerebenchImageSpec,
    target: str = DEFAULT_BUILD_TARGET,
) -> str:
    """Return the canonical short-SHA tag for a repaired Docker image."""
    _, git_sha, _ = _get_sdk_submodule_info()
    short_sha = git_sha[:7] if git_sha != "unknown" else "unknown"
    target_suffix = "" if target == "binary" else f"-{target}"
    return f"{target_image}:{short_sha}-{docker_custom_tag(image_spec)}{target_suffix}"


def _final_tags(
    intermediate_tags: list[str],
    intermediate_image: str,
    target_image: str,
    custom_tag: str,
) -> list[str]:
    """Map SDK intermediate tags to final tags, excluding its generic base tag."""
    prefix = f"{intermediate_image}:"
    return [
        f"{target_image}:{tag.removeprefix(prefix)}"
        for tag in intermediate_tags
        if tag.startswith(prefix) and custom_tag in tag.removeprefix(prefix)
    ]


def build_docker_agent_image(
    image_spec: SwerebenchImageSpec,
    *,
    target_image: str = EVAL_AGENT_SERVER_IMAGE,
    target: str = DEFAULT_BUILD_TARGET,
    push: bool = False,
    force_build: bool = False,
    cached_sdist: Path | None = None,
) -> BuildOutput:
    """Build one SWE-rebench Docker image and its testbed reinstall layer."""
    if target != DEFAULT_BUILD_TARGET:
        return BuildOutput(
            base_image=image_spec.base_image,
            tags=[],
            error=(
                "SWE-rebench Docker builds only support "
                f"{DEFAULT_BUILD_TARGET!r}, got {target!r}"
            ),
        )
    if image_spec.install_command is None:
        return BuildOutput(
            base_image=image_spec.base_image,
            tags=[],
            error="SWE-rebench image spec has no install command",
        )

    primary_tag = primary_agent_image_tag(target_image, image_spec, target)
    image_exists = remote_image_exists if push else local_image_exists
    if not force_build and image_exists(primary_tag):
        logger.info("Using existing SWE-rebench Docker image %s", primary_tag)
        return BuildOutput(
            base_image=image_spec.base_image,
            tags=[primary_tag],
            status="skipped_remote_exists" if push else "built",
            skip_reason="remote_image_exists" if push else "local_image_exists",
        )

    intermediate_image = f"{target_image}{INTERMEDIATE_IMAGE_SUFFIX}"
    custom_tag = docker_custom_tag(image_spec)
    intermediate_primary_tag = primary_agent_image_tag(
        intermediate_image, image_spec, target
    )
    if not push and not force_build and local_image_exists(intermediate_primary_tag):
        logger.info("Reusing local SDK intermediate %s", intermediate_primary_tag)
        sdk_output = BuildOutput(
            base_image=image_spec.base_image,
            tags=[intermediate_primary_tag],
        )
    else:
        sdk_output = build_image(
            base_image=image_spec.base_image,
            target_image=intermediate_image,
            custom_tag=custom_tag,
            target=target,
            push=push,
            force_build=force_build,
            cached_sdist=cached_sdist,
        )
    if sdk_output.error is not None or not sdk_output.tags:
        sdk_output.base_image = image_spec.base_image
        return sdk_output

    final_tags = _final_tags(
        sdk_output.tags,
        intermediate_image,
        target_image,
        custom_tag,
    )
    if not final_tags:
        # The SDK's remote-existence check can reuse its generic base-image tag
        # instead of a custom-tag alias. That intermediate is still valid for
        # the final reinstall layer; emit the canonical repair-specific tag.
        final_tags = [primary_tag]

    encoded_install_command = base64.b64encode(
        image_spec.install_command.encode()
    ).decode()
    output = run_docker_build_layer(
        dockerfile=DOCKERFILE,
        context=DOCKERFILE.parent,
        tags=final_tags,
        build_args={
            "SDK_IMAGE": sdk_output.tags[0],
            "SWEREBENCH_INSTALL_COMMAND_B64": encoded_install_command,
        },
        push=push,
        platform="linux/amd64",
        load=not push,
        # A docker-container builder cannot resolve images loaded in the host
        # daemon. Registry builds can use the active high-performance builder.
        builder=None if push else "default",
    )
    output.base_image = image_spec.base_image
    return output


def _build_one(
    image_spec: SwerebenchImageSpec,
    *,
    target_image: str,
    target: str,
    push: bool,
    force_build: bool,
    cached_sdist: Path | None,
) -> BuildOutput:
    return build_docker_agent_image(
        image_spec,
        target_image=target_image,
        target=target,
        push=push,
        force_build=force_build,
        cached_sdist=cached_sdist,
    )


def build_docker_agent_images(
    image_specs: list[SwerebenchImageSpec],
    *,
    target_image: str,
    target: str,
    push: bool,
    max_workers: int,
    force_build: bool,
) -> list[BuildOutput]:
    """Build SWE-rebench Docker images with a shared SDK source archive."""
    if push:
        # Match SWE-Smith: registry exports use seekable eStargz layers by default.
        os.environ.setdefault("OPENHANDS_IMAGE_COMPRESSION", "estargz")

    workers = max(1, max_workers)
    with _prepare_cached_sdist() as cached_sdist:
        if workers == 1:
            return [
                _build_one(
                    spec,
                    target_image=target_image,
                    target=target,
                    push=push,
                    force_build=force_build,
                    cached_sdist=cached_sdist,
                )
                for spec in tqdm(image_specs, desc="Building Docker images")
            ]

        results: list[BuildOutput] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _build_one,
                    spec,
                    target_image=target_image,
                    target=target,
                    push=push,
                    force_build=force_build,
                    cached_sdist=cached_sdist,
                ): spec
                for spec in image_specs
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Building Docker images",
            ):
                spec = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        BuildOutput(
                            base_image=spec.base_image,
                            tags=[],
                            error=repr(exc),
                            status="failed",
                        )
                    )
        return results


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build local or registry Docker images for SWE-rebench."
    )
    parser.add_argument("--dataset", default="nebius/SWE-rebench")
    parser.add_argument("--split", default="filtered")
    parser.add_argument("--n-limit", type=int, default=0)
    parser.add_argument(
        "--image-limit",
        type=int,
        default=0,
        help="Limit unique images after deduplication (0 = no limit).",
    )
    parser.add_argument("--select", default=None)
    parser.add_argument("--image", default=EVAL_AGENT_SERVER_IMAGE)
    parser.add_argument("--target", default=DEFAULT_BUILD_TARGET)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--manifest", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    image_specs = collect_unique_image_specs(
        dataset=args.dataset,
        split=args.split,
        n_limit=args.n_limit if args.n_limit else None,
        selected_instances_file=args.select,
        image_limit=args.image_limit if args.image_limit else None,
    )
    logger.info("Building %d unique SWE-rebench Docker images", len(image_specs))
    results = build_docker_agent_images(
        image_specs,
        target_image=args.image,
        target=args.target,
        push=args.push,
        max_workers=args.max_workers,
        force_build=args.force_build,
    )

    manifest = (
        Path(args.manifest).expanduser()
        if args.manifest
        else default_build_output_dir(args.dataset, args.split)
        / "docker-manifest.jsonl"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.model_dump()) + "\n")

    failures = [result for result in results if result.error or not result.tags]
    for result in results:
        if result.error:
            print(f"FAILED {result.base_image}: {result.error}", file=sys.stderr)
        else:
            print(f"BUILT {result.base_image}: {result.tags[0]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
