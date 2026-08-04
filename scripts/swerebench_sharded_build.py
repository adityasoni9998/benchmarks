#!/usr/bin/env python3
"""Shard SWE-rebench Apptainer image builds across Slurm array tasks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _patch_apptainer_build_subprocess(module: Any, mksquashfs_processors: int) -> None:
    if mksquashfs_processors <= 0:
        return

    real_run = module.subprocess.run
    mksquashfs_args = f"-processors {mksquashfs_processors}"

    def patched_run(cmd: Any, *args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        if (
            isinstance(cmd, list)
            and len(cmd) >= 2
            and cmd[0] == "apptainer"
            and cmd[1] == "build"
            and "--mksquashfs-args" not in cmd
        ):
            cmd = [cmd[0], cmd[1], "--mksquashfs-args", mksquashfs_args, *cmd[2:]]
        return real_run(cmd, *args, **kwargs)

    module.subprocess.run = patched_run


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--dataset", default="nebius/SWE-rebench")
    parser.add_argument("--split", default="filtered")
    parser.add_argument("--n-limit", type=int, default=0)
    parser.add_argument("--image-limit", type=int, default=0)
    parser.add_argument("--select", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sif-dir", required=True)
    parser.add_argument(
        "--final-sif-dir",
        default=None,
        help="Build in --sif-dir, then copy successful SIFs here atomically.",
    )
    parser.add_argument("--apptainer-cache-dir", required=True)
    parser.add_argument("--apptainer-tmp-dir", required=True)
    parser.add_argument("--definition-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument(
        "--mksquashfs-processors",
        type=int,
        default=0,
        help="Pass '-processors N' to mksquashfs via apptainer build.",
    )
    return parser


def main() -> int:
    args = get_parser().parse_args()

    repo_dir = Path(args.repo_dir).expanduser().resolve()
    sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

    from benchmarks.swerebench import apptainer_build as build_mod

    _patch_apptainer_build_subprocess(build_mod, args.mksquashfs_processors)

    shard_index = (
        args.shard_index
        if args.shard_index is not None
        else _env_int("SLURM_ARRAY_TASK_ID", 0)
    )
    shard_count = (
        args.shard_count
        if args.shard_count is not None
        else _env_int("SLURM_ARRAY_TASK_COUNT", 1)
    )
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard index {shard_index} outside [0, {shard_count})")

    os.environ["APPTAINER_CACHEDIR"] = str(Path(args.apptainer_cache_dir).expanduser())
    os.environ["APPTAINER_TMPDIR"] = str(Path(args.apptainer_tmp_dir).expanduser())
    os.environ.setdefault("OPENHANDS_APPTAINER_UV_CONCURRENT_DOWNLOADS", "2")
    os.environ.setdefault("OPENHANDS_APPTAINER_UV_CONCURRENT_BUILDS", "1")
    os.environ.setdefault("OPENHANDS_APPTAINER_UV_CONCURRENT_INSTALLS", "1")

    image_specs = build_mod.collect_unique_image_specs(
        dataset=args.dataset,
        split=args.split,
        n_limit=args.n_limit if args.n_limit else None,
        selected_instances_file=args.select,
        image_limit=args.image_limit if args.image_limit else None,
    )
    shard_specs = [
        image_spec
        for i, image_spec in enumerate(image_specs)
        if i % shard_count == shard_index
    ]

    final_sif_dir = (
        Path(args.final_sif_dir).expanduser() if args.final_sif_dir else None
    )
    if final_sif_dir is not None and not args.force_build:
        shard_specs = [
            image_spec
            for image_spec in shard_specs
            if not build_mod.apptainer_agent_image_path(
                image_spec.custom_tag,
                sif_dir=final_sif_dir,
            ).exists()
        ]

    print(
        json.dumps(
            {
                "event": "shard_start",
                "total_images": len(image_specs),
                "shard_images": len(shard_specs),
                "shard_index": shard_index,
                "shard_count": shard_count,
                "workers": args.workers,
                "mksquashfs_processors": args.mksquashfs_processors,
                "sif_dir": args.sif_dir,
                "final_sif_dir": str(final_sif_dir) if final_sif_dir else None,
                "apptainer_cache_dir": args.apptainer_cache_dir,
                "apptainer_tmp_dir": args.apptainer_tmp_dir,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    results = build_mod.build_apptainer_agent_images(
        image_specs=shard_specs,
        max_workers=args.workers,
        sif_dir=args.sif_dir,
        apptainer_cache_dir=args.apptainer_cache_dir,
        apptainer_tmp_dir=args.apptainer_tmp_dir,
        definition_dir=args.definition_dir,
        log_dir=args.log_dir,
        force_build=args.force_build,
        show_progress=True,
    )

    if final_sif_dir is not None:
        final_sif_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            if result.error is not None or not result.tags:
                continue
            local_path = Path(result.tags[0])
            final_path = final_sif_dir / local_path.name
            tmp_final_path = final_path.with_suffix(".tmp.sif")
            if final_path.exists() and not args.force_build:
                result.tags[0] = str(final_path)
                continue
            shutil.copy2(local_path, tmp_final_path)
            tmp_final_path.replace(final_path)
            result.tags[0] = str(final_path)

    if args.manifest:
        manifest = Path(args.manifest).expanduser()
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result.model_dump()) + "\n")

    failures = [result for result in results if result.error is not None]
    print(
        json.dumps(
            {
                "event": "shard_done",
                "built": len(results) - len(failures),
                "failed": len(failures),
                "shard_index": shard_index,
                "shard_count": shard_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
