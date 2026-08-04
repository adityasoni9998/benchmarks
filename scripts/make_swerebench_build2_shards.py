#!/usr/bin/env python3
"""Write SWE-rebench instance-id shards for Apptainer image builds.

The build script deduplicates rows to unique Docker images before building.
This shard generator therefore assigns all rows for a given docker_image to the
same shard, which avoids multiple Slurm array tasks building the same SIF path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SWEREBENCH_DATASET_COLUMNS = [
    "instance_id",
    "docker_image",
    "install_config",
]


def _load_dataset_frame(dataset_name: str, split: str) -> pd.DataFrame:
    path = Path(dataset_name).expanduser()
    if path.is_file():
        if path.suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if path.suffix == ".json":
            return pd.read_json(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported local dataset file type: {path}")

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    parquet_files = [
        repo_path
        for repo_path in api.list_repo_files(dataset_name, repo_type="dataset")
        if repo_path.endswith(".parquet")
        and _split_matches_parquet_path(repo_path, split)
    ]
    if not parquet_files:
        raise ValueError(
            f"No parquet files found for split {split!r} in {dataset_name}"
        )

    frames: list[pd.DataFrame] = []
    for parquet_file in sorted(parquet_files):
        local_file = hf_hub_download(
            repo_id=dataset_name,
            filename=parquet_file,
            repo_type="dataset",
        )
        frames.append(pd.read_parquet(local_file, columns=SWEREBENCH_DATASET_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _split_matches_parquet_path(path: str, split: str) -> bool:
    file_name = Path(path).name
    return (
        file_name == f"{split}.parquet"
        or file_name.startswith(f"{split}-")
        or f"/{split}/" in f"/{path}/"
    )


def _install_command(value: Any) -> str:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        return ""
    install = value.get("install")
    return install if isinstance(install, str) else ""


def _write_shards(frame: pd.DataFrame, output_dir: Path, shard_count: int) -> None:
    required_columns = set(SWEREBENCH_DATASET_COLUMNS)
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    image_to_install: dict[str, str] = {}
    groups: list[tuple[str, list[str]]] = []
    for docker_image, group in frame.groupby("docker_image", sort=True):
        install_commands = {
            _install_command(value) for value in group["install_config"].tolist()
        }
        if len(install_commands) != 1:
            raise ValueError(
                "Refusing to shard dataset because one docker_image has multiple "
                f"install commands: {docker_image}"
            )
        image_to_install[str(docker_image)] = next(iter(install_commands))
        instance_ids = sorted(str(value) for value in group["instance_id"].tolist())
        groups.append((str(docker_image), instance_ids))

    groups.sort(key=lambda item: (-len(item[1]), item[0]))
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    shard_sizes = [0 for _ in range(shard_count)]
    shard_images = [0 for _ in range(shard_count)]
    for _, instance_ids in groups:
        shard_index = min(range(shard_count), key=lambda i: (shard_sizes[i], i))
        shards[shard_index].extend(instance_ids)
        shard_sizes[shard_index] += len(instance_ids)
        shard_images[shard_index] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for shard_index, instance_ids in enumerate(shards):
        shard_file = output_dir / f"shard_{shard_index:03d}.txt"
        shard_file.write_text("\n".join(sorted(instance_ids)) + "\n", encoding="utf-8")

    manifest = {
        "shard_count": shard_count,
        "total_instances": int(sum(shard_sizes)),
        "total_images": len(groups),
        "shards": [
            {
                "index": index,
                "instances": shard_sizes[index],
                "images": shard_images[index],
                "select_file": str(output_dir / f"shard_{index:03d}.txt"),
            }
            for index in range(shard_count)
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create disjoint SWE-rebench instance-id shards for build2."
    )
    parser.add_argument("--dataset", default="nebius/SWE-rebench")
    parser.add_argument("--split", default="filtered")
    parser.add_argument("--shard-count", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="Optional deterministic smoke-test limit before sharding.",
    )
    return parser


def main() -> int:
    args = get_parser().parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")

    frame = _load_dataset_frame(args.dataset, args.split)
    if args.n_limit > 0:
        frame = frame.sample(n=min(args.n_limit, len(frame)), random_state=42)

    _write_shards(
        frame=frame,
        output_dir=Path(args.output_dir).expanduser(),
        shard_count=args.shard_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
