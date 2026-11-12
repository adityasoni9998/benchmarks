"""Build Apptainer agent-server SIFs for SWE-bench instances, in parallel and resumably.

Each SIF is built by ensure_apptainer_agent_image (the official SWE-bench image plus the
agent server from the vendored SDK) under $OPENHANDS_APPTAINER_BUILD_ROOT, then linked as
<sif-dir>/<instance_id>.sif: the layout run_infer reads via OPENHANDS_APPTAINER_SIF_DIR.
Instances whose link already exists are skipped, so an interrupted batch can be rerun.

Example:
    uv run python -m benchmarks.swebench.build_apptainer_sifs \
        --sif-dir /groups/.../apptainer/sif_verified --workers 128
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from benchmarks.swebench import constants
from benchmarks.swebench.apptainer_build import ensure_apptainer_agent_image
from benchmarks.swebench.build_images import (
    extract_custom_tag,
    get_official_docker_image,
    should_wrap_instance_id,
)
from benchmarks.utils.dataset import get_dataset


def build_one(instance_id: str, sif_dir: Path) -> dict[str, Any]:
    link = sif_dir / f"{instance_id}.sif"
    if link.exists():
        return {
            "instance_id": instance_id,
            "status": "exists",
            "sif": str(link.resolve()),
        }
    start = time.time()
    base_image = get_official_docker_image(instance_id)
    try:
        path = ensure_apptainer_agent_image(
            base_image=base_image,
            custom_tag=extract_custom_tag(base_image),
            target=constants.DEFAULT_BUILD_TARGET,
            wrap_swebench_deps=should_wrap_instance_id(instance_id),
        )
    except Exception as error:
        return {
            "instance_id": instance_id,
            "status": "failed",
            "error": str(error),
            "seconds": round(time.time() - start, 1),
        }
    tmp_link = link.with_name(link.name + ".tmp")
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(path)
    tmp_link.replace(link)
    return {
        "instance_id": instance_id,
        "status": "built",
        "sif": str(path),
        "seconds": round(time.time() - start, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Apptainer agent-server SIFs for SWE-bench instances."
    )
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sif-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--select", default=None, help="File of instance ids to build (one per line)."
    )
    args = parser.parse_args()

    df = get_dataset(
        dataset_name=args.dataset,
        split=args.split,
        eval_limit=args.limit,
        selected_instances_file=args.select,
    )
    instance_ids = [str(i) for i in df["instance_id"]]
    args.sif_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.sif_dir / "build_status.jsonl"
    lock = threading.Lock()
    counts: dict[str, int] = {}
    print(f"building {len(instance_ids)} SIFs with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_one, i, args.sif_dir) for i in instance_ids]
        for future in as_completed(futures):
            record = future.result()
            with lock:
                counts[record["status"]] = counts.get(record["status"], 0) + 1
                with status_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(
                    f"{sum(counts.values())}/{len(instance_ids)} {counts} "
                    f"{record['instance_id']} {record['status']} {record.get('seconds', '')}",
                    flush=True,
                )
    if counts.get("failed"):
        raise SystemExit(f"{counts['failed']} builds failed; see {status_path}")


if __name__ == "__main__":
    main()
