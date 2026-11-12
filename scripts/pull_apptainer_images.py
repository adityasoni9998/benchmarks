"""Convert a pinned JSONL image manifest to SIFs with bounded concurrency.

Each row requires instance_id and image (an OCI reference including @sha256:).
SIFs are written directly to their final names. Completion records, rather than
file existence alone, make interrupted batches resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


STOP = threading.Event()


@dataclass(frozen=True)
class Image:
    instance_id: str
    image: str


def write_record(path: Path, result: dict[str, object]) -> None:
    # Preserve the previous record if the process stops during a write.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result) + "\n")
    temporary.replace(path)


def read_manifest(path: Path) -> list[Image]:
    images: list[Image] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        instance_id, image = row["instance_id"], row["image"]
        if not isinstance(instance_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", instance_id
        ):
            raise ValueError(f"Unsafe instance_id: {instance_id!r}")
        if not isinstance(image, str) or not re.fullmatch(
            r"(?:docker://)?[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}", image
        ):
            raise ValueError(f"Expected digest-pinned OCI image for {instance_id}")
        if instance_id in seen:
            raise ValueError(f"Duplicate instance_id: {instance_id}")
        seen.add(instance_id)
        images.append(Image(instance_id, image.removeprefix("docker://")))
    return images


@contextmanager
def build_environment() -> Iterator[dict[str, str]]:
    # Apptainer 1.5.5's OCI library rewrites index.json non-atomically.
    # Give each process its own OCI index, retaining shared digest-addressed blobs.
    root = Path(os.environ["APPTAINER_CACHEDIR"]).resolve()
    blobs = root / "cache/blob/blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    metadata = root / "build-metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=metadata) as private:
        layout = Path(private) / "cache/blob"
        layout.mkdir(parents=True)
        (layout / "blobs").symlink_to(blobs, target_is_directory=True)
        yield {**os.environ, "APPTAINER_CACHEDIR": private}


def run_command(command: list[str], log: Path, timeout: int) -> int:
    with build_environment() as environment, log.open("a") as output:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        deadline = time.monotonic() + timeout
        while not STOP.is_set() and time.monotonic() < deadline:
            try:
                return process.wait(
                    timeout=min(2, max(0.01, deadline - time.monotonic()))
                )
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return 130 if STOP.is_set() else 124


def convert(
    image: Image,
    sif_dir: Path,
    report_dir: Path,
    flags: str,
    retries: int,
    timeout: int,
) -> dict[str, object]:
    if STOP.is_set():
        return {"instance_id": image.instance_id, "status": "interrupted"}
    sif = sif_dir / f"{image.instance_id}.sif"
    record = sif_dir / ".completed" / f"{image.instance_id}.json"
    previous = json.loads(record.read_text()) if record.exists() else {}
    if sif.exists():
        if previous.get("image") != image.image:
            raise ValueError(f"Refusing to overwrite untracked/different image: {sif}")
        if (
            previous.get("status") == "ok"
            and previous.get("bytes") == sif.stat().st_size
        ):
            return {**previous, "status": "skipped"}
    start = time.monotonic()
    result: dict[str, object] = {
        "instance_id": image.instance_id,
        "image": image.image,
        "flags": flags,
        "sif": str(sif),
        "status": "running",
    }
    write_record(record, result)
    for attempt in range(retries + 1):
        log = report_dir / f"{image.instance_id}.attempt-{attempt + 1}.log"
        command = ["apptainer", "build"]
        if sif.exists():
            command.append("--force")
        command += ["--mksquashfs-args", flags, str(sif), f"docker://{image.image}"]
        code = run_command(command, log, timeout)
        result.update(attempts=attempt + 1, exit_code=code, log=str(log))
        if code == 0 and sif.is_file() and sif.stat().st_size > 0:
            result.update(status="ok", bytes=sif.stat().st_size)
            break
        if STOP.is_set():
            result["status"] = "interrupted"
            break
        result["status"] = "failed"
        if attempt < retries:
            STOP.wait(5 * 2**attempt)
    result["seconds"] = round(time.monotonic() - start, 3)
    write_record(record, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sif-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--mksquashfs-args",
        default="-comp zstd -Xcompression-level 3 -processors 2 -mem 1G",
    )
    args = parser.parse_args()
    if (
        min(args.workers, args.timeout) < 1
        or min(args.offset, args.limit, args.retries) < 0
    ):
        parser.error(
            "workers/timeout must be positive; offset/limit/retries nonnegative"
        )
    for variable in ("APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"):
        if not os.environ.get(variable):
            parser.error(f"Set {variable} first (source scripts/apptainer_abci_env.sh)")
    images = read_manifest(args.manifest)[args.offset :]
    if args.limit:
        images = images[: args.limit]
    os.umask(0o077)
    signal.signal(signal.SIGINT, lambda _signum, _frame: STOP.set())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: STOP.set())
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.sif_dir / ".completed").mkdir(parents=True, exist_ok=True)
    print(
        f"Selected {len(images)} images; at most {args.workers} concurrent builds; "
        "matching successful completion records will be skipped.",
        flush=True,
    )
    start = time.monotonic()
    results: list[dict[str, object]] = []
    with (
        ThreadPoolExecutor(max_workers=args.workers) as pool,
        (args.report_dir / "results.jsonl").open("a") as output,
    ):
        jobs = {
            pool.submit(
                convert,
                image,
                args.sif_dir,
                args.report_dir,
                args.mksquashfs_args,
                args.retries,
                args.timeout,
            ): image
            for image in images
        }
        for future in as_completed(jobs):
            result: dict[str, object]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "instance_id": jobs[future].instance_id,
                    "status": "failed",
                    "error": str(error),
                }
            results.append(result)
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(json.dumps(result), flush=True)
    elapsed = time.monotonic() - start
    completed = sum(r["status"] == "ok" for r in results)
    failed = sum(r["status"] == "failed" for r in results)
    summary = {
        "workers": args.workers,
        "flags": args.mksquashfs_args,
        "elapsed_seconds": elapsed,
        "completed": completed,
        "failed": failed,
        "skipped": sum(r["status"] == "skipped" for r in results),
        "interrupted": sum(r["status"] == "interrupted" for r in results),
        "images_per_hour": completed * 3600 / elapsed if elapsed else 0,
    }
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    raise SystemExit(130 if STOP.is_set() else 1 if failed else 0)


def run_with_restarts(max_restarts: int = 100, delay: float = 60) -> None:
    """Resume after unexpected runner errors, but never restart an intentional exit."""
    restarts = 0
    while not STOP.is_set():
        try:
            main()
            return
        except Exception:
            # main's executor context joins its workers before reaching here,
            # so a restarted pass cannot overlap builds from the previous pass.
            traceback.print_exc()
            if STOP.is_set():
                break
            if restarts >= max_restarts:
                print("Runner restart limit reached.", file=sys.stderr, flush=True)
                raise SystemExit(1) from None
            restarts += 1
            print(
                f"Runner restart {restarts}/{max_restarts} in {delay:g}s; "
                "completed images will be skipped.",
                file=sys.stderr,
                flush=True,
            )
            STOP.wait(delay)
    raise SystemExit(130)


if __name__ == "__main__":
    run_with_restarts()
