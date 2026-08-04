#!/usr/bin/env python3
"""Mirror SWE-rebench base images from Docker Hub to GHCR with skopeo."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmarks.swerebench.apptainer_build2 import collect_unique_base_images


def split_image_ref(image: str) -> tuple[str, str]:
    """Split an image ref into name and tag, defaulting untagged refs to latest."""
    last_part = image.rsplit("/", maxsplit=1)[-1]
    if ":" in last_part:
        name, tag = image.rsplit(":", maxsplit=1)
        return name, tag
    return image, "latest"


def safe_image_name(name: str) -> str:
    """Return a GHCR-compatible repository name derived from a source image."""
    name = name.lower()
    name = re.sub(r"^docker\.io/", "", name)
    name = re.sub(r"^library/", "", name)
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    return name.strip("-")


def ghcr_ref(source: str, namespace: str, prefix: str) -> str:
    """Return the destination GHCR image reference for a source image."""
    name, tag = split_image_ref(source)
    return f"ghcr.io/{namespace}/{prefix}{safe_image_name(name)}:{tag}"


def run(cmd: Sequence[str], dry_run: bool) -> subprocess.CompletedProcess[str]:
    """Run a command, or print it in dry-run mode."""
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, check=False, text=True)


def destination_exists(destination: str, authfile: Path, dry_run: bool) -> bool:
    """Return whether the destination image already exists in GHCR."""
    cmd = [
        "skopeo",
        "inspect",
        "--authfile",
        str(authfile),
        f"docker://{destination}",
    ]
    proc = run(cmd, dry_run)
    return proc.returncode == 0


def run_skopeo_copy(
    source: str,
    destination: str,
    authfile: Path,
    *,
    all_platforms: bool,
    dry_run: bool,
) -> subprocess.CompletedProcess[str]:
    """Copy a source image to a destination registry with skopeo."""
    cmd = [
        "skopeo",
        "copy",
        "--authfile",
        str(authfile),
    ]
    if all_platforms:
        cmd.append("--all")
    else:
        cmd.extend(["--override-os", "linux", "--override-arch", "amd64"])
    cmd.extend(
        [
            f"docker://{source}",
            f"docker://{destination}",
        ]
    )
    return run(cmd, dry_run)


def load_completed_sources(map_file: Path) -> set[str]:
    """Load source refs that already have a completed map entry."""
    if not map_file.exists():
        return set()
    completed: set[str] = set()
    with map_file.open(encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.split()
            if len(parts) == 2:
                completed.add(parts[0])
    return completed


def append_mapping(map_file: Path, source: str, destination: str) -> None:
    """Append one successful source-to-destination mapping."""
    with map_file.open("a", encoding="utf-8") as f:
        f.write(f"{source} {destination}\n")


def append_failure(
    failures_file: Path, source: str, destination: str, code: int
) -> None:
    """Append one failed source-to-destination copy attempt."""
    with failures_file.open("a", encoding="utf-8") as f:
        f.write(f"{source} {destination} exit_code={code}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror SWE-rebench filtered split Docker images to GHCR."
    )
    parser.add_argument("--dataset", default="nebius/SWE-rebench")
    parser.add_argument("--split", default="filtered")
    parser.add_argument("--ghcr-namespace", default=os.getenv("GHCR_NAMESPACE"))
    parser.add_argument("--prefix", default="swerebench-")
    parser.add_argument(
        "--authfile",
        default=os.getenv(
            "REGISTRY_AUTH_FILE",
            str(Path.home() / ".config" / "containers" / "auth.json"),
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="scripts",
        help="Directory for source image list and source-to-GHCR map.",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="Limit rows before deduplicating images, for smoke tests.",
    )
    parser.add_argument(
        "--select",
        default=None,
        help="Optional selected instances file, matching apptainer_build2.py.",
    )
    parser.add_argument(
        "--all-platforms",
        action="store_true",
        help="Copy every platform in the source manifest list instead of linux/amd64.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue mirroring remaining images after a skopeo copy failure.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not inspect GHCR and skip destinations that already exist.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    namespace = args.ghcr_namespace or os.getenv("GH_USER")
    if not namespace:
        print(
            "error: set --ghcr-namespace or GHCR_NAMESPACE/GH_USER",
            file=sys.stderr,
        )
        return 2

    authfile = Path(args.authfile).expanduser()
    if not authfile.exists() and not args.dry_run:
        print(f"error: authfile does not exist: {authfile}", file=sys.stderr)
        print("run: skopeo login --authfile <path> ghcr.io ...", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_file = out_dir / "source-images.txt"
    map_file = out_dir / "ghcr-image-map.txt"
    failures_file = out_dir / "ghcr-image-failures.txt"

    base_images = collect_unique_base_images(
        dataset=args.dataset,
        split=args.split,
        n_limit=args.n_limit if args.n_limit else None,
        selected_instances_file=args.select,
    )
    sources = sorted({image for image, _install in base_images})

    images_file.write_text("\n".join(sources) + "\n", encoding="utf-8")
    completed = load_completed_sources(map_file)

    failures = 0
    for source in sources:
        destination = ghcr_ref(source, namespace, args.prefix)
        if source in completed:
            print(f"SKIP mapped: {source} -> {destination}", flush=True)
            continue
        if not args.no_skip_existing and destination_exists(
            destination, authfile, args.dry_run
        ):
            print(f"SKIP exists: {source} -> {destination}", flush=True)
            append_mapping(map_file, source, destination)
            completed.add(source)
            continue

        print(f"{source} -> {destination}", flush=True)
        proc = run_skopeo_copy(
            source,
            destination,
            authfile,
            all_platforms=args.all_platforms,
            dry_run=args.dry_run,
        )
        if proc.returncode != 0:
            failures += 1
            append_failure(failures_file, source, destination, proc.returncode)
            if not args.continue_on_error:
                print(
                    f"FAILED {source}: skopeo exited {proc.returncode}", file=sys.stderr
                )
                return proc.returncode
            print(f"FAILED {source}: skopeo exited {proc.returncode}", file=sys.stderr)
            continue

        append_mapping(map_file, source, destination)
        completed.add(source)

    print(f"Wrote {images_file}")
    print(f"Wrote {map_file}")
    if failures:
        print(f"Wrote {failures_file}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
