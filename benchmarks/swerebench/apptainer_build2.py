"""Local Apptainer builds for SWE-rebench agent-server images.

SWE-rebench rows carry the exact base image in their ``docker_image`` field.
Unlike SWE-bench, there are no per-repository dependency wrappers here; the
Apptainer definition installs the OpenHands agent server directly into that
base image.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from tqdm.auto import tqdm

from benchmarks.swebench.build_base_images import dockerfile_content_hash
from benchmarks.utils.build_utils import BuildOutput, _get_sdk_submodule_info
from benchmarks.utils.dataset import get_dataset
from openhands.sdk import get_logger


logger = get_logger(__name__)

TargetType = Literal["source-minimal"]

BUILD_TARGET_SOURCE_MINIMAL: TargetType = "source-minimal"
DEFAULT_BUILD_TARGET: TargetType = BUILD_TARGET_SOURCE_MINIMAL

DEFAULT_APPTAINER_BUILD_ROOT = (
    Path.home() / ".cache" / "openhands" / "swerebench-apptainer-agent-images"
)
SUPPORTED_APPTAINER_TARGETS = {BUILD_TARGET_SOURCE_MINIMAL}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sdk_root() -> Path:
    return _repo_root() / "vendor" / "software-agent-sdk"


def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _build_root() -> Path:
    return Path(
        os.getenv("OPENHANDS_APPTAINER_BUILD_ROOT", str(DEFAULT_APPTAINER_BUILD_ROOT))
    ).expanduser()


def _force_build_enabled() -> bool:
    return os.getenv("OPENHANDS_APPTAINER_FORCE_BUILD", "").lower() in {
        "1",
        "true",
        "yes",
    }


def docker_image_from_row(row: Mapping[str, object]) -> str:
    """Return the SWE-rebench base image recorded in a dataset row."""
    docker_image = row.get("docker_image")
    if not isinstance(docker_image, str) or not docker_image:
        raise ValueError("SWE-rebench row is missing a non-empty docker_image field")
    return docker_image


def install_command_from_row(row: Mapping[str, object]) -> str:
    """Return the SWE-rebench install command recorded in a dataset row."""
    return row["install_config"]["install"]  # type: ignore[index]


def extract_custom_tag(base_image: str) -> str:
    """Return a stable cache key component for a SWE-rebench Docker image."""
    name_tag = base_image.split("/")[-1]
    name = name_tag.split(":")[0]
    return name


def apptainer_agent_image_path(
    custom_tag: str,
    target: TargetType = DEFAULT_BUILD_TARGET,
) -> Path:
    """Return the local Apptainer SIF path for a SWE-rebench agent image."""
    _, git_sha, _ = _get_sdk_submodule_info()
    sdk_short_sha = git_sha[:7] if git_sha != "unknown" else "unknown"
    content_hash = dockerfile_content_hash()
    name = _sanitize_filename(f"{sdk_short_sha}-{content_hash}-{custom_tag}-{target}")
    return _build_root() / f"{name}.sif"


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _package_install_script() -> str:
    """Return package setup shell matching the minimal Docker target."""
    return r"""
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then
    apt-get -o Acquire::Retries=5 update
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        bash ca-certificates curl wget sudo apt-utils git jq tmux tar \
        build-essential coreutils util-linux procps findutils grep sed \
        apt-transport-https gnupg lsb-release xz-utils
    rm -rf /var/lib/apt/lists/*
elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache \
        bash ca-certificates curl wget sudo git jq tmux tar build-base \
        coreutils util-linux procps findutils grep sed gnupg shadow xz
elif command -v microdnf >/dev/null 2>&1; then
    microdnf install -y \
        bash ca-certificates curl wget sudo git jq tmux tar make gcc gcc-c++ \
        coreutils util-linux procps-ng findutils grep sed shadow-utils \
        gnupg2 xz
    microdnf clean all
elif command -v dnf >/dev/null 2>&1; then
    dnf install -y \
        bash ca-certificates curl wget sudo git jq tmux tar make gcc gcc-c++ \
        coreutils util-linux procps-ng findutils grep sed shadow-utils \
        gnupg2 xz
    dnf clean all
elif command -v yum >/dev/null 2>&1; then
    yum install -y \
        bash ca-certificates curl wget sudo git jq tmux tar make gcc gcc-c++ \
        coreutils util-linux procps-ng findutils grep sed shadow-utils \
        gnupg2 xz
    yum clean all
elif command -v zypper >/dev/null 2>&1; then
    zypper --non-interactive install --no-recommends \
        bash ca-certificates curl wget sudo git jq tmux tar make gcc gcc-c++ \
        coreutils util-linux procps findutils grep sed shadow gpg2 xz
    zypper clean --all
else
    echo "Unsupported base image: no known package manager found" >&2
    exit 1
fi
"""


REMOTE_CONDA_ACTIVATE = (
    "__swebench_restore_nounset=0; "
    "case $- in *u*) __swebench_restore_nounset=1; set +u;; esac; "
    "if [ -f /opt/conda/etc/profile.d/conda.sh ]; then "
    "source /opt/conda/etc/profile.d/conda.sh; "
    "elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then "
    "source /opt/miniconda3/etc/profile.d/conda.sh; "
    "elif [ -f /opt/miniconda3/bin/activate ]; then "
    "source /opt/miniconda3/bin/activate; "
    "else echo 'Conda activation script not found' >&2; exit 1; fi"
)


def _make_remote_eval_compatible(eval_script_list: list[str]) -> list[str]:
    """Mirror SWE-rebench harness remote eval command compatibility handling."""
    return [
        REMOTE_CONDA_ACTIVATE
        if cmd == "source /opt/miniconda3/bin/activate"
        else f'{cmd}; if [ "${{__swebench_restore_nounset:-0}}" = 1 ]; then set -u; fi'
        if cmd.startswith("conda activate ")
        else cmd
        for cmd in eval_script_list
    ]


def _testbed_reinstall_script(install_command: str | None) -> str:
    """Return a %post script that refreshes installs from /testbed."""
    if install_command is None:
        return ""
    commands = _make_remote_eval_compatible(
        [
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "cd /testbed",
            "git config --global --add safe.directory /testbed",
            install_command,
        ]
    )
    command_body = "\n".join(commands)
    return f"""
    if [ -d /testbed ]; then
        if ! bash <<'SWEREBENCH_REINSTALL'
set -euxo pipefail
{command_body}
SWEREBENCH_REINSTALL
        then
            echo "SWE-rebench /testbed reinstall failed; continuing image build" >&2
        fi
    fi
"""


def _definition_file_content(
    base_image: str,
    git_sha: str,
    git_ref: str,
    uv_path: Path,
    uvx_path: Path | None,
    install_command: str | None,
) -> str:
    sdk_root = _sdk_root()
    testbed_reinstall_script = _testbed_reinstall_script(install_command)
    uvx_files = f"    {uvx_path} /usr/local/bin/uvx\n" if uvx_path else ""
    uv_concurrent_downloads = os.getenv(
        "OPENHANDS_APPTAINER_UV_CONCURRENT_DOWNLOADS", "4"
    )
    uv_concurrent_builds = os.getenv("OPENHANDS_APPTAINER_UV_CONCURRENT_BUILDS", "1")
    uv_concurrent_installs = os.getenv(
        "OPENHANDS_APPTAINER_UV_CONCURRENT_INSTALLS", "1"
    )
    return f"""Bootstrap: docker
From: {base_image}

%files
    {uv_path} /usr/local/bin/uv
{uvx_files}\
    {sdk_root / "pyproject.toml"} /agent-server/pyproject.toml
    {sdk_root / "uv.lock"} /agent-server/uv.lock
    {sdk_root / "README.md"} /agent-server/README.md
    {sdk_root / "LICENSE"} /agent-server/LICENSE
    {sdk_root / "openhands-sdk"} /agent-server/openhands-sdk
    {sdk_root / "openhands-tools"} /agent-server/openhands-tools
    {sdk_root / "openhands-workspace"} /agent-server/openhands-workspace
    {sdk_root / "openhands-agent-server"} /agent-server/openhands-agent-server

%post
    set -eux
    {_package_install_script()}
    {testbed_reinstall_script}

    USERNAME=openhands
    UID=10001
    GID=10001
    grep -Eq "^[^:]*:[^:]*:${{GID}}:" /etc/group || groupadd -g "${{GID}}" "${{USERNAME}}"
    grep -Eq "^${{USERNAME}}:" /etc/passwd || useradd -m -u "${{UID}}" -g "${{GID}}" -s /bin/bash "${{USERNAME}}"
    usermod -aG sudo "${{USERNAME}}" 2>/dev/null || true
    echo "${{USERNAME}} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    mkdir -p /workspace/project /agent-server/uv-managed-python
    chown -R "${{USERNAME}}:${{USERNAME}}" /workspace /agent-server

    chmod 0755 /usr/local/bin/uv
    if [ -e /usr/local/bin/uvx ]; then chmod 0755 /usr/local/bin/uvx; fi

    su "${{USERNAME}}" -s /bin/bash -c 'cd /agent-server && \\
        export HOME=/home/openhands && \\
        export UV_CONCURRENT_DOWNLOADS={uv_concurrent_downloads} && \\
        export UV_CONCURRENT_BUILDS={uv_concurrent_builds} && \\
        export UV_CONCURRENT_INSTALLS={uv_concurrent_installs} && \\
        export UV_PROJECT_ENVIRONMENT=/agent-server/.venv && \\
        export UV_PYTHON_INSTALL_DIR=/agent-server/uv-managed-python && \\
        uv python install 3.13 && \\
        uv venv --python-preference only-managed --python 3.13 .venv && \\
        uv sync --frozen --no-editable --managed-python --extra boto3 && \\
        uv pip install --python /agent-server/.venv/bin/python "transformers>=4.56.0,<5" && \\
        readlink -f .venv/bin/python | grep -q "^/agent-server/uv-managed-python/"'

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export OH_ENABLE_VNC=false
    export LOG_JSON=true
    export OPENHANDS_BUILD_GIT_SHA={git_sha}
    export OPENHANDS_BUILD_GIT_REF={git_ref}

%runscript
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export OH_ENABLE_VNC=false
    export LOG_JSON=true
    export OPENHANDS_BUILD_GIT_SHA={git_sha}
    export OPENHANDS_BUILD_GIT_REF={git_ref}
    exec /agent-server/.venv/bin/python -m openhands.agent_server "$@"
"""


def build_apptainer_agent_image(
    base_image: str,
    custom_tag: str,
    target: TargetType = DEFAULT_BUILD_TARGET,
    install_command: str | None = None,
) -> BuildOutput:
    """Build a local Apptainer agent-server SIF from a SWE-rebench base image."""
    if target not in SUPPORTED_APPTAINER_TARGETS:
        return BuildOutput(
            base_image=base_image,
            tags=[],
            error=(
                f"Apptainer local builds currently support "
                f"{sorted(SUPPORTED_APPTAINER_TARGETS)}, got {target!r}"
            ),
        )

    if shutil.which("apptainer") is None:
        return BuildOutput(
            base_image=base_image,
            tags=[],
            error="Apptainer is not available on PATH",
        )
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        return BuildOutput(
            base_image=base_image,
            tags=[],
            error="uv is not available on PATH",
        )
    uvx_bin = shutil.which("uvx")

    image_path = apptainer_agent_image_path(custom_tag, target)
    if image_path.exists() and not _force_build_enabled():
        logger.info("Using existing Apptainer agent SIF %s", image_path)
        return BuildOutput(base_image=base_image, tags=[str(image_path)], error=None)

    build_root = _build_root()
    log_dir = build_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True, exist_ok=True)

    tmp_image = image_path.with_suffix(".tmp.sif")
    _remove_path(tmp_image)

    git_ref, git_sha, _ = _get_sdk_submodule_info()
    definition = build_root / f"{image_path.name}.def"
    definition.write_text(
        _definition_file_content(
            base_image=base_image,
            git_sha=git_sha,
            git_ref=git_ref,
            uv_path=Path(uv_bin).resolve(),
            uvx_path=Path(uvx_bin).resolve() if uvx_bin else None,
            install_command=install_command,
        )
    )

    log_path = log_dir / f"{image_path.name}.log"
    cmd = ["apptainer", "build", str(tmp_image), str(definition)]
    logger.info("Building Apptainer agent SIF: %s", " ".join(cmd))
    env = os.environ.copy()
    if "APPTAINER_CACHEDIR" not in env:
        env["APPTAINER_CACHEDIR"] = str(build_root / "cache")
    for key in ("APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"):
        if env.get(key):
            Path(env[key]).expanduser().mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        _remove_path(tmp_image)
        return BuildOutput(
            base_image=base_image,
            tags=[],
            error=f"Apptainer build failed with exit code {proc.returncode}",
            log_path=str(log_path),
        )

    tmp_image.replace(image_path)
    logger.info("Built Apptainer agent SIF %s", image_path)
    return BuildOutput(
        base_image=base_image,
        tags=[str(image_path)],
        error=None,
        log_path=str(log_path),
    )


def ensure_apptainer_agent_image(
    base_image: str,
    custom_tag: str,
    target: TargetType = DEFAULT_BUILD_TARGET,
    install_command: str | None = None,
) -> Path:
    """Build or reuse a local Apptainer agent-server SIF."""
    output = build_apptainer_agent_image(
        base_image=base_image,
        custom_tag=custom_tag,
        target=target,
        install_command=install_command,
    )
    logger.info("Apptainer image build output: %s", output)
    if output.error is not None:
        raise RuntimeError(f"Apptainer image build failed: {output.error}")
    if not output.tags:
        raise RuntimeError("Apptainer image build produced no image path")
    return Path(output.tags[0])


def collect_unique_base_images(
    dataset: str,
    split: str,
    n_limit: int | None = None,
    selected_instances_file: str | None = None,
) -> list[tuple[str, str]]:
    """Load SWE-rebench rows and return unique base images with install commands."""
    df = get_dataset(
        dataset_name=dataset,
        split=split,
        eval_limit=n_limit if n_limit else None,
        selected_instances_file=selected_instances_file,
    )
    base_images = sorted(
        {
            (
                docker_image_from_row(row.to_dict()),
                install_command_from_row(row.to_dict()),
            )
            for _, row in df.iterrows()
        }
    )
    return base_images


def _build_one_base_image(base_image: tuple[str, str]) -> BuildOutput:
    docker_image, install_command = base_image
    return build_apptainer_agent_image(
        base_image=docker_image,
        custom_tag=extract_custom_tag(docker_image),
        install_command=install_command,
    )


def build_apptainer_agent_images(
    base_images: list[tuple[str, str]],
    *,
    max_workers: int,
) -> list[BuildOutput]:
    """Build SWE-rebench Apptainer agent images in parallel."""
    workers = max(1, max_workers)
    if workers == 1:
        return [
            _build_one_base_image(base_image)
            for base_image in tqdm(
                base_images,
                desc="Building Apptainer SIFs",
                disable=False,
            )
        ]

    results: list[BuildOutput] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _build_one_base_image,
                base_image,
            ): base_image
            for base_image in base_images
        }
        completed = as_completed(futures)
        progress = tqdm(
            completed,
            total=len(futures),
            desc="Building Apptainer SIFs",
            disable=False,
        )
        for future in progress:
            base_image = futures[future]
            docker_image, _ = base_image
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    BuildOutput(base_image=docker_image, tags=[], error=str(exc))
                )
    return results


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build OpenHands Apptainer agent-server SIFs for SWE-rebench "
            "docker_image bases."
        )
    )
    parser.add_argument(
        "--dataset",
        default="nebius/SWE-rebench",
        help="Dataset name or local JSON/JSONL file.",
    )
    parser.add_argument(
        "--split",
        default="filtered",
        help="Dataset split to read.",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="Limit rows before deduplicating docker_image values (0 = no limit).",
    )
    parser.add_argument(
        "--select",
        default=None,
        help="Path to text file containing instance IDs to select.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel Apptainer builds.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSONL file for per-image BuildOutput records.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)

    base_images = collect_unique_base_images(
        dataset=args.dataset,
        split=args.split,
        n_limit=args.n_limit,
        selected_instances_file=args.select,
    )
    logger.info("Building %d unique SWE-rebench base images", len(base_images))

    results = build_apptainer_agent_images(
        base_images=base_images,
        max_workers=args.max_workers,
    )

    if args.manifest:
        manifest = Path(args.manifest).expanduser()
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result.model_dump()) + "\n")

    failures = [result for result in results if result.error is not None]
    for result in results:
        if result.error is not None:
            print(f"FAILED {result.base_image}: {result.error}", file=sys.stderr)
        elif result.tags:
            print(f"BUILT {result.base_image}: {result.tags[0]}")
        else:
            print(
                f"FAILED {result.base_image}: no image path produced", file=sys.stderr
            )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
