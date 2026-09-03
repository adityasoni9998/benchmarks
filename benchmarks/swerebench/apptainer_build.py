"""Local Apptainer builds for SWE-rebench agent-server images.

SWE-rebench rows carry the exact base image in their ``docker_image`` field.
Unlike SWE-bench, there are no per-repository dependency wrappers here; the
Apptainer definition installs the OpenHands agent server directly into that
base image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from tqdm.auto import tqdm

from benchmarks.utils.build_utils import BuildOutput, _get_sdk_submodule_info
from openhands.sdk import get_logger


logger = get_logger(__name__)

TargetType = Literal["source-minimal"]

BUILD_TARGET_SOURCE_MINIMAL: TargetType = "source-minimal"
DEFAULT_BUILD_TARGET: TargetType = BUILD_TARGET_SOURCE_MINIMAL

DEFAULT_APPTAINER_BUILD_ROOT = (
    Path.home() / ".cache" / "openhands" / "swerebench-apptainer-agent-images"
)
DEFAULT_APPTAINER_CACHEDIR = Path("/data/user_data/adityabs/apptainer_cache")
DEFAULT_APPTAINER_TMPDIR = Path("/data/user_data/adityabs/apptainer_tmp")
SWEREBENCH_REPAIR_VERSION = "swerebench-testbed-reinstall-v2"
SWEREBENCH_DATASET_COLUMNS = [
    "instance_id",
    "repo",
    "docker_image",
    "install_config",
]


@dataclass(frozen=True)
class SwerebenchImageSpec:
    base_image: str
    custom_tag: str
    install_command: str | None
    repo: str | None = None
    instance_id: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sdk_root() -> Path:
    return _repo_root() / "vendor" / "software-agent-sdk"


def _sdk_dockerfile() -> Path:
    dockerfile = (
        _sdk_root()
        / "openhands-agent-server"
        / "openhands"
        / "agent_server"
        / "docker"
        / "Dockerfile"
    )
    if not dockerfile.exists():
        raise FileNotFoundError(
            f"SDK Dockerfile not found at {dockerfile}. "
            "Make sure submodules are initialized."
        )
    return dockerfile


def dockerfile_content_hash() -> str:
    """Return a short content hash for the SDK agent-server Dockerfile."""
    content = _sdk_dockerfile().read_text()
    return hashlib.sha256(content.encode()).hexdigest()[:7]


def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _build_root() -> Path:
    return Path(
        os.getenv("OPENHANDS_APPTAINER_BUILD_ROOT", str(DEFAULT_APPTAINER_BUILD_ROOT))
    ).expanduser()


def _resolve_dir(path: str | Path | None, default: Path) -> Path:
    return Path(path).expanduser() if path is not None else default


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


def install_command_from_row(row: Mapping[str, object]) -> str | None:
    """Return the SWE-rebench dataset install command for a row, if present."""
    install_config = row.get("install_config")
    if isinstance(install_config, str):
        install_config = json.loads(install_config)
    if not isinstance(install_config, Mapping):
        return None
    install_command = install_config.get("install")
    if isinstance(install_command, str) and install_command.strip():
        return install_command
    return None


def instance_id_from_row(row: Mapping[str, object]) -> str:
    """Return a SWE-rebench instance id from a dataset row."""
    instance_id = row.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("SWE-rebench row is missing a non-empty instance_id field")
    return instance_id


def _load_selected_instances(select_file_path: str) -> set[str]:
    selected_instances: set[str] = set()
    with open(select_file_path, encoding="utf-8") as f:
        for line in f:
            instance_id = line.strip()
            if instance_id:
                selected_instances.add(instance_id)
    if not selected_instances:
        raise ValueError(f"Select file is empty: {select_file_path}")
    return selected_instances


def _read_local_dataset_file(dataset: str) -> pd.DataFrame:
    path = Path(dataset).expanduser()
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=SWEREBENCH_DATASET_COLUMNS)
    raise ValueError(f"Unsupported local dataset file type: {path}")


def _split_matches_parquet_path(path: str, split: str) -> bool:
    file_name = Path(path).name
    return (
        file_name == f"{split}.parquet"
        or file_name.startswith(f"{split}-")
        or f"/{split}/" in f"/{path}/"
    )


def _read_hf_parquet_columns(dataset: str, split: str) -> pd.DataFrame:
    """Read only columns needed for builds, avoiding datasets feature inference."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    parquet_files = [
        path
        for path in api.list_repo_files(dataset, repo_type="dataset")
        if path.endswith(".parquet") and _split_matches_parquet_path(path, split)
    ]
    if not parquet_files:
        raise ValueError(f"No parquet files found for split {split!r} in {dataset}")

    frames: list[pd.DataFrame] = []
    for parquet_file in sorted(parquet_files):
        local_file = hf_hub_download(
            repo_id=dataset,
            filename=parquet_file,
            repo_type="dataset",
        )
        frames.append(pd.read_parquet(local_file, columns=SWEREBENCH_DATASET_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _load_swerebench_rows(
    dataset: str,
    split: str,
    n_limit: int | None,
    selected_instances_file: str | None,
) -> pd.DataFrame:
    if Path(dataset).expanduser().is_file():
        df = _read_local_dataset_file(dataset)
    else:
        df = _read_hf_parquet_columns(dataset, split)

    if selected_instances_file is not None:
        selected_instances = _load_selected_instances(selected_instances_file)
        df = cast(
            pd.DataFrame,
            df[df["instance_id"].isin(sorted(selected_instances))],
        )

    if n_limit is not None and n_limit > 0:
        df = df.head(n_limit)

    return df


def extract_custom_tag(base_image: str) -> str:
    """Return a stable cache key component for a SWE-rebench Docker image."""
    return _sanitize_filename(base_image)


def _repair_config_hash(install_command: str | None) -> str:
    content = json.dumps(
        {
            "version": SWEREBENCH_REPAIR_VERSION,
            "install": install_command or "",
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:7]


def image_spec_from_row(row: Mapping[str, object]) -> SwerebenchImageSpec:
    base_image = docker_image_from_row(row)
    install_command = install_command_from_row(row)
    custom_tag = (
        f"{extract_custom_tag(base_image)}-{SWEREBENCH_REPAIR_VERSION}-"
        f"{_repair_config_hash(install_command)}"
    )
    return SwerebenchImageSpec(
        base_image=base_image,
        custom_tag=custom_tag,
        install_command=install_command,
        repo=str(row["repo"]) if isinstance(row.get("repo"), str) else None,
        instance_id=instance_id_from_row(row),
    )


def apptainer_agent_image_path(
    custom_tag: str,
    target: str = DEFAULT_BUILD_TARGET,
    sif_dir: str | Path | None = None,
) -> Path:
    """Return the local Apptainer SIF path for a SWE-rebench agent image."""
    _, git_sha, _ = _get_sdk_submodule_info()
    sdk_short_sha = git_sha[:7] if git_sha != "unknown" else "unknown"
    content_hash = dockerfile_content_hash()
    name = _sanitize_filename(f"{sdk_short_sha}-{content_hash}-{custom_tag}-{target}")
    return _resolve_dir(sif_dir, _build_root()) / f"{name}.sif"


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
    """Return a %post script that refreshes editable installs from /testbed."""
    if install_command is None:
        return ""
    commands = _make_remote_eval_compatible(
        [
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "cd /testbed",
            "git config --global --add safe.directory /testbed",
            "cd /testbed",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            install_command,
        ]
    )
    command_body = "\n".join(commands)
    return f"""
    if [ -d /testbed ]; then
        bash <<'SWEREBENCH_REINSTALL'
set -uxo pipefail
{command_body}
SWEREBENCH_REINSTALL
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
    grep -Eq "^[^:]*:[^:]*:${{GID}}:" /etc/group || \\
        groupadd -g "${{GID}}" "${{USERNAME}}"
    grep -Eq "^${{USERNAME}}:" /etc/passwd || \\
        useradd -m -u "${{UID}}" -g "${{GID}}" -s /bin/bash "${{USERNAME}}"
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
        uv pip install --python /agent-server/.venv/bin/python \\
            "transformers>=4.56.0,<5" && \\
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
    custom_tag: str | None = None,
    target: str = DEFAULT_BUILD_TARGET,
    *,
    install_command: str | None = None,
    sif_dir: str | Path | None = None,
    apptainer_cache_dir: str | Path | None = None,
    apptainer_tmp_dir: str | Path | None = None,
    definition_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    force_build: bool | None = None,
) -> BuildOutput:
    """Build a local Apptainer agent-server SIF from a SWE-rebench base image."""
    if target != BUILD_TARGET_SOURCE_MINIMAL:
        return BuildOutput(
            base_image=base_image,
            tags=[],
            error=(
                "SWE-rebench Apptainer local builds only support "
                f"{BUILD_TARGET_SOURCE_MINIMAL!r}, got {target!r}"
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

    image_path = apptainer_agent_image_path(
        custom_tag or extract_custom_tag(base_image),
        target,
        sif_dir=sif_dir,
    )
    if force_build is None:
        force_build = _force_build_enabled()
    if image_path.exists() and not force_build:
        logger.info("Using existing Apptainer agent SIF %s", image_path)
        return BuildOutput(base_image=base_image, tags=[str(image_path)], error=None)

    build_root = image_path.parent
    build_root.mkdir(parents=True, exist_ok=True)
    resolved_log_dir = _resolve_dir(log_dir, build_root / "logs")
    resolved_definition_dir = _resolve_dir(definition_dir, build_root / "definitions")
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    resolved_definition_dir.mkdir(parents=True, exist_ok=True)

    tmp_image = image_path.with_suffix(".tmp.sif")
    _remove_path(tmp_image)

    git_ref, git_sha, _ = _get_sdk_submodule_info()
    definition = resolved_definition_dir / f"{image_path.name}.def"
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

    log_path = resolved_log_dir / f"{image_path.name}.log"
    cmd = ["apptainer", "build", str(tmp_image), str(definition)]
    logger.info("Building Apptainer agent SIF: %s", " ".join(cmd))
    env = os.environ.copy()
    if apptainer_cache_dir is not None:
        env["APPTAINER_CACHEDIR"] = str(Path(apptainer_cache_dir).expanduser())
    elif "APPTAINER_CACHEDIR" not in env:
        env["APPTAINER_CACHEDIR"] = str(DEFAULT_APPTAINER_CACHEDIR)
    if apptainer_tmp_dir is not None:
        env["APPTAINER_TMPDIR"] = str(Path(apptainer_tmp_dir).expanduser())
    elif "APPTAINER_TMPDIR" not in env:
        env["APPTAINER_TMPDIR"] = str(DEFAULT_APPTAINER_TMPDIR)
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
    custom_tag: str | None = None,
    target: str = DEFAULT_BUILD_TARGET,
    *,
    install_command: str | None = None,
    sif_dir: str | Path | None = None,
    apptainer_cache_dir: str | Path | None = None,
    apptainer_tmp_dir: str | Path | None = None,
    definition_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    force_build: bool | None = None,
) -> Path:
    """Build or reuse a local Apptainer agent-server SIF."""
    output = build_apptainer_agent_image(
        base_image=base_image,
        custom_tag=custom_tag,
        target=target,
        install_command=install_command,
        sif_dir=sif_dir,
        apptainer_cache_dir=apptainer_cache_dir,
        apptainer_tmp_dir=apptainer_tmp_dir,
        definition_dir=definition_dir,
        log_dir=log_dir,
        force_build=force_build,
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
    image_limit: int | None = None,
) -> list[str]:
    """Load SWE-rebench rows and return unique base images from docker_image."""
    df = _load_swerebench_rows(
        dataset=dataset,
        split=split,
        n_limit=n_limit if n_limit else None,
        selected_instances_file=selected_instances_file,
    )
    base_images = sorted(
        {docker_image_from_row(row.to_dict()) for _, row in df.iterrows()}
    )
    if image_limit is not None and image_limit > 0:
        return base_images[:image_limit]
    return base_images


def collect_unique_image_specs(
    dataset: str,
    split: str,
    n_limit: int | None = None,
    selected_instances_file: str | None = None,
    image_limit: int | None = None,
) -> list[SwerebenchImageSpec]:
    """Load SWE-rebench rows and return unique image specs from docker_image."""
    df = _load_swerebench_rows(
        dataset=dataset,
        split=split,
        n_limit=n_limit if n_limit else None,
        selected_instances_file=selected_instances_file,
    )

    specs_by_image: dict[str, SwerebenchImageSpec] = {}
    for _, row in df.iterrows():
        spec = image_spec_from_row(row.to_dict())
        existing = specs_by_image.get(spec.base_image)
        if existing is not None and (
            existing.custom_tag != spec.custom_tag
            or existing.install_command != spec.install_command
        ):
            raise ValueError(
                "Conflicting SWE-rebench install configs for "
                f"{spec.base_image!r}: {existing.install_command!r} vs "
                f"{spec.install_command!r}"
            )
        specs_by_image[spec.base_image] = spec

    image_specs = [specs_by_image[key] for key in sorted(specs_by_image)]
    if image_limit is not None and image_limit > 0:
        return image_specs[:image_limit]
    return image_specs


def _build_one_image_spec(
    image_spec: SwerebenchImageSpec,
    *,
    sif_dir: str,
    apptainer_cache_dir: str,
    apptainer_tmp_dir: str,
    definition_dir: str | None,
    log_dir: str | None,
    force_build: bool,
) -> BuildOutput:
    return build_apptainer_agent_image(
        base_image=image_spec.base_image,
        custom_tag=image_spec.custom_tag,
        install_command=image_spec.install_command,
        sif_dir=sif_dir,
        apptainer_cache_dir=apptainer_cache_dir,
        apptainer_tmp_dir=apptainer_tmp_dir,
        definition_dir=definition_dir,
        log_dir=log_dir,
        force_build=force_build,
    )


def build_apptainer_agent_images(
    image_specs: list[SwerebenchImageSpec],
    *,
    max_workers: int,
    sif_dir: str | Path,
    apptainer_cache_dir: str | Path,
    apptainer_tmp_dir: str | Path,
    definition_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    force_build: bool = False,
    show_progress: bool = True,
) -> list[BuildOutput]:
    """Build SWE-rebench Apptainer agent images in parallel."""
    resolved_sif_dir = str(Path(sif_dir).expanduser())
    resolved_cache_dir = str(Path(apptainer_cache_dir).expanduser())
    resolved_tmp_dir = str(Path(apptainer_tmp_dir).expanduser())
    resolved_definition_dir = (
        str(Path(definition_dir).expanduser()) if definition_dir is not None else None
    )
    resolved_log_dir = str(Path(log_dir).expanduser()) if log_dir is not None else None

    for directory in (resolved_sif_dir, resolved_cache_dir, resolved_tmp_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)
    if resolved_definition_dir is not None:
        Path(resolved_definition_dir).mkdir(parents=True, exist_ok=True)
    if resolved_log_dir is not None:
        Path(resolved_log_dir).mkdir(parents=True, exist_ok=True)

    workers = max(1, max_workers)
    if workers == 1:
        return [
            _build_one_image_spec(
                image_spec,
                sif_dir=resolved_sif_dir,
                apptainer_cache_dir=resolved_cache_dir,
                apptainer_tmp_dir=resolved_tmp_dir,
                definition_dir=resolved_definition_dir,
                log_dir=resolved_log_dir,
                force_build=force_build,
            )
            for image_spec in tqdm(
                image_specs,
                desc="Building Apptainer SIFs",
                disable=not show_progress,
            )
        ]

    results: list[BuildOutput] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _build_one_image_spec,
                image_spec,
                sif_dir=resolved_sif_dir,
                apptainer_cache_dir=resolved_cache_dir,
                apptainer_tmp_dir=resolved_tmp_dir,
                definition_dir=resolved_definition_dir,
                log_dir=resolved_log_dir,
                force_build=force_build,
            ): image_spec
            for image_spec in image_specs
        }
        completed = as_completed(futures)
        progress = tqdm(
            completed,
            total=len(futures),
            desc="Building Apptainer SIFs",
            disable=not show_progress,
        )
        for future in progress:
            image_spec = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    BuildOutput(
                        base_image=image_spec.base_image,
                        tags=[],
                        error=str(exc),
                    )
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
        "--image-limit",
        type=int,
        default=0,
        help="Limit unique docker_image specs after deduplication (0 = no limit).",
    )
    parser.add_argument(
        "--select",
        default=None,
        help="Path to text file containing instance IDs to select.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel Apptainer builds.",
    )
    parser.add_argument(
        "--sif-dir",
        required=True,
        help="Directory where final .sif files are written.",
    )
    parser.add_argument(
        "--apptainer-cache-dir",
        required=True,
        help="Directory exported as APPTAINER_CACHEDIR for each build.",
    )
    parser.add_argument(
        "--apptainer-tmp-dir",
        required=True,
        help="Directory exported as APPTAINER_TMPDIR for each build.",
    )
    parser.add_argument(
        "--definition-dir",
        default=None,
        help="Directory for generated Apptainer definition files.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for build logs.",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Rebuild even when the target .sif already exists.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSONL file for per-image BuildOutput records.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)

    image_specs = collect_unique_image_specs(
        dataset=args.dataset,
        split=args.split,
        n_limit=args.n_limit,
        selected_instances_file=args.select,
        image_limit=args.image_limit,
    )
    logger.info("Building %d unique SWE-rebench base images", len(image_specs))

    results = build_apptainer_agent_images(
        image_specs=image_specs,
        max_workers=args.max_workers,
        sif_dir=args.sif_dir,
        apptainer_cache_dir=args.apptainer_cache_dir,
        apptainer_tmp_dir=args.apptainer_tmp_dir,
        definition_dir=args.definition_dir,
        log_dir=args.log_dir,
        force_build=args.force_build,
        show_progress=not args.no_progress,
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
                f"FAILED {result.base_image}: no image path produced",
                file=sys.stderr,
            )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
