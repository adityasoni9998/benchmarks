import base64
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from benchmarks.swerebench import docker_build
from benchmarks.swerebench.apptainer_build import SwerebenchImageSpec
from benchmarks.utils import build_utils
from benchmarks.utils.build_utils import BuildOutput


def _image_spec() -> SwerebenchImageSpec:
    return SwerebenchImageSpec(
        base_image="swerebench/sweb.eval.x86_64.owner_1776_repo-1",
        custom_tag="swerebench_repo-reinstall-v1-hash",
        install_command="pip install -e .[dev]",
    )


def test_docker_custom_tag_is_stable_and_within_sdk_tag_budget() -> None:
    spec = SwerebenchImageSpec(
        base_image="swerebench/base",
        custom_tag="x" * 100,
        install_command="pip install -e .",
    )

    custom_tag = docker_build.docker_custom_tag(spec)

    assert len(custom_tag) == docker_build.MAX_DOCKER_CUSTOM_TAG_LENGTH
    assert custom_tag == docker_build.docker_custom_tag(spec)
    assert custom_tag.startswith("x" * 51 + "-")


def test_primary_agent_image_tag_uses_sdk_sha(monkeypatch) -> None:
    monkeypatch.setattr(
        docker_build,
        "_get_sdk_submodule_info",
        lambda: ("modal_workspace", "abcdef123456", "1.0"),
    )

    assert docker_build.primary_agent_image_tag(
        "local/swerebench-agent", _image_spec()
    ) == (
        "local/swerebench-agent:abcdef1-swerebench_repo-reinstall-v1-hash-"
        "source-minimal"
    )


def test_build_adds_reinstall_layer(monkeypatch) -> None:
    spec = _image_spec()
    monkeypatch.setattr(docker_build, "local_image_exists", lambda tag: False)
    monkeypatch.setattr(
        docker_build,
        "_get_sdk_submodule_info",
        lambda: ("modal_workspace", "abcdef123456", "1.0"),
    )
    monkeypatch.setattr(
        docker_build,
        "build_image",
        lambda **kwargs: BuildOutput(
            base_image=kwargs["base_image"],
            tags=[
                "local/agent-swerebench-unrepaired:abcdef1-"
                "swerebench_repo-reinstall-v1-hash-source-minimal",
                "local/agent-swerebench-unrepaired:abcdef123456-"
                "swerebench_repo-reinstall-v1-hash-source-minimal",
                "local/agent-swerebench-unrepaired:abcdef1-generic-base-tag-"
                "source-minimal",
            ],
        ),
    )
    captured = {}

    def fake_layer(**kwargs):
        captured.update(kwargs)
        return BuildOutput(base_image="wrapper", tags=kwargs["tags"])

    monkeypatch.setattr(docker_build, "run_docker_build_layer", fake_layer)

    output = docker_build.build_docker_agent_image(
        spec,
        target_image="local/agent",
    )

    assert output.base_image == spec.base_image
    assert output.tags == [
        "local/agent:abcdef1-swerebench_repo-reinstall-v1-hash-source-minimal",
        "local/agent:abcdef123456-swerebench_repo-reinstall-v1-hash-source-minimal",
    ]
    encoded = captured["build_args"]["SWEREBENCH_INSTALL_COMMAND_B64"]
    assert base64.b64decode(encoded).decode() == "pip install -e .[dev]"
    assert captured["push"] is False
    assert captured["load"] is True
    assert captured["builder"] == "default"


def test_build_reuses_generic_remote_intermediate_for_canonical_final_tag(
    monkeypatch,
) -> None:
    spec = _image_spec()
    monkeypatch.setattr(docker_build, "remote_image_exists", lambda tag: False)
    monkeypatch.setattr(
        docker_build,
        "_get_sdk_submodule_info",
        lambda: ("modal_workspace", "abcdef123456", "1.0"),
    )
    monkeypatch.setattr(
        docker_build,
        "build_image",
        lambda **kwargs: BuildOutput(
            base_image=kwargs["base_image"],
            tags=[
                "local/agent-swerebench-unrepaired:abcdef1-generic-base-tag-"
                "source-minimal"
            ],
            status="skipped_remote_exists",
            skip_reason="remote_image_exists",
        ),
    )
    captured = {}

    def fake_layer(**kwargs):
        captured.update(kwargs)
        return BuildOutput(base_image="wrapper", tags=kwargs["tags"])

    monkeypatch.setattr(docker_build, "run_docker_build_layer", fake_layer)

    output = docker_build.build_docker_agent_image(
        spec,
        target_image="local/agent",
        push=True,
        force_build=False,
    )

    assert output.tags == [
        "local/agent:abcdef1-swerebench_repo-reinstall-v1-hash-source-minimal"
    ]
    assert captured["build_args"]["SDK_IMAGE"].endswith(
        ":abcdef1-generic-base-tag-source-minimal"
    )


def test_reinstall_script_activates_testbed_and_runs_editable_install(
    tmp_path: Path,
) -> None:
    conda_root = tmp_path / "opt" / "miniconda3"
    testbed = tmp_path / "testbed"
    bin_dir = tmp_path / "bin"
    (conda_root / "etc" / "profile.d").mkdir(parents=True)
    testbed.mkdir()
    bin_dir.mkdir()
    (conda_root / "etc" / "profile.d" / "conda.sh").write_text(
        'conda() { printf "%s\\n" "$*" >> "$SWEREBENCH_TEST_LOG"; }\n'
    )
    (bin_dir / "git").write_text(
        '#!/bin/sh\nprintf "git %s\\n" "$*" >> "$SWEREBENCH_TEST_LOG"\n'
    )
    (bin_dir / "pip").write_text(
        '#!/bin/sh\nprintf "pip %s at %s\\n" "$*" "$PWD" >> "$SWEREBENCH_TEST_LOG"\n'
    )
    for executable in (bin_dir / "git", bin_dir / "pip"):
        executable.chmod(0o755)

    script = docker_build.DOCKERFILE.with_name("reinstall_testbed.sh").read_text()
    script = script.replace("/opt/miniconda3", str(conda_root)).replace(
        "/testbed", str(testbed)
    )
    test_script = tmp_path / "reinstall.sh"
    test_script.write_text(script)
    test_script.chmod(0o755)
    log_path = tmp_path / "commands.log"
    command = base64.b64encode(b"pip install -e .[dev]").decode()

    subprocess.run(
        [str(test_script), command],
        check=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SWEREBENCH_TEST_LOG": str(log_path),
        },
    )

    assert log_path.read_text().splitlines() == [
        "activate testbed",
        f"git config --global --add safe.directory {testbed}",
        f"pip install -e .[dev] at {testbed}",
    ]


def test_dockerfile_treats_reinstall_as_best_effort() -> None:
    dockerfile = docker_build.DOCKERFILE.read_text()

    assert "if ! bash /tmp/swerebench-reinstall-testbed.sh" in dockerfile
    assert "continuing image build" in dockerfile


def test_wrapper_push_uses_estargz(monkeypatch, tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n")
    monkeypatch.setenv("OPENHANDS_IMAGE_COMPRESSION", "estargz")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(build_utils.subprocess, "run", fake_run)

    output = build_utils.run_docker_build_layer(
        dockerfile=dockerfile,
        context=tmp_path,
        tags=["registry.example/agent:test"],
        push=True,
    )

    assert output.error is None
    output_arg = captured["cmd"][captured["cmd"].index("--output") + 1]
    assert output_arg == (
        "type=registry,compression=estargz,force-compression=true,oci-mediatypes=true"
    )
    assert "--push" not in captured["cmd"]
