"""Tests for SWE-bench Apptainer image build fallback."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks.swebench import (
    apptainer_build,
    build_images,
    run_infer as swebench_run_infer,
)
from benchmarks.utils.models import EvalInstance


class FakeApptainerWorkspace:
    """Capture DirectoryOverlayWorkspace constructor arguments and commands."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.commands: list[str] = []

    def execute_command(self, command: str, timeout: float | None = None):
        self.commands.append(command)
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


def _scratch(monkeypatch, tmp_path) -> Path:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv(swebench_run_infer.APPTAINER_SCRATCH_DIR_ENV, str(scratch))
    monkeypatch.setenv("APPTAINER_CACHEDIR", str(tmp_path / "cache"))
    monkeypatch.delenv(swebench_run_infer.APPTAINER_SIF_DIR_ENV, raising=False)
    return scratch


def _evaluation():
    metadata = SimpleNamespace(
        workspace_type="apptainer",
        agent_type="default",
        env_setup_commands=[],
        llm=SimpleNamespace(custom_tokenizer=None),
    )
    evaluation = object.__new__(swebench_run_infer.SWEBenchEvaluation)
    object.__setattr__(evaluation, "metadata", metadata)
    object.__setattr__(evaluation, "current_attempt", 1)
    return evaluation


def test_unsupported_apptainer_build_target_returns_error():
    output = apptainer_build.build_apptainer_agent_image(
        base_image="docker.io/swebench/example:latest",
        custom_tag="example",
        target="binary",
    )

    assert output.tags == []
    assert output.error is not None
    assert "source-minimal" in output.error


def test_apptainer_definition_installs_transformers_for_token_counting():
    definition = apptainer_build._definition_file_content(
        base_image="docker.io/swebench/example:latest",
        git_sha="abc123",
        git_ref="main",
        wrap_swebench_deps=False,
        uv_path=Path("/usr/local/bin/uv"),
        uvx_path=None,
    )

    assert 'uv pip install --python /agent-server/.venv/bin/python "transformers' in (
        definition
    )


def test_failed_forced_apptainer_rebuild_keeps_existing_sif(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENHANDS_APPTAINER_BUILD_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENHANDS_APPTAINER_FORCE_BUILD", "1")
    monkeypatch.setattr(
        apptainer_build,
        "_get_sdk_submodule_info",
        lambda: ("main", "abcdef123456", ""),
    )
    monkeypatch.setattr(apptainer_build, "dockerfile_content_hash", lambda: "hash")
    monkeypatch.setattr(
        apptainer_build.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"apptainer", "uv", "uvx"} else None,
    )
    monkeypatch.setattr(
        apptainer_build.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=42, stdout="build failed"),
    )

    existing_image = apptainer_build.apptainer_agent_image_path("tag")
    existing_image.parent.mkdir(parents=True, exist_ok=True)
    existing_image.write_text("old working image")

    output = apptainer_build.build_apptainer_agent_image(
        base_image="docker.io/swebench/example:latest",
        custom_tag="tag",
    )

    assert output.error == "Apptainer build failed with exit code 42"
    assert existing_image.read_text() == "old working image"


def test_apptainer_workspace_uses_registry_image_when_available(monkeypatch, tmp_path):
    _scratch(monkeypatch, tmp_path)
    monkeypatch.setattr(swebench_run_infer, "remote_image_exists", lambda image: True)
    monkeypatch.setattr(
        swebench_run_infer,
        "DirectoryOverlayWorkspace",
        FakeApptainerWorkspace,
    )

    workspace = _evaluation().prepare_workspace(
        EvalInstance(id="django__django-12345", data={})
    )

    assert isinstance(workspace, FakeApptainerWorkspace)
    assert workspace.kwargs["server_image"] is not None
    assert workspace.kwargs["sif_file"] is None
    assert workspace.kwargs["extra_bind_mounts"] == []


def test_apptainer_workspace_binds_existing_custom_tokenizer(monkeypatch, tmp_path):
    _scratch(monkeypatch, tmp_path)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    evaluation = _evaluation()
    evaluation.metadata.llm.custom_tokenizer = str(tokenizer_dir)

    monkeypatch.setattr(swebench_run_infer, "remote_image_exists", lambda image: True)
    monkeypatch.setattr(
        swebench_run_infer,
        "DirectoryOverlayWorkspace",
        FakeApptainerWorkspace,
    )

    workspace = evaluation.prepare_workspace(
        EvalInstance(id="django__django-12345", data={})
    )

    assert isinstance(workspace, FakeApptainerWorkspace)
    assert workspace.kwargs["extra_bind_mounts"] == [
        f"{tokenizer_dir}:{tokenizer_dir}:ro"
    ]


def test_apptainer_workspace_builds_local_sif_when_registry_image_missing(
    monkeypatch, tmp_path
):
    _scratch(monkeypatch, tmp_path)
    built = {}

    def fake_build(**kwargs):
        built.update(kwargs)
        return Path("/tmp/local-agent.sif")

    monkeypatch.setattr(swebench_run_infer, "remote_image_exists", lambda image: False)
    monkeypatch.setattr(
        swebench_run_infer,
        "ensure_apptainer_agent_image",
        fake_build,
    )
    monkeypatch.setattr(
        swebench_run_infer,
        "DirectoryOverlayWorkspace",
        FakeApptainerWorkspace,
    )

    workspace = _evaluation().prepare_workspace(
        EvalInstance(id="django__django-12345", data={})
    )

    assert isinstance(workspace, FakeApptainerWorkspace)
    assert workspace.kwargs["sif_file"] == "/tmp/local-agent.sif"
    assert workspace.kwargs["server_image"] is None
    assert workspace.kwargs["extra_bind_mounts"] == []
    assert built["base_image"].startswith("docker.io/swebench/")
    assert built["custom_tag"] == "sweb.eval.x86_64.django_1776_django-12345"
    assert built["target"] == "source-minimal"


def test_swebench_image_template_overrides_official_image(monkeypatch):
    monkeypatch.setenv(
        "OPENHANDS_SWEBENCH_IMAGE_TEMPLATE",
        "ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}:latest",
    )

    assert (
        build_images.get_official_docker_image("astropy__astropy-12907")
        == "ghcr.io/epoch-research/"
        "swe-bench.eval.x86_64.astropy__astropy-12907:latest"
    )


def test_apptainer_workspace_matches_hyperion_settings(monkeypatch, tmp_path):
    scratch = _scratch(monkeypatch, tmp_path)
    monkeypatch.setattr(swebench_run_infer, "remote_image_exists", lambda image: True)
    monkeypatch.setattr(
        swebench_run_infer, "DirectoryOverlayWorkspace", FakeApptainerWorkspace
    )

    workspace = _evaluation().prepare_workspace(
        EvalInstance(id="django__django-12345", data={})
    )

    assert isinstance(workspace, FakeApptainerWorkspace)
    assert workspace.kwargs["working_dir"] == "/testbed"
    assert workspace.kwargs["scratch_dir"] == str(scratch)
    assert workspace.kwargs["cache_dir"] == str(tmp_path / "cache")
    assert workspace.kwargs["health_check_timeout"] == 600
    assert workspace.kwargs["detach_logs"] is False
    assert "mount_dir" not in workspace.kwargs
    # The agent's terminal starts with testbed activated.
    assert workspace.commands == [
        f"cat >> ~/.bashrc <<'EOF'\n{swebench_run_infer.ACTIVATE_TESTBED}EOF"
    ]
    assert "conda activate testbed" in swebench_run_infer.ACTIVATE_TESTBED


def test_apptainer_workspace_uses_prebuilt_sif_dir(monkeypatch, tmp_path):
    _scratch(monkeypatch, tmp_path)
    sif_dir = tmp_path / "sif"
    sif_dir.mkdir()
    (sif_dir / "django__django-12345.sif").write_text("sif")
    monkeypatch.setenv(swebench_run_infer.APPTAINER_SIF_DIR_ENV, str(sif_dir))
    monkeypatch.setattr(
        swebench_run_infer,
        "remote_image_exists",
        lambda image: (_ for _ in ()).throw(AssertionError("registry not consulted")),
    )
    monkeypatch.setattr(
        swebench_run_infer, "DirectoryOverlayWorkspace", FakeApptainerWorkspace
    )

    workspace = _evaluation().prepare_workspace(
        EvalInstance(id="django__django-12345", data={})
    )

    assert isinstance(workspace, FakeApptainerWorkspace)
    assert workspace.kwargs["sif_file"] == str(sif_dir / "django__django-12345.sif")
    assert workspace.kwargs["server_image"] is None


def test_apptainer_workspace_missing_prebuilt_sif_fails(monkeypatch, tmp_path):
    import pytest

    _scratch(monkeypatch, tmp_path)
    monkeypatch.setenv(swebench_run_infer.APPTAINER_SIF_DIR_ENV, str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Prebuilt Apptainer SIF not found"):
        _evaluation().prepare_workspace(
            EvalInstance(id="django__django-12345", data={})
        )


def test_apptainer_workspace_requires_scratch_dir(monkeypatch, tmp_path):
    import pytest

    monkeypatch.delenv(swebench_run_infer.APPTAINER_SCRATCH_DIR_ENV, raising=False)
    monkeypatch.delenv(swebench_run_infer.APPTAINER_SIF_DIR_ENV, raising=False)
    monkeypatch.setattr(swebench_run_infer, "remote_image_exists", lambda image: True)

    with pytest.raises(ValueError, match=swebench_run_infer.APPTAINER_SCRATCH_DIR_ENV):
        _evaluation().prepare_workspace(
            EvalInstance(id="django__django-12345", data={})
        )


def test_apptainer_workspace_start_is_retried_like_hyperion(monkeypatch, tmp_path):
    # Attempt 1: the constructor fails (e.g. the agent-server port was taken).
    # Attempt 2: the sandbox starts but setup fails, so it must be cleaned up.
    # Attempt 3: succeeds; the instance never sees the failures.
    _scratch(monkeypatch, tmp_path)
    sif_dir = tmp_path / "sif"
    sif_dir.mkdir()
    (sif_dir / "django__django-12345.sif").write_text("sif")
    monkeypatch.setenv(swebench_run_infer.APPTAINER_SIF_DIR_ENV, str(sif_dir))
    created: list["FlakyWorkspace"] = []

    class FlakyWorkspace(FakeApptainerWorkspace):
        def __init__(self, **kwargs: Any) -> None:
            if not created:
                created.append(self)
                raise RuntimeError("Port 12345 is not available")
            super().__init__(**kwargs)
            created.append(self)
            self.cleaned_up = False

        def execute_command(self, command: str, timeout: float | None = None):
            if len(created) == 2:
                raise RuntimeError("Container process stopped unexpectedly")
            return super().execute_command(command, timeout)

        def cleanup(self) -> None:
            self.cleaned_up = True

    monkeypatch.setattr(swebench_run_infer, "DirectoryOverlayWorkspace", FlakyWorkspace)

    workspace = _evaluation().prepare_workspace(
        EvalInstance(id="django__django-12345", data={})
    )

    assert len(created) == 3
    assert workspace is created[2]
    assert created[1].cleaned_up and not created[2].cleaned_up
    assert any("conda activate testbed" in c for c in created[2].commands)


def test_apptainer_workspace_start_gives_up_after_three_attempts(monkeypatch, tmp_path):
    import pytest

    _scratch(monkeypatch, tmp_path)
    sif_dir = tmp_path / "sif"
    sif_dir.mkdir()
    (sif_dir / "django__django-12345.sif").write_text("sif")
    monkeypatch.setenv(swebench_run_infer.APPTAINER_SIF_DIR_ENV, str(sif_dir))
    attempts: list[int] = []

    def always_fails(**kwargs: Any):
        attempts.append(1)
        raise RuntimeError("Port 12345 is not available")

    monkeypatch.setattr(swebench_run_infer, "DirectoryOverlayWorkspace", always_fails)

    with pytest.raises(RuntimeError, match="Port 12345 is not available"):
        _evaluation().prepare_workspace(
            EvalInstance(id="django__django-12345", data={})
        )
    assert len(attempts) == swebench_run_infer.APPTAINER_START_ATTEMPTS == 3
