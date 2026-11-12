"""The fork's Modal patch grades on agent-server images by default, or official images."""

from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.utils import modal_patches


SPEC = SimpleNamespace(instance_id="astropy__astropy-12907")


def test_official_images(monkeypatch):
    monkeypatch.setenv(modal_patches.MODAL_IMAGE_SOURCE_ENV, "official")
    monkeypatch.delenv("IMAGE_TAG_PREFIX", raising=False)

    assert (
        modal_patches._build_prebuilt_image_tag(SPEC)
        == "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    )


def test_agent_server_images_by_default(monkeypatch):
    monkeypatch.delenv(modal_patches.MODAL_IMAGE_SOURCE_ENV, raising=False)
    monkeypatch.setattr(modal_patches, "_get_image_tag_prefix", lambda: "abc1234")
    monkeypatch.setattr(
        modal_patches, "_get_agent_server_image_repo", lambda: "ghcr.io/x/agent"
    )
    monkeypatch.setenv("SWEBENCH_IMAGE_TARGET", "source-minimal")

    assert modal_patches._build_prebuilt_image_tag(SPEC) == (
        "ghcr.io/x/agent:abc1234-sweb.eval.x86_64.astropy_1776_astropy-12907"
        "-source-minimal"
    )


def test_unknown_image_source_is_rejected(monkeypatch):
    monkeypatch.setenv(modal_patches.MODAL_IMAGE_SOURCE_ENV, "dockerhub")
    with pytest.raises(RuntimeError, match="must be 'agent-server' or 'official'"):
        modal_patches._build_prebuilt_image_tag(SPEC)


def _patched_runtime(monkeypatch):
    from swebench.harness.modal_eval import run_evaluation_modal as mod

    runtime_cls = mod.ModalSandboxRuntime

    def legacy_write_file(self, file_path, content):
        raise AssertionError("legacy Sandbox.open API used")

    monkeypatch.setattr(runtime_cls, "write_file", legacy_write_file)
    modal_patches._patch_modal_sandbox_filesystem_api()
    return runtime_cls


class _FakeFilesystem:
    def __init__(self, fail_paths=()):
        self.written: dict[str, str] = {}
        self.fail_paths = set(fail_paths)

    def write_text(self, data, remote_path):
        if remote_path in self.fail_paths:
            raise OSError("Read-only file system")
        self.written[remote_path] = data


def test_write_file_uses_sandbox_filesystem_api(monkeypatch):
    runtime_cls = _patched_runtime(monkeypatch)
    runtime: Any = SimpleNamespace(
        sandbox=SimpleNamespace(filesystem=_FakeFilesystem())
    )

    runtime_cls.write_file(runtime, "/tmp/patch.diff", "diff --git a b\n")

    assert runtime.sandbox.filesystem.written == {"/tmp/patch.diff": "diff --git a b\n"}


def test_cgroup_write_failure_is_ignored_but_others_raise(monkeypatch):
    runtime_cls = _patched_runtime(monkeypatch)
    fs = _FakeFilesystem(fail_paths={modal_patches.CGROUP_CPU_SHARES, "/eval.sh"})
    runtime: Any = SimpleNamespace(sandbox=SimpleNamespace(filesystem=fs))

    runtime_cls.write_file(runtime, modal_patches.CGROUP_CPU_SHARES, "2048")
    with pytest.raises(OSError):
        runtime_cls.write_file(runtime, "/eval.sh", "echo")
