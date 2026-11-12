import json
import sys
from pathlib import Path

import pytest

from scripts import pull_apptainer_images as pull


IMAGE = "docker.io/example/server@sha256:" + "a" * 64


@pytest.mark.parametrize("instance_id", ["../escape", "/absolute", "", "a/b"])
def test_manifest_rejects_unsafe_filenames(tmp_path: Path, instance_id: str) -> None:
    manifest = tmp_path / "images.jsonl"
    manifest.write_text(json.dumps({"instance_id": instance_id, "image": IMAGE}) + "\n")
    with pytest.raises(ValueError, match="Unsafe instance_id"):
        pull.read_manifest(manifest)


def test_manifest_requires_pinned_images_and_unique_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "images.jsonl"
    row = {"instance_id": "owner__repo-1", "image": "docker.io/example/server:latest"}
    manifest.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="digest-pinned"):
        pull.read_manifest(manifest)
    row["image"] = IMAGE
    manifest.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        pull.read_manifest(manifest)


def test_retry_and_resume_require_success_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".completed").mkdir()
    image = pull.Image("owner__repo-1", IMAGE)
    attempts: list[list[str]] = []

    def run(command: list[str], log: Path, timeout: int) -> int:
        attempts.append(command)
        Path(command[-2]).write_bytes(b"sif-data")
        return 1 if len(attempts) == 1 else 0

    monkeypatch.setattr(pull, "run_command", run)
    monkeypatch.setattr(pull.STOP, "wait", lambda seconds: None)
    result = pull.convert(image, tmp_path, tmp_path, "-processors 2", 1, 10)
    assert result["status"] == "ok"
    assert "--force" not in attempts[0]
    assert "--force" in attempts[1]
    result = pull.convert(image, tmp_path, tmp_path, "-processors 2", 1, 10)
    assert result["status"] == "skipped"
    assert len(attempts) == 2
    # A truncated output must be rebuilt even when a success record exists.
    (tmp_path / "owner__repo-1.sif").write_bytes(b"bad")
    assert (
        pull.convert(image, tmp_path, tmp_path, "-processors 2", 0, 10)["status"]
        == "ok"
    )
    assert len(attempts) == 3


def test_untracked_sif_is_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / ".completed").mkdir()
    (tmp_path / "owner__repo-1.sif").write_bytes(b"existing")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        pull.convert(pull.Image("owner__repo-1", IMAGE), tmp_path, tmp_path, "", 0, 10)


def test_cancelled_work_does_not_start_a_build(tmp_path: Path) -> None:
    pull.STOP.set()
    try:
        result = pull.convert(
            pull.Image("owner__repo-1", IMAGE), tmp_path, tmp_path, "", 0, 10
        )
        assert result["status"] == "interrupted"
        assert list(tmp_path.iterdir()) == []
    finally:
        pull.STOP.clear()


def test_one_worker_exception_does_not_abort_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "images.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"instance_id": name, "image": IMAGE}) + "\n"
            for name in ("broken", "healthy")
        )
    )
    report = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pull_apptainer_images.py",
            "--manifest",
            str(manifest),
            "--sif-dir",
            str(tmp_path / "sif"),
            "--report-dir",
            str(report),
            "--workers",
            "2",
        ],
    )
    monkeypatch.setenv("APPTAINER_CACHEDIR", str(tmp_path / "cache"))
    monkeypatch.setenv("APPTAINER_TMPDIR", str(tmp_path / "scratch"))
    monkeypatch.setattr(pull.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(pull.os, "umask", lambda _mode: 0)

    def convert(
        image: pull.Image,
        sif_dir: Path,
        report_dir: Path,
        flags: str,
        retries: int,
        timeout: int,
    ) -> dict[str, object]:
        if image.instance_id == "broken":
            raise RuntimeError("simulated per-image failure")
        return {"instance_id": image.instance_id, "status": "ok"}

    monkeypatch.setattr(pull, "convert", convert)
    with pytest.raises(SystemExit) as exit_info:
        pull.main()
    assert exit_info.value.code == 1
    summary = json.loads((report / "summary.json").read_text())
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    results = [
        json.loads(line) for line in (report / "results.jsonl").read_text().splitlines()
    ]
    assert {row["instance_id"]: row["status"] for row in results} == {
        "broken": "failed",
        "healthy": "ok",
    }


def test_runner_restart_resumes_completed_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".completed").mkdir()
    builds: list[list[str]] = []
    statuses: list[object] = []

    def run(command: list[str], log: Path, timeout: int) -> int:
        builds.append(command)
        Path(command[-2]).write_bytes(b"sif-data")
        return 0

    def main() -> None:
        result = pull.convert(
            pull.Image("owner__repo-1", IMAGE), tmp_path, tmp_path, "", 0, 10
        )
        statuses.append(result["status"])
        if len(statuses) == 1:
            raise OSError("simulated reporting failure after image completion")

    monkeypatch.setattr(pull, "run_command", run)
    monkeypatch.setattr(pull, "main", main)
    pull.run_with_restarts(delay=0)
    assert statuses == ["ok", "skipped"]
    assert len(builds) == 1


def test_runner_stops_after_100_restarts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def main() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("persistent failure")

    monkeypatch.setattr(pull, "main", main)
    with pytest.raises(SystemExit) as exit_info:
        pull.run_with_restarts(delay=0)
    assert exit_info.value.code == 1
    assert calls == 101  # Initial attempt plus 100 retries.


@pytest.mark.parametrize("code", [0, 1, 2, 130])
def test_runner_does_not_restart_normal_exits(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    calls = 0

    def main() -> None:
        nonlocal calls
        calls += 1
        raise SystemExit(code)

    monkeypatch.setattr(pull, "main", main)
    with pytest.raises(SystemExit) as exit_info:
        pull.run_with_restarts(delay=0)
    assert exit_info.value.code == code
    assert calls == 1


def test_stop_during_restart_delay_prevents_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def main() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(pull, "main", main)
    monkeypatch.setattr(pull.STOP, "wait", lambda _seconds: pull.STOP.set())
    try:
        with pytest.raises(SystemExit) as exit_info:
            pull.run_with_restarts()
        assert exit_info.value.code == 130
        assert calls == 1
    finally:
        pull.STOP.clear()


def test_interrupted_record_update_preserves_previous_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "image.json"
    previous: dict[str, object] = {"image": IMAGE, "status": "running"}
    pull.write_record(record, previous)

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("simulated interrupted record update")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError):
        pull.write_record(record, {"image": IMAGE, "status": "ok"})
    assert json.loads(record.read_text()) == previous


def test_builds_have_private_indexes_and_share_persistent_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APPTAINER_CACHEDIR", str(tmp_path))
    with pull.build_environment() as first, pull.build_environment() as second:
        first_layout = Path(first["APPTAINER_CACHEDIR"]) / "cache/blob"
        second_layout = Path(second["APPTAINER_CACHEDIR"]) / "cache/blob"
        assert first_layout != second_layout
        (first_layout / "index.json").write_text('{"private": true}')
        assert not (second_layout / "index.json").exists()
        (first_layout / "blobs/digest").write_bytes(b"cached-layer")
        assert (second_layout / "blobs/digest").read_bytes() == b"cached-layer"
        assert pull.os.environ["APPTAINER_CACHEDIR"] == str(tmp_path)
    assert not first_layout.exists()
    assert not second_layout.exists()
    assert (tmp_path / "cache/blob/blobs/digest").read_bytes() == b"cached-layer"
