import asyncio
import json
import threading
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from benchmarks.swerebench import modal_build_images
from benchmarks.swerebench.apptainer_build import SwerebenchImageSpec
from benchmarks.swerebench.modal_build_images import (
    MAX_IMAGES_PER_VM,
    MAX_MODAL_IMAGE_NAME_LENGTH,
    BuildRequest,
    ParallelNamedImagePublisher,
    SandboxBuildConfig,
    _cleanup_sandbox,
    _cleanup_sandbox_async,
    _print_summary,
    _require_complete_run,
    build_requests,
    collect_run_failures,
    expected_registry_tag,
    group_build_requests_by_repo,
    modal_image_name,
    parse_repos,
    publish_builder_runtime_image,
    publish_registry_images,
    registry_refs_from_build_results,
    registry_refs_from_results,
    run_repo_batched_vm_builds,
    select_image_specs_by_repo,
)


def _request() -> BuildRequest:
    return BuildRequest(
        base_image="swerebench/sweb.eval.x86_64.owner_1776_repo-1",
        custom_tag="swerebench_sweb.eval.x86_64.owner_1776_repo-1-repair-hash",
        install_command="pip install -e .[dev]",
        destination_image="docker.io/adityasoni8/eval-agent-server",
        push=True,
    )


def test_build_requests_preserve_install_command_and_shorten_tag() -> None:
    spec = SwerebenchImageSpec(
        base_image="swerebench/base",
        custom_tag="x" * 100,
        install_command="pip install -e .",
        repo="owner/repo",
        instance_id="owner__repo-1",
    )

    request = build_requests([spec], push=True)[0]

    assert request.install_command == "pip install -e ."
    assert len(request.custom_tag) == 64
    assert request.push is True
    assert request.force_build is True
    assert request.repo == "owner/repo"
    assert request.instance_id == "owner__repo-1"


def test_build_requests_can_skip_existing_registry_images() -> None:
    spec = SwerebenchImageSpec(
        base_image="swerebench/base",
        custom_tag="tag",
        install_command="pip install -e .",
    )

    request = build_requests([spec], push=True, force_build=False)[0]

    assert request.force_build is False


def test_sandbox_cleanup_errors_are_recorded_not_raised() -> None:
    sandbox = MagicMock()
    sandbox.terminate.side_effect = RuntimeError("already timed out")
    sandbox.detach.side_effect = RuntimeError("already detached")

    error = _cleanup_sandbox(sandbox)

    assert error is not None
    assert "terminate: RuntimeError: already timed out" in error
    assert "detach: RuntimeError: already detached" in error


def test_async_sandbox_cleanup_errors_are_recorded_not_raised() -> None:
    sandbox = MagicMock()
    sandbox.terminate.aio = AsyncMock(side_effect=RuntimeError("already timed out"))
    sandbox.detach.aio = AsyncMock(side_effect=RuntimeError("already detached"))

    error = asyncio.run(_cleanup_sandbox_async(sandbox))

    assert error is not None
    assert "terminate: RuntimeError: already timed out" in error
    assert "detach: RuntimeError: already detached" in error


def test_build_logs_are_quiet_by_default() -> None:
    assert SandboxBuildConfig().stream_build_logs is False


def test_build_requests_reject_missing_install_command() -> None:
    spec = SwerebenchImageSpec("swerebench/base", "base", None)

    try:
        build_requests([spec])
    except ValueError as exc:
        assert "no install command" in str(exc)
    else:
        raise AssertionError("Expected missing install command to be rejected")


def test_expected_registry_tag() -> None:
    assert expected_registry_tag(_request()) == (
        "docker.io/adityasoni8/eval-agent-server:5acdf05-"
        "swerebench_sweb.eval.x86_64.owner_1776_repo-1-repair-hash-source-minimal"
    )


def test_modal_image_name_uses_swesmith_rules_and_length_limits() -> None:
    image_name = modal_image_name(expected_registry_tag(_request()))
    name, tag = image_name.rsplit(":", 1)

    assert name == "docker.io__adityasoni8__eval-agent-server"
    assert "/" not in image_name
    assert len(name) <= MAX_MODAL_IMAGE_NAME_LENGTH
    assert len(tag) <= MAX_MODAL_IMAGE_NAME_LENGTH


def test_registry_refs_from_results_selects_successful_pushes(
    tmp_path: Path,
) -> None:
    request = asdict_request = _request().__dict__
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            [
                {"success": True, "request": request},
                {"success": False, "request": request},
                {"success": True, "request": {**asdict_request, "push": False}},
            ]
        )
    )

    assert registry_refs_from_results(results_path) == [
        expected_registry_tag(_request())
    ]


def test_registry_refs_from_build_results_selects_successful_pushes() -> None:
    request = _request().__dict__

    assert registry_refs_from_build_results(
        [
            {"success": True, "request": request},
            {"success": False, "request": request},
            {"success": True, "request": {**request, "push": False}},
        ]
    ) == [expected_registry_tag(_request())]


def test_parallel_named_image_publisher_deduplicates_and_runs_concurrently(
    monkeypatch,
) -> None:
    barrier = threading.Barrier(2)
    calls: list[str] = []

    def fake_publish(image_ref: str, **kwargs) -> dict[str, object]:
        calls.append(image_ref)
        barrier.wait(timeout=2)
        return {
            "duration_seconds": 0.01,
            "image_ref": image_ref,
            "modal_image": f"named:{image_ref}",
            "success": True,
        }

    monkeypatch.setattr(
        "benchmarks.swerebench.modal_build_images._publish_registry_image_result",
        fake_publish,
    )

    results = publish_registry_images(
        ["registry/image:a", "registry/image:b", "registry/image:a"],
        max_workers=2,
        show_progress=False,
    )

    assert sorted(calls) == ["registry/image:a", "registry/image:b"]
    assert len(results) == 2
    assert all(result["success"] for result in results)


def test_failed_publish_does_not_prevent_later_publish(monkeypatch) -> None:
    calls: list[str] = []

    def fake_publish(image_ref: str, **kwargs) -> dict[str, object]:
        calls.append(image_ref)
        return {
            "duration_seconds": 0.01,
            "image_ref": image_ref,
            "success": image_ref.endswith(":good"),
            "error": None if image_ref.endswith(":good") else "publish failed",
        }

    monkeypatch.setattr(
        "benchmarks.swerebench.modal_build_images._publish_registry_image_result",
        fake_publish,
    )

    results = publish_registry_images(
        ["registry/image:bad", "registry/image:good"],
        max_workers=1,
        show_progress=False,
    )

    assert calls == ["registry/image:bad", "registry/image:good"]
    assert [result["success"] for result in results] == [False, True]


def test_named_image_publisher_accepts_successful_build_result(monkeypatch) -> None:
    published: list[str] = []

    def fake_publish(image_ref: str, **kwargs) -> dict[str, object]:
        published.append(image_ref)
        return {
            "duration_seconds": 0.01,
            "image_ref": image_ref,
            "modal_image": "named:image",
            "success": True,
        }

    monkeypatch.setattr(
        "benchmarks.swerebench.modal_build_images._publish_registry_image_result",
        fake_publish,
    )
    publisher = ParallelNamedImagePublisher(max_workers=1, show_progress=False)
    publisher.submit_build_result({"success": True, "request": _request().__dict__})

    results = publisher.finish()

    assert published == [expected_registry_tag(_request())]
    assert results[0]["success"] is True


def test_named_image_import_forces_registry_refresh(monkeypatch) -> None:
    source_image = MagicMock()
    prepared_image = source_image.entrypoint.return_value
    built_image = MagicMock()
    build_future: Future[object] = Future()
    build_future.set_result(built_image)
    publish_future: Future[None] = Future()
    publish_future.set_result(None)
    prepared_image.build.return_value = build_future
    built_image.publish.return_value = publish_future
    from_registry = MagicMock(return_value=source_image)
    app_handle = object()
    monkeypatch.setattr(modal_build_images.modal.Image, "from_registry", from_registry)
    monkeypatch.setattr(
        modal_build_images.modal.App,
        "lookup",
        MagicMock(return_value=app_handle),
    )

    image_ref = expected_registry_tag(_request())
    name = modal_build_images.publish_registry_image(
        image_ref, stream_output=False, announce=False
    )

    from_registry.assert_called_once_with(
        image_ref,
        secret=modal_build_images.registry_secret,
        force_build=True,
    )
    prepared_image.build.assert_called_once_with(app_handle, _future=True)
    built_image.publish.assert_called_once_with(name, _future=True)


def test_named_image_import_timeout_cancels_stalled_modal_future(monkeypatch) -> None:
    source_image = MagicMock()
    prepared_image = source_image.entrypoint.return_value
    stalled_future: Future[object] = Future()
    prepared_image.build.return_value = stalled_future
    monkeypatch.setattr(
        modal_build_images.modal.Image,
        "from_registry",
        MagicMock(return_value=source_image),
    )
    monkeypatch.setattr(
        modal_build_images.modal.App,
        "lookup",
        MagicMock(return_value=object()),
    )

    try:
        modal_build_images.publish_registry_image(
            expected_registry_tag(_request()),
            stream_output=False,
            announce=False,
            timeout_seconds=0,
        )
    except TimeoutError as exc:
        assert "while importing" in str(exc)
    else:
        raise AssertionError("Expected stalled Modal import to time out")

    assert stalled_future.cancelled()


def test_builder_runtime_is_published_before_use(monkeypatch) -> None:
    definition = MagicMock()
    built_image = definition.build.return_value
    app_handle = modal_build_images.app
    monkeypatch.setattr(modal_build_images, "builder_image_definition", definition)

    name = publish_builder_runtime_image(
        app_handle=app_handle,
        announce=False,
    )

    definition.build.assert_called_once_with(app_handle)
    built_image.publish.assert_called_once_with(
        modal_build_images.BUILDER_MODAL_IMAGE_NAME
    )
    assert name == modal_build_images.BUILDER_MODAL_IMAGE_NAME


def test_repo_batches_use_native_async_concurrency(monkeypatch, tmp_path) -> None:
    requests = [
        BuildRequest(
            base_image=f"base-{index}",
            custom_tag=f"tag-{index}",
            install_command="pip install -e .",
            destination_image="registry/image",
            repo="owner/repo",
            instance_id=f"owner__repo-{index}",
        )
        for index in range(3)
    ]
    active = 0
    peak_active = 0

    async def fake_build(batch, config, on_result):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return [
            {
                "success": True,
                "request": request.__dict__,
            }
            for request in batch
        ]

    monkeypatch.setattr(
        modal_build_images,
        "_build_repo_batch_in_vm_sandbox_async",
        fake_build,
    )

    results = run_repo_batched_vm_builds(
        requests,
        max_workers=2,
        max_images_per_vm=1,
        config=SandboxBuildConfig(),
        show_progress=False,
        output_dir=tmp_path,
    )

    assert len(results) == 3
    assert peak_active == 2


def test_complete_run_rejects_missing_publish() -> None:
    build_result = {
        "success": True,
        "request": {**_request().__dict__, "push": True},
    }

    try:
        _require_complete_run([build_result], [])
    except RuntimeError as exc:
        assert "missing_publishes=1" in str(exc)
    else:
        raise AssertionError("Expected missing named-image publish to fail the run")


def test_failure_report_correlates_build_and_publish_failures() -> None:
    request = _request().__dict__
    image_ref = expected_registry_tag(_request())

    failures = collect_run_failures(
        [
            {
                "success": False,
                "request": request,
                "error": "build failed",
                "sandbox_id": "sb-1",
            }
        ],
        [
            {
                "success": False,
                "image_ref": image_ref,
                "error": "publish failed",
                "instance_id": request["instance_id"],
                "repo": request["repo"],
                "base_image": request["base_image"],
                "sandbox_id": "sb-1",
            }
        ],
    )

    assert [failure["stage"] for failure in failures] == ["build", "publish"]
    assert all(failure["instance_id"] == request["instance_id"] for failure in failures)


def test_parse_repos_deduplicates_and_preserves_order() -> None:
    assert parse_repos("owner/b, owner/a,owner/b,,") == ["owner/b", "owner/a"]


def test_select_and_group_requests_builds_full_capped_vm_batches() -> None:
    specs = [
        SwerebenchImageSpec(
            base_image=f"swerebench/{repo.replace('/', '_')}-{index}",
            custom_tag=f"{repo.replace('/', '_')}-{index}",
            install_command="pip install -e .",
            repo=repo,
            instance_id=f"{repo.replace('/', '__')}-{index}",
        )
        for repo in ("owner/repo-a", "owner/repo-b", "owner/repo-c")
        for index in range(10)
    ]

    selected = select_image_specs_by_repo(
        specs,
        repos=["owner/repo-a", "owner/repo-b", "owner/repo-c"],
        images_per_repo=8,
    )
    batches = group_build_requests_by_repo(
        build_requests(selected, push=True), max_images_per_vm=8
    )

    assert len(selected) == 24
    assert len(batches) == 3
    assert [len(batch) for batch in batches] == [8, 8, 8]
    assert all(len({request.repo for request in batch}) == 1 for batch in batches)


def test_repo_batching_fills_underfull_vms_from_following_repos() -> None:
    requests = [
        BuildRequest(
            base_image=f"image-{repo}-{index}",
            custom_tag=f"tag-{repo}-{index}",
            install_command="pip install -e .",
            destination_image="registry/image",
            push=True,
            repo=f"owner/{repo}",
            instance_id=f"{repo}-{index}",
        )
        for repo, count in (("a", 3), ("b", 2), ("c", 5))
        for index in range(count)
    ]

    batches = group_build_requests_by_repo(requests, max_images_per_vm=8)

    assert [len(batch) for batch in batches] == [8, 2]
    assert {request.repo for request in batches[0]} == {
        "owner/a",
        "owner/b",
        "owner/c",
    }
    assert {request.repo for request in batches[1]} == {"owner/c"}


def test_repo_batch_limits_reject_more_than_safety_cap() -> None:
    try:
        group_build_requests_by_repo([], max_images_per_vm=MAX_IMAGES_PER_VM + 1)
    except ValueError as exc:
        assert "safety cap" in str(exc)
    else:
        raise AssertionError("Expected oversized VM batches to be rejected")


def test_repo_batch_summary_counts_shared_sandboxes_once(capsys) -> None:
    results = [
        {
            "duration_seconds": 10.0,
            "success": True,
            "sandbox_id": "sandbox-a",
            "batch_duration_seconds": 25.0,
        },
        {
            "duration_seconds": 12.0,
            "success": True,
            "sandbox_id": "sandbox-a",
            "batch_duration_seconds": 25.0,
        },
        {
            "duration_seconds": 8.0,
            "success": False,
            "sandbox_id": "sandbox-b",
            "batch_duration_seconds": 15.0,
        },
    ]

    _print_summary(results, wall_clock_seconds=26.0)
    summary = json.loads(capsys.readouterr().out)

    assert summary["successful"] == 2
    assert summary["failed"] == 1
    assert summary["vm_count"] == 2
    assert summary["wall_clock_seconds"] == 26.0
    assert summary["cumulative_image_build_seconds"] == 30.0
    assert summary["cumulative_vm_seconds"] == 40.0
