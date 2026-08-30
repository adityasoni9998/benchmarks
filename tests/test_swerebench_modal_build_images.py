import json
from pathlib import Path

from benchmarks.swerebench.apptainer_build import SwerebenchImageSpec
from benchmarks.swerebench.modal_build_images import (
    MAX_MODAL_IMAGE_NAME_LENGTH,
    BuildRequest,
    build_requests,
    expected_registry_tag,
    modal_image_name,
    registry_refs_from_results,
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
    )

    request = build_requests([spec], push=True)[0]

    assert request.install_command == "pip install -e ."
    assert len(request.custom_tag) == 64
    assert request.push is True


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
