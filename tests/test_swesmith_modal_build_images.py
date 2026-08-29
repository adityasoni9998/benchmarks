import json
from pathlib import Path

from benchmarks.swesmith.modal_build_images import (
    MAX_MODAL_IMAGE_NAME_LENGTH,
    VM_BASE_IMAGES,
    BuildRequest,
    custom_tag_for_base_image,
    expected_registry_tag,
    load_base_images,
    modal_image_name,
    publish_registry_images,
    registry_refs_from_results,
)


def test_vm_build_sample_has_eight_images() -> None:
    assert len(VM_BASE_IMAGES) == 8
    assert len(set(VM_BASE_IMAGES)) == 8


def test_image_tag_helpers() -> None:
    base_image = "docker.io/swebench/swesmith.x86_64.owner_1776_repo.abcdef01"
    request = BuildRequest(
        base_image=base_image,
        custom_tag=custom_tag_for_base_image(base_image),
        destination_image="docker.io/adityasoni8/eval-agent-server",
        push=True,
    )

    assert request.custom_tag == "swesmith.x86_64.owner_1776_repo.abcdef01"
    assert expected_registry_tag(request) == (
        "docker.io/adityasoni8/eval-agent-server:5acdf05-"
        "swesmith.x86_64.owner_1776_repo.abcdef01-source-minimal"
    )


def test_modal_image_name_normalizes_and_shortens_tag() -> None:
    image_ref = (
        "docker.io/adityasoni8/eval-agent-server:5acdf05-"
        "swesmith.x86_64.luozhouyang_1776_python-string-similarity.115acaac-"
        "source-minimal"
    )

    image_name = modal_image_name(image_ref)
    name, tag = image_name.rsplit(":", 1)

    assert name == "docker.io__adityasoni8__eval-agent-server"
    assert "/" not in image_name
    assert "x86_64" not in tag
    assert len(name) <= MAX_MODAL_IMAGE_NAME_LENGTH
    assert len(tag) <= MAX_MODAL_IMAGE_NAME_LENGTH


def test_load_base_images_from_dataset_json(tmp_path: Path) -> None:
    manifest = tmp_path / "images.json"
    manifest.write_text(
        json.dumps(
            [
                {"image_name": "swebench/swesmith.x86_64.owner.repo"},
                {"image_name": "docker.io/swebench/SWESMITH.x86_64.owner.repo"},
                "another/image:tag",
            ]
        )
    )

    assert load_base_images(manifest) == [
        "docker.io/swebench/swesmith.x86_64.owner.repo",
        "docker.io/another/image:tag",
    ]


def test_registry_refs_from_results_selects_successful_pushes(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.json"
    request = {
        "base_image": "docker.io/swebench/swesmith.x86_64.owner.repo",
        "custom_tag": "swesmith.x86_64.owner.repo",
        "destination_image": "docker.io/adityasoni8/eval-agent-server",
        "push": True,
    }
    results_path.write_text(
        json.dumps(
            [
                {"success": True, "request": request},
                {"success": True, "request": {**request, "push": False}},
                {"success": False, "request": request},
            ]
        )
    )

    assert registry_refs_from_results(results_path) == [
        "docker.io/adityasoni8/eval-agent-server:5acdf05-"
        "swesmith.x86_64.owner.repo-source-minimal"
    ]


def test_publish_registry_images_accepts_empty_input() -> None:
    assert publish_registry_images([], max_workers=8) == []
