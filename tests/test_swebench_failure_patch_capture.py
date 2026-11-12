from types import SimpleNamespace
from typing import cast

from benchmarks.swebench.run_infer import SWEBenchEvaluation
from benchmarks.utils.critics import PassCritic
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.models import EvalInstance, EvalMetadata
from openhands.sdk import LLM
from openhands.sdk.workspace import RemoteWorkspace


class DummyEvaluation(Evaluation):
    def prepare_instances(self):
        return []

    def prepare_workspace(
        self,
        instance,
        resource_factor=1,
        forward_env=None,
    ):
        raise NotImplementedError

    def evaluate_instance(self, instance, workspace):
        raise NotImplementedError


class FakeWorkspace:
    def __init__(self):
        self.commands = []

    def execute_command(self, command, *args, **kwargs):
        self.commands.append(command)
        if "git --no-pager diff --no-color --cached " in command:
            return SimpleNamespace(
                exit_code=0,
                stdout="diff --git a/file.py b/file.py\n",
                stderr="",
            )
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


def _metadata(tmp_path):
    return EvalMetadata(
        llm=LLM(model="test-model", api_key="test-key"),
        dataset="test-dataset",
        max_iterations=1,
        eval_output_dir=str(tmp_path),
        critic=PassCritic(),
    )


def test_error_output_preserves_failure_test_result(tmp_path):
    evaluation = DummyEvaluation(metadata=_metadata(tmp_path))
    instance = EvalInstance(id="instance-1", data={})

    output = evaluation._create_error_output(
        instance,
        RuntimeError("boom"),
        retry_count=0,
        test_result={"git_patch": "diff --git a/file.py b/file.py\n"},
    )

    assert output.error is not None
    assert output.test_result["git_patch"].startswith("diff --git")


def test_swebench_collect_failure_test_result_gets_git_patch(tmp_path):
    evaluation = SWEBenchEvaluation(metadata=_metadata(tmp_path))
    instance = EvalInstance(
        id="django__django-12345",
        data={
            "repo": "django/django",
            "base_commit": "abc123",
        },
    )
    workspace = FakeWorkspace()

    result = evaluation.collect_failure_test_result(
        instance,
        cast(RemoteWorkspace, workspace),
        RuntimeError("Remote conversation got stuck"),
    )

    assert result == {
        "git_patch": "diff --git a/file.py b/file.py\n",
        "git_patch_captured_on_error": True,
    }
    assert workspace.commands == [
        "cd /workspace/django/ ; git add -A",
        "cd /workspace/django/ ; git --no-pager diff --no-color --cached HEAD",
    ]


def test_swebench_apptainer_collects_failure_patch_from_testbed(tmp_path):
    metadata = _metadata(tmp_path).model_copy(update={"workspace_type": "apptainer"})
    evaluation = SWEBenchEvaluation(metadata=metadata)
    instance = EvalInstance(
        id="django__django-12345",
        data={"repo": "django/django", "base_commit": "abc123"},
    )
    workspace = FakeWorkspace()

    evaluation.collect_failure_test_result(
        instance,
        cast(RemoteWorkspace, workspace),
        RuntimeError("Remote conversation got stuck"),
    )

    # Apptainer agents edit the image's repository in place.
    assert workspace.commands == [
        "cd /testbed ; git add -A",
        "cd /testbed ; git --no-pager diff --no-color --cached HEAD",
    ]


def test_swebench_patches_diff_against_recorded_start_commit(tmp_path):
    # The SWE-bench images add a setup commit on top of base_commit; patches are
    # taken against the commit the agent started from, which evaluate_instance records.
    metadata = _metadata(tmp_path).model_copy(update={"workspace_type": "apptainer"})
    evaluation = SWEBenchEvaluation(metadata=metadata)
    start = "d350420dae50c80ca33b845734c31428d62af0a8"
    instance = EvalInstance(
        id="astropy__astropy-12907",
        data={
            "repo": "astropy/astropy",
            "base_commit": "abc123",
            "start_commit": start,
        },
    )
    workspace = FakeWorkspace()

    evaluation.get_git_patch(instance, cast(RemoteWorkspace, workspace))
    evaluation.get_staged_git_patch(instance, cast(RemoteWorkspace, workspace))

    assert workspace.commands == [
        f"cd /testbed ; git --no-pager diff --no-color {start} HEAD",
        f"cd /testbed ; git --no-pager diff --no-color --cached {start}",
    ]
