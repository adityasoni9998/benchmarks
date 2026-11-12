"""grade_run: merged, cleaned predictions and a whole-dataset summary."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from benchmarks.swebench import grade_run
from benchmarks.utils.constants import MODEL_NAME_OR_PATH


SOURCE = (
    "diff --git a/pkg/mod.py b/pkg/mod.py\n"
    "--- a/pkg/mod.py\n"
    "+++ b/pkg/mod.py\n"
    "@@ -1 +1 @@\n"
    "-y = 2\n"
    "+y = 3\n"
)
SETUP = (
    "diff --git a/setup.py b/setup.py\n"
    "--- a/setup.py\n"
    "+++ b/setup.py\n"
    "@@ -1 +1 @@\n"
    "-a\n"
    "+b\n"
)
BINARY = (
    "diff --git a/x.pickle b/x.pickle\n"
    "new file mode 100644\n"
    "index 0000000000..7576f63c01\n"
    "Binary files /dev/null and b/x.pickle differ\n"
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _row(instance_id: str, patch: str, error=None, category=None) -> dict:
    test_result = {"git_patch": patch}
    if category:
        test_result["conversation_error_category"] = category
    return {"instance_id": instance_id, "test_result": test_result, "error": error}


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    out = run / "shard_0" / "ds" / "openai" / "model"
    _write(
        out / "output.jsonl",
        [
            _row("a__a-1", SOURCE + SETUP + BINARY, category="turn_limit"),
            _row("b__b-2", ""),
        ],
    )
    _write(out / "output_errors.jsonl", [_row("c__c-3", SOURCE, error="failed")])
    out1 = run / "shard_1" / "ds" / "openai" / "model"
    # A stale error row for an instance that later finished: output.jsonl wins.
    _write(out1 / "output_errors.jsonl", [_row("a__a-1", "", error="old")])
    return run


def test_load_predictions_merges_shards_and_error_rows(tmp_path):
    preds = grade_run.load_predictions(_run_dir(tmp_path))

    assert set(preds) == {"a__a-1", "b__b-2", "c__c-3"}
    assert preds["a__a-1"]["source"] == "output.jsonl"
    assert preds["a__a-1"]["conversation_error_category"] == "turn_limit"
    assert preds["c__c-3"]["source"] == "output_errors.jsonl"


def test_clean_patch_removes_setup_and_binary_blocks_only():
    assert grade_run.clean_patch(SOURCE + SETUP + BINARY) == SOURCE
    assert grade_run.clean_patch(SOURCE) == SOURCE


def test_main_modal_writes_predictions_and_whole_dataset_summary(tmp_path):
    run = _run_dir(tmp_path)
    out = tmp_path / "graded"
    dataset = [
        {"instance_id": i, "patch": "gold"}
        for i in ("a__a-1", "b__b-2", "c__c-3", "d__d-4")
    ]

    def fake_harness(predictions_file, run_id, **kwargs):
        assert kwargs["modal"] is True
        logs = out / "logs" / "run_evaluation" / run_id / MODEL_NAME_OR_PATH
        for instance_id, resolved in (("a__a-1", True), ("c__c-3", False)):
            report = {
                instance_id: {"patch_successfully_applied": True, "resolved": resolved}
            }
            (logs / instance_id).mkdir(parents=True)
            (logs / instance_id / "report.json").write_text(json.dumps(report))

    argv = ["grade_run", "--run-dir", str(run), "--out", str(out), "--backend", "modal"]
    with (
        patch.object(sys, "argv", argv),
        patch("datasets.load_dataset", return_value=dataset),
        patch(
            "benchmarks.swebench.eval_infer.run_swebench_evaluation",
            side_effect=fake_harness,
        ),
    ):
        grade_run.main()

    predictions = {
        json.loads(line)["instance_id"]: json.loads(line)["model_patch"]
        for line in (out / "predictions.jsonl").read_text().splitlines()
    }
    assert predictions == {
        "a__a-1": SOURCE,
        "b__b-2": "",
        "c__c-3": SOURCE,
        "d__d-4": "",
    }
    summary = json.loads((out / "summary.json").read_text())
    assert summary["total"] == 4
    assert summary["resolved"] == 1
    assert summary["empty_patch"] == 2
    assert summary["by_prediction_source"] == {
        "output.jsonl": 2,
        "output_errors.jsonl": 1,
        "missing": 1,
    }
    rows = [
        json.loads(line) for line in (out / "instances.jsonl").read_text().splitlines()
    ]
    assert [r["resolved"] for r in rows] == [True, False, False, False]
