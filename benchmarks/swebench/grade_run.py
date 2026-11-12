"""Grade the patches of a SWE-bench run, with Apptainer or on Modal.

Predictions come from a run directory of run_infer outputs: every shard's
output.jsonl plus output_errors.jsonl (instances that used up their attempts are
graded on the patch captured from their final tree). Patches are cleaned once for
both backends, as eval_infer does, plus binary hunks: setup files are removed with
remove_files_from_patch and binary-file blocks with remove_binary_diffs; every other
byte is kept. The cleaned predictions are written to <out>/predictions.jsonl, so
both backends grade identical patches.

Backends:
  apptainer  one read-only SIF per instance (<sif-dir>/<instance_id>.sif, the
             official SWE-bench image) with a fresh directory overlay per grading
             run; containers start like the agent sandboxes (shared user namespace,
             --containall --no-eval --no-init --no-umask) plus --cleanenv. Inside:
             apptainer_eval.score_shell (git apply chain + eval.sh), then
             get_eval_report. Reports are kept per instance, so reruns resume.
  modal      the official harness on Modal (eval_infer.run_swebench_evaluation,
             i.e. swebench.harness.run_evaluation --modal true), on the official
             docker.io/swebench images by default (--modal-image-source).

Every dataset instance gets a row in <out>/instances.jsonl; an instance with no
prediction counts as unresolved, and <out>/summary.json reports out of the whole
dataset. --gold / --empty grade the dataset's gold patches / empty patches instead,
to validate a backend.

Examples (from the repo root):
  uv run python -m benchmarks.swebench.grade_run --run-dir RUN --out RUN_grading \\
      --backend apptainer --sif-dir SIFS --scratch /local/$PBS_JOBID/grade
  uv run python -m benchmarks.swebench.grade_run --run-dir RUN --out RUN_modal \\
      --backend modal
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from benchmarks.swebench import constants
from benchmarks.swebench.apptainer_eval import (
    APPTAINER_EXEC_TIMEOUT_BUFFER_SECONDS,
    score_shell,
)
from benchmarks.utils.apptainer_namespace import (
    NAMESPACE_ENV,
    namespace_command,
    shared_apptainer_namespace,
)
from benchmarks.utils.constants import MODEL_NAME_OR_PATH
from benchmarks.utils.patch_utils import remove_binary_diffs, remove_files_from_patch


def clean_patch(patch: str) -> str:
    return remove_binary_diffs(
        remove_files_from_patch(patch, constants.SETUP_FILES_TO_REMOVE)
    )


def load_predictions(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Final record per instance; output.jsonl rows win over output_errors.jsonl."""
    preds: dict[str, dict[str, Any]] = {}
    for kind in ("output_errors.jsonl", "output.jsonl"):
        pattern = str(run_dir / "shard_*" / "**" / kind)
        for path in sorted(glob.glob(pattern, recursive=True)):
            for line in open(path):
                if not line.strip():
                    continue
                row = json.loads(line)
                test_result = row.get("test_result") or {}
                preds[row["instance_id"]] = {
                    "raw_patch": test_result.get("git_patch") or "",
                    "source": kind,
                    "error": row.get("error"),
                    "conversation_error_category": test_result.get(
                        "conversation_error_category"
                    ),
                }
    return preds


def grade_one_apptainer(
    instance: dict[str, Any],
    patch: str,
    out_dir: Path,
    sif_dir: Path,
    scratch: Path,
    apptainer: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    from swebench.harness.constants import APPLY_PATCH_FAIL, TESTS_TIMEOUT
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    instance_id = instance["instance_id"]
    work_dir = out_dir / "instances" / instance_id
    report_path = work_dir / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())[instance_id]
    work_dir.mkdir(parents=True, exist_ok=True)
    if not patch.strip():
        report = {
            "patch_exists": False,
            "patch_successfully_applied": False,
            "resolved": False,
        }
        report_path.write_text(json.dumps({instance_id: report}, indent=2))
        return report

    test_spec = make_test_spec(cast(Any, instance))
    (work_dir / "patch.diff").write_text(patch)
    (work_dir / "eval.sh").write_text(test_spec.eval_script)
    runtime = Path(tempfile.mkdtemp(prefix=f"grade-{instance_id}-", dir=scratch))
    try:
        for name in ("overlay/upper", "overlay/work", "workdir", "tmp"):
            (runtime / name).mkdir(parents=True)
        command = [
            apptainer,
            "exec",
            "--containall",
            "--cleanenv",
            "--no-eval",
            "--no-init",
            "--no-umask",
            "--overlay",
            str(runtime / "overlay"),
            "--workdir",
            str(runtime / "workdir"),
            "--bind",
            f"{work_dir}:/mnt",
            str(sif_dir / f"{instance_id}.sif"),
            "bash",
            "-lc",
            score_shell(timeout_seconds, APPLY_PATCH_FAIL, TESTS_TIMEOUT),
        ]
        namespace = os.environ.get(NAMESPACE_ENV)
        if namespace:
            command = namespace_command(namespace, command)
        tmp = str(runtime / "tmp")
        env = {**os.environ, "APPTAINER_TMPDIR": tmp, "TMPDIR": tmp}
        start = time.time()
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds + APPTAINER_EXEC_TIMEOUT_BUFFER_SECONDS,
            env=env,
        )
        (work_dir / "apptainer_exec.log").write_text(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"apptainer exec exited {proc.returncode}; see "
                f"{work_dir}/apptainer_exec.log"
            )
        prediction = {
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": MODEL_NAME_OR_PATH,
        }
        report = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=str(work_dir / "test_output.txt"),
            include_tests_status=True,
        )[instance_id]
        report["grade_seconds"] = round(time.time() - start, 1)
    finally:
        shutil.rmtree(runtime, ignore_errors=True)
    report_path.write_text(json.dumps({instance_id: report}, indent=2))
    return report


def grade_apptainer(
    instances: list[dict[str, Any]],
    patches: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    apptainer = args.apptainer or shutil.which("apptainer")
    if not apptainer:
        raise SystemExit("apptainer not found; pass --apptainer")
    if not args.sif_dir or not args.scratch:
        raise SystemExit("--backend apptainer needs --sif-dir and --scratch")
    args.scratch.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    with (
        shared_apptainer_namespace(enabled=True),
        ThreadPoolExecutor(args.workers) as pool,
    ):
        futures = {
            pool.submit(
                grade_one_apptainer,
                inst,
                patches[inst["instance_id"]],
                args.out,
                args.sif_dir,
                args.scratch,
                apptainer,
                args.timeout,
            ): inst["instance_id"]
            for inst in instances
        }
        for n, future in enumerate(as_completed(futures), 1):
            instance_id = futures[future]
            try:
                report = future.result()
            except Exception as error:  # infrastructure failure: recorded, unresolved
                report = {"resolved": False, "grading_error": str(error)[:500]}
            with lock:
                results[instance_id] = report
                resolved = sum(bool(r.get("resolved")) for r in results.values())
                print(
                    f"{n}/{len(instances)} resolved so far {resolved} | {instance_id} "
                    f"resolved={report.get('resolved')} "
                    f"applied={report.get('patch_successfully_applied')}",
                    flush=True,
                )
    return results


def grade_modal(
    instances: list[dict[str, Any]],
    predictions_file: Path,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    from benchmarks.swebench.eval_infer import run_swebench_evaluation
    from benchmarks.utils.modal_patches import (
        MODAL_IMAGE_SOURCE_ENV,
        MODAL_SANDBOX_CPU_ENV,
        MODAL_SANDBOX_MEMORY_ENV,
    )

    # Read by the fork's Modal patch, in the harness subprocess and baked into the
    # remote function image: "official" grades on docker.io/swebench images.
    os.environ[MODAL_IMAGE_SOURCE_ENV] = args.modal_image_source
    # Sandbox requests; hyperion's Modal grader uses cpu=0.8, memory=128 MiB.
    os.environ[MODAL_SANDBOX_CPU_ENV] = str(args.modal_sandbox_cpu)
    os.environ[MODAL_SANDBOX_MEMORY_ENV] = str(args.modal_sandbox_memory)
    run_id = args.run_id or args.out.name
    run_swebench_evaluation(
        str(predictions_file),
        run_id,
        dataset=args.dataset,
        workers=args.workers,
        split=args.split,
        modal=True,
        timeout=args.timeout,
    )
    model = MODEL_NAME_OR_PATH
    log_root = args.out / "logs" / "run_evaluation" / run_id / model
    results: dict[str, dict[str, Any]] = {}
    for inst in instances:
        instance_id = inst["instance_id"]
        report_path = log_root / instance_id / "report.json"
        if report_path.exists():
            results[instance_id] = json.loads(report_path.read_text())[instance_id]
        else:
            # Empty patches are skipped by the harness; others failed to grade.
            results[instance_id] = {"patch_exists": False, "resolved": False}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["apptainer", "modal"], required=True)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="eval.sh timeout in seconds (SWE-bench harness default: 1800)",
    )
    parser.add_argument("--sif-dir", type=Path, help="apptainer: <id>.sif per instance")
    parser.add_argument("--scratch", type=Path, help="apptainer: node-local scratch")
    parser.add_argument("--apptainer", help="apptainer binary (default: on PATH)")
    parser.add_argument("--run-id", help="modal: harness run id (default: out name)")
    parser.add_argument(
        "--modal-image-source",
        choices=["official", "agent-server"],
        default="official",
        help="modal: official SWE-bench images (default) or prebuilt agent-server images",
    )
    parser.add_argument(
        "--modal-sandbox-cpu",
        type=float,
        default=0.8,
        help="modal: CPU request per grading sandbox (hyperion: 0.8; harness: 4)",
    )
    parser.add_argument(
        "--modal-sandbox-memory",
        type=int,
        default=128,
        help="modal: memory request (MiB) per grading sandbox (hyperion: 128)",
    )
    parser.add_argument("--gold", action="store_true", help="grade the gold patches")
    parser.add_argument("--empty", action="store_true", help="grade empty patches")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not (args.gold or args.empty or args.run_dir):
        parser.error("--run-dir is required unless --gold or --empty")

    from datasets import load_dataset

    instances = sorted(
        load_dataset(args.dataset, split=args.split), key=lambda x: x["instance_id"]
    )
    if args.limit:
        instances = instances[: args.limit]
    preds = {} if (args.gold or args.empty) else load_predictions(args.run_dir)
    patches: dict[str, str] = {}
    for inst in instances:
        instance_id = inst["instance_id"]
        if args.gold:
            patches[instance_id] = inst["patch"]
        elif args.empty:
            patches[instance_id] = ""
        else:
            pred = preds.get(instance_id)
            patches[instance_id] = clean_patch(pred["raw_patch"]) if pred else ""

    args.out.mkdir(parents=True, exist_ok=True)
    predictions_file = args.out / "predictions.jsonl"
    with predictions_file.open("w") as f:
        for instance_id, patch in patches.items():
            row = {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": MODEL_NAME_OR_PATH,
            }
            f.write(json.dumps(row) + "\n")

    if args.backend == "apptainer":
        results = grade_apptainer(instances, patches, args)
    else:
        results = grade_modal(instances, predictions_file, args)

    default_source = "gold" if args.gold else "empty" if args.empty else "missing"
    rows = []
    for inst in instances:
        instance_id = inst["instance_id"]
        report, pred = results.get(instance_id, {}), preds.get(instance_id, {})
        rows.append(
            {
                "instance_id": instance_id,
                "resolved": bool(report.get("resolved")),
                "patch_exists": bool(patches[instance_id].strip()),
                "patch_applied": bool(report.get("patch_successfully_applied")),
                "grading_error": report.get("grading_error"),
                "prediction_source": pred.get("source", default_source),
                "agent_error": pred.get("error"),
                "conversation_error_category": pred.get("conversation_error_category"),
            }
        )
    summary = {
        "backend": args.backend,
        "total": len(rows),
        "resolved": sum(r["resolved"] for r in rows),
        "patch_applied": sum(r["patch_applied"] for r in rows),
        "empty_patch": sum(not r["patch_exists"] for r in rows),
        "grading_errors": sum(bool(r["grading_error"]) for r in rows),
        "by_prediction_source": dict(Counter(r["prediction_source"] for r in rows)),
        "resolved_by_source": dict(
            Counter(r["prediction_source"] for r in rows if r["resolved"])
        ),
    }
    summary["resolve_rate"] = summary["resolved"] / summary["total"]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(args.out / "instances.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
