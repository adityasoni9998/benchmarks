# SWE-bench Verified on ABCI with Apptainer

Scripts for evaluating miles/tinker checkpoints on SWE-bench Verified on ABCI full nodes
(8 H200 GPUs, 192 CPUs): SGLang serves the model, `run_infer` runs agents in Apptainer
sandboxes set up like hyperion training's, and `grade_run` grades the patches.

## One-time setup

1. Instance lists (`uv run python -m benchmarks.swebench.abci.instance_lists ...`):
   - `ids $SIF_DIR/all_ids.txt`: sorted instance ids (agent SIF builds, eval shards);
   - `grading-images $GRADING_SIF_DIR/images.txt`: official image per instance.
2. Agent SIFs (official image + agent server): on one or more nodes,
   `SIF_DIR=... OPENHANDS_APPTAINER_BUILD_ROOT=... build_agent_sifs_node.sh /local/$JOB 32`.
3. Grading SIFs (official images as they are):
   `GRADING_SIF_DIR=... build_grading_sifs_node.sh /local/$JOB 48`.
4. Check every SIF: `check_sif.sh <dir>/<instance_id>.sif` (size vs SIF header, full
   squashfs listing, container starts and `/testbed` HEAD resolves). A grading SIF's
   `/testbed` HEAD should equal its agent SIF's.

## Checkpoints

`convert_ckpt.sh <.../weights/<step>/iter_0000001> <hf output dir>` converts a tinker
Megatron `torch_dist` policy checkpoint to Hugging Face safetensors with miles' own converter
(`convert_tinker_ckpt.py`; tinker checkpoints lack `common.pt`, so the few Megatron shape
arguments come from the base model's `config.json`) and checks the result against the base
model (`verify_hf_ckpt.py`).

## One evaluation

On a node:

```bash
serve_sglang.sh <hf model dir> <served name>          # all 8 GPUs, data parallel
OPENHANDS_APPTAINER_SIF_DIR=<agent SIFs> launch_eval.sh pass1 <llm config> <run dir> /local/$JOB
GRADING_SIF_DIR=<grading SIFs> eval_then_grade.sh <run dir> /local/$JOB   # detached; grades once all run_infer processes exit
monitor.sh <run dir> <node>                            # read-only snapshots
```

The LLM config is a `run_infer` JSON config with `"model": "openai/<served name>"` and
`"base_url": "http://127.0.0.1:30000/v1"` plus sampling (temperature 0.7, top_p 0.8,
top_k 20, max_output_tokens 4096).

### Modes (`launch_eval.sh`)

| | `harness` | `pass1` (genuine pass@1, as in hyperion training) |
|---|---|---|
| turns | 500 per `run()` call | 75 |
| fake user ("please continue") | up to 10 | none: a message ends the episode |
| conversation timeout | 1 hour | 1 hour |
| agent-caused conversation error (turn limit, stuck, context window, timeout, malformed tool call) | instance retried | episode ends, its patch is graded |
| other failures (sandbox start, sandbox or LLM-server failure) | retried (4 attempts) | retried (3 attempts) |

Sandbox start is retried 3 times before an attempt counts (`APPTAINER_START_ATTEMPTS`).

## Grading

`python -m benchmarks.swebench.grade_run --run-dir RUN --out OUT --backend {apptainer,modal}`
merges every shard's `output.jsonl` and `output_errors.jsonl`, removes setup-file and binary
hunks once (`predictions.jsonl`), and reports over the whole dataset (`summary.json`,
`instances.jsonl`); an instance without a prediction counts as unresolved. `--gold` and
`--empty` grade the gold or empty patches to validate a backend. The `modal` backend runs the
official harness on Modal (`eval_infer.run_swebench_evaluation`), which needs Modal credentials;
by default it uses the official images and hyperion's sandbox requests (`cpu=0.8`,
`memory=128` MiB). Each instance's tests time out after 1800 s (`--timeout`, the SWE-bench
harness default); on Modal, 500 instances in parallel take about 10 minutes.

Known gold-patch failures with the Apptainer backend (491/500 resolve): four `psf__requests`
instances need internet access; `pylint-dev__pylint-6528`, `pylint-dev__pylint-7277` and
probably `sphinx-doc__sphinx-9711` fail on the images' installed package versions;
`astropy__astropy-7606` fails under `--cleanenv`'s `LANG=C` (ASCII in its Python 3.6);
`django__django-10097` is not diagnosed yet.
