## ABCI: reuse the published Modal images

Use the `swerebench_apptainer` branch, based on `swerebench_modal` at
`8c93073`. The older `apptainer_eval_support` branch predates this work.
The SDK submodule is pinned to `5acdf05ae2fc6224f92114db6f557b181cd0f280`.

The Modal builder pushes to `docker.io/adityasoni8/eval-agent-server` using
the `dockerhub-adityasoni8` Modal secret. Its `source-minimal` images already
contain OpenHands and a final layer that re-runs the dataset install command
in `/testbed`, repairing editable installs. Conda activation supports both
`/opt/conda` and `/opt/miniconda3`, and temporarily disables `nounset`.
The image tags include the `swerebench-testbed-reinstall-v2` repair key
(hashed/truncated by `docker_custom_tag`). These fixes are retained on this
branch. Converting these images to SIF preserves their installed filesystem;
there is no need to install OpenHands again.

Registry audit on 2026-10-25: all **6,542 unique images** in the `filtered`
split at dataset revision `89cdfbab4ab1bd8f5a658bb212d1b63624f4f881` have
their expected repaired `5acdf05` tags published; **zero are missing**.
The mapping is saved locally at
`/groups/gai51740/aci19058gb/apptainer/manifests/filtered-registry-audit.jsonl`.
This checks tag presence for the entire split and image configuration/history
for one sample, without downloading image layers. It does not verify all blobs.

The original Docker repair step logs failures and continues. Registry presence
does not establish that every repair succeeded or that each image passes tests.
Smoke-test conversion and runtime behavior before a full collection download.

Authenticate once on the host, entering a Docker Hub access token at the prompt:

```bash
umask 077
apptainer registry login --username adityasoni8 docker://docker.io
```

This saves the small credentials file in Apptainer's default home configuration;
image data is stored separately below. Never put the token in this repository.
Authenticated Docker Pro pulls have no pull-rate quota, though Docker's abuse
limits still apply: [Docker Hub limits](https://docs.docker.com/docker-hub/usage/).
See also [Apptainer registry authentication](https://apptainer.org/docs/user/main/registry.html).

Inside an allocated compute node, from the repository root:

```bash
# PBS_LOCALDIR normally supplies this. If absent, use this allocation's /local path.
# export ABCI_SCRATCH=/local/JOB_ID.pbs1
source scripts/apptainer_abci_env.sh
```

The helper uses `${USER_ID:-$(id -un)}` and sets:

| Variable | Location |
| --- | --- |
| `CONTAINER_ROOT` | `/groups/gai51740/<account>/apptainer` |
| `OPENHANDS_APPTAINER_BUILD_ROOT` | `$CONTAINER_ROOT/sif` |
| `APPTAINER_CACHEDIR` | `$CONTAINER_ROOT/cache/apptainer` |
| `APPTAINER_TMPDIR`, `TMPDIR` | `${ABCI_SCRATCH:-$PBS_LOCALDIR}/apptainer` |
| `UV_CACHE_DIR`, `HF_HOME` | `$CONTAINER_ROOT/cache/uv`, `$CONTAINER_ROOT/cache/huggingface` |

It rejects missing scratch and paths outside `/local`, matching the host-side
scratch convention in `~/miles/deploy/abci/SINGLE_NODE.md` and `MULTI_NODE.md`.
Run it separately on each allocated node. The old Slurm recipes below target
another cluster and should not be submitted on ABCI.

The helper uses `umask 077` and private directories. On this account,
`/groups/gai51740/aci19058gb` also has `0700` access permissions (with the
group filesystem's inherited setgid bit).
Other group members cannot traverse or modify its contents. However,
`/groups/gai51740` itself is group-writable without a sticky bit; preventing
other group members from renaming the top-level account directory requires
an administrator-managed parent policy. Do not change the shared parent's mode.

For a first conversion, this tox image digest was verified in the registry on
2026-10-25. Its image configuration includes the OpenHands entrypoint and the
testbed reinstall layer:

```bash
apptainer build \
  --mksquashfs-args "-comp zstd -Xcompression-level 3 -processors 2 -mem 1G" \
  "$OPENHANDS_APPTAINER_BUILD_ROOT/tox-dev__tox-1243.sif" \
  docker://docker.io/adityasoni8/eval-agent-server@sha256:ae7f514917323c7f1f4cb871107cc2064d613177a989657a333af49a627626a3

apptainer exec --cleanenv \
  "$OPENHANDS_APPTAINER_BUILD_ROOT/tox-dev__tox-1243.sif" \
  /agent-server/.venv/bin/python -c 'import openhands.agent_server; print("OpenHands import OK")'
```

Keep caching enabled and use the shared per-user `APPTAINER_CACHEDIR` above.
Cached OCI blobs allow images with identical layers, retries, and subsequent jobs
to reuse downloads. Do not pass `--disable-cache` or set
`APPTAINER_DISABLE_CACHE=true`. Budget disk for cached blobs and any cached
converted SIFs in addition to final outputs. Inspect usage with
`apptainer cache list`; do not clean the cache while conversions are active.
Parallel access requires atomic rename support on the cache filesystem; a shared
cache does not guarantee that simultaneous cold requests download each layer only
once. See [Apptainer caching](https://apptainer.org/docs/user/1.4/build_env.html).
Local scratch must fit the unpacked image and conversion overhead per concurrent
worker; compressed Docker sizes do not measure scratch requirements. Start with
one worker. Use digest-pinned references when preparing the full build manifest.

Conversion does not make `/testbed` writable inside an immutable SIF. The plugin
still needs a per-instance writable workspace or overlay and a working
agent-server launch; an import smoke test alone does not validate that integration.

### Batch conversion

`scripts/pull_apptainer_images.py` is a standard-library-only scheduler for
ordinary `apptainer build` commands; it does not install OpenHands or execute
benchmark code. Run it with the host's `python3`. The full audited manifest is
now pinned by digest at `$CONTAINER_ROOT/manifests/filtered-pinned.jsonl`.
Every row contains `instance_id` and `image`, plus registry size/layer metadata.
The inventory contains 16.89 TB of summed compressed layers and 9.69 TB of unique
layers; final SIF storage is additional to the persistent layer cache.

After sourcing the ABCI helper, choose the concurrency for the allocated nodes:

```bash
BUILD_WORKERS=64
python3 scripts/pull_apptainer_images.py \
  --manifest "$CONTAINER_ROOT/manifests/filtered-pinned.jsonl" \
  --sif-dir "$OPENHANDS_APPTAINER_BUILD_ROOT" \
  --report-dir "$CONTAINER_ROOT/logs/build-$(date +%Y%m%d-%H%M%S)" \
  --workers "$BUILD_WORKERS" \
  --mksquashfs-args "-comp zstd -Xcompression-level 3 -processors 2 -mem 1G"
```

Each output is named `{instance_id}.sif`, directly matching the plugin template.
The runner writes successful completion records under `sif/.completed/` and
skips matching completed images on restart. File existence alone is not enough.
It refuses to overwrite an untracked SIF or one recorded for another source.
Failed images are retried twice by default, without stopping other workers.
Each attempt has its own log, and `results.jsonl` and `summary.json` record the
outcome. A nonzero batch exit indicates failures after all images were attempted.
Ctrl-C/SIGTERM stops active workers and leaves the batch resumable. The default
per-attempt timeout is 1,800 seconds; override with `--timeout` if needed.

An outer loop catches unexpected Python exceptions from the runner and resumes
after 60 seconds, allowing up to 100 restarts (101 total passes). Existing workers
finish before the next pass starts, and successful images are skipped. Completion
records are replaced atomically; SIFs still use their final filenames directly.
Attempt logs append on restart so earlier diagnostics remain available.
Normal completion, a completed batch with failed images, invalid CLI arguments,
and Ctrl-C/SIGTERM do not trigger this outer loop. Failed images still get two
retries per pass. A killed Python process, expired PBS allocation, or node failure
requires relaunching the command; the same completion records allow it to resume.

For disjoint multi-node runs, use the same manifest with non-overlapping
`--offset`/`--limit` ranges (a limit of zero means the remainder). Do not run
overlapping ranges against the same SIF directory. Cache storage is shared;
temporary extraction files stay on each node's local scratch.

The runner gives each Apptainer process a temporary private cache index beneath
`$APPTAINER_CACHEDIR/build-metadata/`, with its `cache/blob/blobs` directory linked
to the persistent shared blob directory. This keeps downloaded layers reusable
across all nodes without concurrent writes to the same OCI `index.json`.
This is necessary for Apptainer 1.5.5: the initial ten-node launch exposed malformed
JSON from concurrent index updates. Private metadata is removed after each build;
shared blobs and successful SIFs remain. Do not clean the cache during a build.

The 2026-10-25 experiments and per-image timings are stored under
`$CONTAINER_ROOT/benchmarks/compression-20261025/`. The selected concurrency is
64 builds. Its 128-image sample finished in 890.82 seconds (517 images/hour),
with a mean per-image duration of 380.15 seconds and no failures. A 128-way test
was not started. The observed batch rate alone does not guarantee a sub-12-hour
full run; continuous queue utilization and image mix will affect the result.

The original gzip and zstd
test SIFs have identical contents, symlink targets, stored ownership, and modes
across 123,844 entries, apart from the generated build-date label. Both started
the plugin's agent server and executed a writable `/testbed` command successfully.
Both also reproduced an existing plugin cleanup error: host-side `shutil.rmtree`
cannot remove some overlay directories owned by mapped subordinate UIDs. This
requires a plugin cleanup fix before sustained training, independently of the
compression choice. See `plugin-smoke.log` and `plugin-smoke-gzip.log`.

If an image is missing from the registry, the retained source builder installs
the pinned SDK and applies the repair directly. Select only the missing instances:

```bash
uv run python -m benchmarks.swerebench.apptainer_build \
  --dataset nebius/SWE-rebench --split filtered --select missing-instances.txt \
  --max-workers 1 \
  --sif-dir "$OPENHANDS_APPTAINER_BUILD_ROOT" \
  --apptainer-cache-dir "$APPTAINER_CACHEDIR" \
  --apptainer-tmp-dir "$APPTAINER_TMPDIR"
```

Direct `%post` builds require working rootless/fakeroot support; installing the
Apptainer executable alone does not establish that. Prefer registry conversion
for images already built. Avoid `apptainer_build2.py`, which is an older variant.

## Historical cluster recipes

```bash
export APPTAINER_CACHEDIR=/data/user_data/adityabs/apptainer_cache
export APPTAINER_TMPDIR=/tmp/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR
export OPENHANDS_APPTAINER_BUILD_ROOT=$APPTAINER_CACHEDIR

export GH_USER='adityasoni9998'
export REGISTRY_AUTH_FILE="$HOME/.config/containers/auth.json"
source .venv/bin/activate
python scripts/mirror_swerebench_to_ghcr.py --n-limit 2 --dry-run

python benchmarks/swerebench/apptainer_build2.py \
  --dataset nebius/SWE-rebench \
  --split filtered \
  --max-workers 32
```

```bash
.venv/bin/python scripts/make_swerebench_build2_shards.py \
  --dataset nebius/SWE-rebench \
  --split filtered \
  --shard-count 16 \
  --output-dir /data/user_data/adityabs/apptainer_cache/swerebench_build2_shards_16
```

```bash
sbatch \
    --exclude=babel-w9-16,babel-t9-20 \
    scripts/sbatch_swerebench_build2_one_shard.sh 3
```

## Docker

Build one or two images locally before scaling up. The Docker builder uses the
same SDK submodule and `source-minimal` target as SWE-Smith, then repeats each
row's install command in `/testbed` just like the Apptainer builder.

```bash
uv run python benchmarks/swerebench/docker_build.py \
  --dataset nebius/SWE-rebench \
  --split filtered \
  --image local/swerebench-agent-server \
  --image-limit 2 \
  --max-workers 1
```

The first tag for every successful build is written to
`builds/nebius/SWE-rebench/filtered/docker-manifest.jsonl`. Launch the agent
server from one of those tags with:

```bash
docker run --rm --name swerebench-agent-smoke -p 8000:8000 IMAGE_TAG
```

Use `--push` with an authenticated registry image name to publish images. As in
the SWE-Smith builder, pushed layers default to eStargz via
`OPENHANDS_IMAGE_COMPRESSION=estargz`. Set that environment variable to another
value to use the normal registry exporter.

## Modal

Build and push images in Modal VM Sandboxes. Build logs are quiet by default;
the local process reports completed image and sandbox counts. Successful pushes
are submitted concurrently to a separate local pool that imports and publishes
Modal named Images, without blocking the next build in a VM.

Before launching VM workers, the command builds (or reuses Modal's cache for)
the shared runtime once and publishes it as
`swerebench-modal-vm-builder:latest`. Every VM then starts from that named
Image, and VM batches are launched through Modal's asynchronous V2 API.

Plan the entire filtered split without creating Sandboxes:

```bash
uv run modal run --profile neulab --quiet \
  benchmarks/swerebench/modal_build_images.py \
  --method push-vm-repo-batch \
  --image-limit 0 \
  --max-images-per-vm 8 \
  --dry-run
```

Run the entire filtered split with all 818 VM Sandboxes requested concurrently
and 32 independent named-image publication workers:

```bash
uv run modal run --profile neulab --quiet \
  benchmarks/swerebench/modal_build_images.py \
  --method push-vm-repo-batch \
  --image-limit 0 \
  --max-images-per-vm 8 \
  --max-workers 818 \
  --publish-max-workers 32 \
  --cpu 1 \
  --memory 4096 \
  --sandbox-timeout-seconds 3600 \
  --sandbox-v2
```

Requests are ordered by repository and packed into batches of eight. An
underfilled repository shares its VM with the next repository, and only the
final VM may contain fewer than eight images. A failed image build is recorded
and the VM continues with its next request. Named-image publication failures
are also isolated from later publications and from VM builds.

Build results are written to
`.agent_tmp/modal-build-<timestamp>/vm-repo-batch.json`, named-image publication
results to `modal-publish.json`, and any failures to `failures.json`. Failure
records include the stage, instance id, repository, base image, registry image,
Modal image when applicable, sandbox id, and error. The command exits nonzero
after all work is attempted if any required build or publication failed.

To recover an interrupted full run, first skip registry images that already
exist and rebuild only missing Docker images:

```bash
uv run modal run --profile neulab --quiet \
  benchmarks/swerebench/modal_build_images.py \
  --method push-vm-repo-batch --image-limit 0 \
  --max-images-per-vm 8 --max-workers 256 \
  --cpu 1 --memory 4096 --sandbox-timeout-seconds 7200 --sandbox-v2 \
  --no-force-build --no-publish-named-images
```

After that succeeds, publish every dataset registry image in a separate,
restartable 256-worker job. Publishing an existing name updates that name and
is safe to repeat. Each registry import and name publication has a one-hour
timeout by default, so a stalled Modal stream is recorded as a failure instead
of holding the entire job open; override it with
`--publish-timeout-seconds` when needed:

```bash
uv run modal run --profile neulab --quiet \
  benchmarks/swerebench/modal_build_images.py \
  --method publish-dataset --image-limit 0 --publish-max-workers 256
```

Compare requested and actual CPU/memory in Modal's post-run metrics rather than
polling the build Sandboxes. The benchmark and SDK revisions are pinned in the
builder image, and the local reinstall files are overlaid so an uncommitted
review build exercises the exact files under test.

Publish one reproducibly random successful registry image as a named Modal
Image with:

```bash
uv run modal run benchmarks/swerebench/modal_build_images.py \
  --method publish-random-result \
  --results-file .agent_tmp/modal-build-<timestamp>/vm-sandbox.json \
  --random-seed 20260830
```

Named images follow SWE-Smith's rules: replace `/` with `__`, remove
`.x86_64` and shorten `-source-minimal` where needed, then use an eight-character
content hash if either Modal name component still exceeds 64 characters.
