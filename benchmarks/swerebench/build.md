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
