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

Build and push eight images concurrently in Modal VM Sandboxes:

```bash
uv run modal run benchmarks/swerebench/modal_build_images.py \
  --method push-vm-batch \
  --image-limit 8 \
  --max-workers 8 \
  --cpu 2 \
  --memory 8192 \
  --sandbox-v2
```

Results are written to `.agent_tmp/modal-build-<timestamp>/vm-sandbox.json`.
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
