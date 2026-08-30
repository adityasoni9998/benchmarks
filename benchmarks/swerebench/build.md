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
