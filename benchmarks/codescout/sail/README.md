# CodeScout on Sail

This recipe builds one reusable CodeScout agent-server image. It contains the
CodeScout localization tool and prompt, Git, ripgrep, tmux, and the source
OpenHands server. It does not install browser binaries, desktop services,
Node/ACP agents, or benchmark-specific environments.

The SDK submodule pins `d3fd1c497860fdcba49e3304f348b10a4f1c370b` on
[`sail-workspace-hardened`](https://github.com/adityasoni9998/software-agent-sdk/tree/sail-workspace-hardened).
That branch combines the current Modal backend with Sail and the CodeScout
example assets. Its Sail extra supports SDK `>=0.11.0,<0.12`.

## Build on Ogma

Clone this branch under `/tmp` and initialize the submodule, then run:

```bash
git submodule update --init
bash benchmarks/codescout/sail/build.sh --push
```

Docker must already be authenticated to Docker Hub for publishing. `SDK_DIR`
can select another clean SDK checkout; `IMAGE` overrides the destination tag.
The script derives the revision from that checkout and embeds it in the server.
It prints the published digest. The build context is the SDK checkout; the
Dockerfile-specific ignore file deliberately includes its example assets.

Built and published on Ogma on 2026-09-16:

```text
adityasoni8/codescout-agent-server-sail-workspace:d3fd1c4-slim
adityasoni8/codescout-agent-server-sail-workspace@sha256:4cbd6c9b815dab1cb08a5fe85b1b7d1413bd1b6726d873f9b25003be0a765400
```

Architecture: amd64. Docker size: 706,155,691 bytes. Registry compressed layers:
216,453,803 bytes. The build installs from the SDK lockfile without development
dependencies. The image includes both workspace extras for inspection and reuse;
the Sail credential belongs on the controlling client, never in the image.

## Run

Use the digest above with `SailWorkspace(target_type="source", size="s",
memory_limit_gib=2, disk_limit_gib=8)`. Sail ignores OCI entrypoints, so
SailWorkspace explicitly starts `/agent-server/.venv/bin/python -m
openhands.agent_server` and authenticates its HTTP listener with a fresh session
key. Check `expected_server_git_sha` against the pinned SDK revision.

Sail caches imported image definitions per organization. A digest makes the
content stable; do not force a rebuild for each rollout. The first image import
is separate from VM provisioning. See the
[Sail image documentation](https://docs.sailresearch.com/sailboxes-images).

The Platoon `plugins/codescout_sail` plugin supplies a synthetic CodeScout smoke
test that needs no benchmark dataset or training service:

```bash
python -m platoon.codescout_sail.smoke \
  --endpoint http://PUBLIC_HOST:PUBLIC_PORT/v1 \
  --function add --output /tmp/codescout-smoke.json
```

Set `SAIL_API_KEY` on the client and `VLLM_API_KEY` to the model endpoint key.
The test uses the plugin's prompt, terminal and localization tools, and reward
calculation. It validates an exact localization and records lifecycle and spend
responses after cleanup. No training is started.
