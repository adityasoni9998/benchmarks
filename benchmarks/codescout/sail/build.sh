#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$recipe_dir" rev-parse --show-toplevel)"
sdk_dir="${SDK_DIR:-$repo_root/vendor/software-agent-sdk}"
sdk_sha="$(git -C "$sdk_dir" rev-parse HEAD)"
image="${IMAGE:-adityasoni8/codescout-agent-server-sail-workspace:${sdk_sha:0:7}-slim}"

if [[ -n "$(git -C "$sdk_dir" status --porcelain)" ]]; then
    echo 'SDK checkout must be clean so the image revision is reproducible.' >&2
    exit 1
fi

docker build --progress plain --platform linux/amd64 --target codescout \
    --build-arg "SDK_SHA=$sdk_sha" \
    --file "$recipe_dir/Dockerfile" --tag "$image" "$sdk_dir"
docker image inspect "$image" --format '{{.Size}} bytes; {{.Architecture}}'
if [[ "${1:-}" == --push ]]; then
    docker push "$image"
    docker image inspect "$image" --format '{{json .RepoDigests}}'
fi
