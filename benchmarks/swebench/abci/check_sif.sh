#!/usr/bin/env bash
# Integrity check of one SWE-bench SIF (agent or grading image), given as <dir>/<instance id>.sif.
# 1. file size equals the end of the squashfs partition recorded in the SIF header (no truncation);
# 2. unsquashfs can read the superblock and list the whole tree (squashfs keeps its inode and
#    directory tables at the end of the image, so a partial write fails here);
# 3. the container starts and /testbed is a git repository whose HEAD resolves.
# Prints "OK <id> testbed-files=<n> head=<sha>" or "BAD <id> <reason>".
#
# usage: check_sif.sh <sif path or link>
# env:   APPTAINER (default: apptainer on PATH), UNSQUASHFS (default: Apptainer's bundled one,
#        else unsquashfs on PATH)
set -uo pipefail
link="$1"
id=$(basename "$link" .sif)
APPTAINER="${APPTAINER:-$(command -v apptainer || echo "$HOME/.local/apptainer/x86_64/bin/apptainer")}"
bundled="$(dirname "$(readlink -f "$APPTAINER")")/../libexec/apptainer/bin/unsquashfs"
UNSQ="${UNSQUASHFS:-$([[ -x "$bundled" ]] && echo "$bundled" || command -v unsquashfs)}"
sif=$(readlink -f "$link") || { echo "BAD $id unresolved-link"; exit; }
[[ -f "$sif" ]] || { echo "BAD $id missing-target"; exit; }
part=$("$APPTAINER" sif list "$sif" 2>&1 | awk -F'|' '/Squashfs/ {gsub(/ /,"",$4); print $4}')
[[ -n "$part" ]] || { echo "BAD $id no-squashfs-partition"; exit; }
start=${part%-*}
end=${part#*-}
size=$(stat -c %s "$sif")
[[ "$size" -eq "$end" ]] || { echo "BAD $id size=$size partition-end=$end"; exit; }
fs_bytes=$("$UNSQ" -o "$start" -s "$sif" 2>/dev/null | awk '/Filesystem size/ {print $3; exit}')
[[ -n "$fs_bytes" ]] || { echo "BAD $id unreadable-superblock"; exit; }
nfiles=$("$UNSQ" -o "$start" -l "$sif" 2>/dev/null | grep -c '^squashfs-root/testbed/')
[[ "$nfiles" -gt 50 ]] || { echo "BAD $id listing-failed testbed-files=$nfiles"; exit; }
head=$(timeout 120 "$APPTAINER" exec --containall "$sif" sh -c 'cd /testbed && git rev-parse HEAD' 2>/dev/null)
[[ "$head" =~ ^[0-9a-f]{40}$ ]] || { echo "BAD $id exec-or-git-failed"; exit; }
echo "OK $id testbed-files=$nfiles head=${head:0:9}"
