# pyright: reportMissingImports=false
# Runs inside miles' SIF (torch, safetensors, miles tools), not in this repo's environment.
"""Check a converted HF checkpoint against the base model.

Passes if tensor names, shapes and dtypes match the base model, most tensors changed, and every
tensor's relative change is small (a wrong layer/QKV mapping would give changes of order 1).
usage: python verify_hf_ckpt.py <converted dir> <base hf dir>
"""

import collections
import json
import os
import re
import sys

import safetensors.torch


def load_all(directory):
    index = json.load(open(os.path.join(directory, "model.safetensors.index.json")))
    tensors = {}
    for filename in sorted(set(index["weight_map"].values())):
        tensors.update(safetensors.torch.load_file(os.path.join(directory, filename)))
    return tensors


converted, base = load_all(sys.argv[1]), load_all(sys.argv[2])
assert set(converted) == set(base), (
    sorted(set(base) - set(converted))[:5],
    sorted(set(converted) - set(base))[:5],
)


def group(name):
    return re.sub(r"layers\.\d+\.", "layers.N.", name)


relative, unchanged = {}, collections.Counter()
for name, tensor in base.items():
    assert (
        converted[name].shape == tensor.shape and converted[name].dtype == tensor.dtype
    ), name
    delta = (converted[name].float() - tensor.float()).norm().item()
    relative[name] = delta / max(tensor.float().norm().item(), 1e-12)
    if delta == 0:
        unchanged[group(name)] += 1
worst = sorted(relative.items(), key=lambda kv: -kv[1])[:3]
print(
    f"{len(base)} tensors: names/shapes/dtypes match base; {len(base) - sum(unchanged.values())} changed"
)
print("unchanged by group:", dict(unchanged))
print("largest relative changes:", [(group(k), f"{v:.2e}") for k, v in worst])
assert sum(unchanged.values()) < 0.5 * len(base), (
    "most tensors identical to base: wrong checkpoint?"
)
assert worst[0][1] < 0.1, (
    "a tensor changed by >10% relative: suspect a conversion mapping error"
)
print("OK")
