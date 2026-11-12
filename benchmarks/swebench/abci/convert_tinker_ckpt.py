# pyright: reportMissingImports=false
# Runs inside miles' SIF (torch, safetensors, miles tools), not in this repo's environment.
"""Convert a miles/tinker Megatron torch_dist checkpoint (dense Qwen3) to Hugging Face safetensors.

miles' tools/convert_torch_dist_to_hf.py reads Megatron's args from <iter>/common.pt, which
tinker checkpoints do not have. This wrapper reuses that tool's loader, planner (which skips
optimizer and *_state entries), conversion and asset copy unchanged, and builds the few
Megatron args the dense Qwen converter needs from the base model's config.json instead.

The input checkpoint is only read. The output directory must not exist.
After saving, verify_hf_ckpt.py checks the result against the base model.

usage (inside miles' container): python convert_tinker_ckpt.py <iter dir> <output dir> <base hf dir>
"""

import json
import os
import sys
import time
from types import SimpleNamespace

import torch


sys.path.insert(0, "/root/miles/tools")
import convert_torch_dist_to_hf as tool  # noqa: E402


input_dir, output_dir, origin_hf_dir = sys.argv[1:4]
if os.path.exists(output_dir):
    raise SystemExit(f"output exists, refusing to overwrite: {output_dir}")

config = json.load(open(os.path.join(origin_hf_dir, "config.json")))
assert config["architectures"] == ["Qwen3ForCausalLM"], config["architectures"]
megatron_args = SimpleNamespace(
    num_layers=config["num_hidden_layers"],
    hidden_size=config["hidden_size"],
    num_attention_heads=config["num_attention_heads"],
    num_query_groups=config["num_key_value_heads"],
    kv_channels=config["head_dim"],
    vocab_size=config["vocab_size"],
    num_experts=None,
)

state_dict = {}
t = time.time()
tool.dist_cp.state_dict_loader._load_state_dict(
    state_dict,
    storage_reader=tool.WrappedStorageReader(input_dir),
    planner=tool.EmptyStateDictLoadPlanner(),
    no_dist=True,
)
print(f"loaded {len(state_dict)} model tensors in {time.time() - t:.1f}s", flush=True)
dtype = {str(v.dtype) for v in state_dict.values() if isinstance(v, torch.Tensor)}
print("checkpoint dtypes:", dtype, flush=True)
# HF Qwen3 weights are bf16; keep the checkpoint's values, stored at the model's dtype.
state_dict = {
    k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v)
    for k, v in state_dict.items()
}

tool.save_tensors(
    megatron_args,
    "qwen3",
    state_dict,
    output_dir,
    5 * 1024**3,
    megatron_args.vocab_size,
)
tool.copy_assets(origin_hf_dir, output_dir)

# Verify against the base model (names/shapes/dtypes, small but nonzero changes).
import runpy  # noqa: E402


sys.argv = ["verify_hf_ckpt.py", output_dir, origin_hf_dir]
runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_hf_ckpt.py"),
    run_name="__main__",
)
