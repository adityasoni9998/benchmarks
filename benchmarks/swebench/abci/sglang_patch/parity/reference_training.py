# pyright: reportMissingImports=false
# Runs in hyperion's environment (platoon, tinker-cookbook, transformers), not in this repo's.
"""Training's parsing: hyperion's qwen3_instruct_exact renderer on the tokenised text, then the
proxy's conversion (platoon/train/tinker/proxy.py _parse_response): reply text + tool calls."""

import json
import sys

from platoon.train.tinker.proxy import TinkerLLM
from platoon.train.tinker.qwen3_exact_renderer import register_qwen3_exact_renderer
from tinker_cookbook.renderers import get_renderer
from transformers import AutoTokenizer


MODEL = sys.argv[1]
tok = AutoTokenizer.from_pretrained(MODEL)
register_qwen3_exact_renderer()
renderer = get_renderer(
    "qwen3_instruct_exact", tok, model_name="Qwen/Qwen3-4B-Instruct-2507"
)
im_end = tok.convert_tokens_to_ids("<|im_end|>")
out = {}
for case in json.load(open(sys.argv[2])):
    tokens = tok.encode(case["text"], add_special_tokens=False) + (
        [] if case["truncated"] else [im_end]
    )
    msg, ok = renderer.parse_response(tokens)
    content = TinkerLLM._normalize_message_content(None, msg["content"])
    calls = (
        [
            [tc.function.name, json.loads(tc.function.arguments)]
            for tc in (msg.get("tool_calls") or [])
        ]
        if ok
        else []
    )
    out[case["name"]] = {"content": content, "calls": calls, "parse_success": ok}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print("reference:", len(out))
