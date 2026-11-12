# pyright: reportMissingImports=false
# Runs inside the SGLang image (miles SIF), not in this repo's environment.
"""The patched SGLang path for non-streaming chat completions (sitecustomize loaded via PYTHONPATH)."""

import json
import sys

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.function_call.qwen25_detector import Qwen25Detector


assert getattr(Qwen25Detector, "_hyperion_tool_call_parse", False), (
    "detector patch not loaded"
)
assert getattr(OpenAIServingChat, "_hyperion_tool_call_parse", False), (
    "serving_chat patch not loaded"
)
tools = [
    Tool(type="function", function=Function(name=n, parameters={"type": "object"}))
    for n in ("terminal", "file_editor", "task_tracker", "finish", "think")
]
detector = Qwen25Detector()
out = {}
for case in json.load(open(sys.argv[1])):
    text = case["text"]
    if case["truncated"]:
        res = OpenAIServingChat._process_tool_calls(
            None, text, tools, {"type": "length", "matched": None}
        )
        content, calls = (
            res.remaining_text,
            [
                [c.function.name, json.loads(c.function.arguments)]
                for c in (res.tool_calls or [])
            ],
        )
    elif not detector.has_tool_call(text):
        content, calls = text, []
    else:
        r = detector.detect_and_parse(text, tools)
        content, calls = (
            r.normal_text,
            [[c.name, json.loads(c.parameters)] for c in r.calls],
        )
    out[case["name"]] = {"content": content, "calls": calls}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("patched sglang:", len(out))
