# pyright: reportMissingImports=false
# Runs inside the SGLang image (miles SIF), not in this repo's environment.
"""Make SGLang's Qwen tool-call parsing follow hyperion training's renderer.

Loaded into every SGLang server process via PYTHONPATH (serve_sglang.sh, which also
sets SGLANG_FORWARD_UNKNOWN_TOOLS=1). Training served the model through hyperion's
LiteLLM proxy, which parsed sampled tokens with tinker-cookbook's parse_content_blocks /
_parse_tool_call_json and hyperion's qwen3_instruct_exact renderer:
  - tool calls are <tool_call>...</tool_call> blocks (no newlines required);
  - a call to a tool the agent does not have is still a call (the agent then reports
    the unknown tool to the model);
  - if any block is not a JSON object with a string "name" and an object
    "arguments", the whole raw response is the reply text (no calls), so the agent
    treats it as a message and the episode ends;
  - a response with no complete block (e.g. an unclosed <tool_call>) is text;
  - a response without the stop token (cut off at max tokens) is not parsed: its raw
    text is the reply (platoon/train/tinker/proxy.py, parse_success False).
SGLang's Qwen25Detector instead requires "<tool_call>\\n...\\n</tool_call>", drops blocks
that fail to parse (and, by default, calls to unknown tools), and parses tool calls in
truncated responses. This patches, for non-streaming chat completions:
  Qwen25Detector.has_tool_call     any "<tool_call>" marks a possible tool call;
  Qwen25Detector.detect_and_parse  training's rules above;
  OpenAIServingChat._process_tool_calls  truncated responses ("length") are raw text.
"""

import importlib.abc
import os
import runpy
import sys


# Keep the image's own sitecustomize (e.g. Debian's apport hook) working.
_SYSTEM_SITECUSTOMIZE = (
    f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/sitecustomize.py"
)
if os.path.exists(_SYSTEM_SITECUSTOMIZE):
    runpy.run_path(_SYSTEM_SITECUSTOMIZE, run_name="_system_sitecustomize")

TOOL_CALL_BLOCK = r"<tool_call>(.*?)</tool_call>"


def parse_like_training(text, tools):
    """Training's tool-call parsing as (normal_text, [(name, arguments_json)]).

    No calls means the whole text is the reply. Kept free of SGLang imports so it can
    be tested against hyperion's renderer directly.
    """
    import json
    import re

    if "<tool_call>" not in text:
        return text, []
    matches = list(re.finditer(TOOL_CALL_BLOCK, text, re.DOTALL))
    calls = []
    for match in matches:
        try:
            call = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            call = None
        if not (
            isinstance(call, dict)
            and isinstance(call.get("name"), str)
            and isinstance(call.get("arguments"), dict)
        ):
            return text, []  # an unparsable block: the whole raw response is text
        calls.append((call["name"], json.dumps(call["arguments"], ensure_ascii=False)))
    if not calls:
        return text, []  # no complete block, e.g. an unclosed <tool_call>
    return text[: matches[0].start()].strip(), calls


def _patch_detector(module) -> None:
    import logging

    from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem

    logger = logging.getLogger("hyperion_tool_call_parse")

    def has_tool_call(self, text):
        return "<tool_call>" in text

    def detect_and_parse(self, text, tools):
        normal_text, calls = parse_like_training(text, tools)
        if "<tool_call>" in text and not calls:
            logger.warning(
                "Unparsable or unclosed tool call; the raw response is the reply"
            )
        tool_indices = {tool.function.name: i for i, tool in enumerate(tools or [])}
        return StreamingParseResult(
            normal_text=normal_text,
            calls=[
                ToolCallItem(
                    tool_index=tool_indices.get(name, -1), name=name, parameters=args
                )
                for name, args in calls
            ],
        )

    module.Qwen25Detector.has_tool_call = has_tool_call
    module.Qwen25Detector.detect_and_parse = detect_and_parse
    module.Qwen25Detector._hyperion_tool_call_parse = True


def _patch_serving_chat(module) -> None:
    from sglang.srt.entrypoints.openai.protocol import ToolCallProcessingResult

    cls = module.OpenAIServingChat
    original = cls._process_tool_calls

    def _process_tool_calls(self, text, tools, finish_reason, *args, **kwargs):
        if (finish_reason or {}).get("type") == "length":
            # Cut off before the stop token: training returned the raw text, unparsed.
            return ToolCallProcessingResult(None, text, finish_reason)
        return original(self, text, tools, finish_reason, *args, **kwargs)

    cls._process_tool_calls = _process_tool_calls
    cls._hyperion_tool_call_parse = True


_PATCHES = {
    "sglang.srt.function_call.qwen25_detector": _patch_detector,
    "sglang.srt.entrypoints.openai.serving_chat": _patch_serving_chat,
}


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = _PATCHES.get(name)
        if patch is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                break
        else:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, _PatchOnImport())
