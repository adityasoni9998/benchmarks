"""Build tool-call parsing test cases: real malformed calls from the base pass@1 SGLang log + edge cases."""

import json
import re
import sys


log = sys.argv[1]
cases = []
seen = set()
for line in open(log, errors="ignore"):
    m = re.search(r"Failed to parse JSON part: (.*), JSON parse error: ", line)
    if m and m.group(1) not in seen and len(cases) < 40:
        seen.add(m.group(1))
        cases.append(
            {
                "name": f"logged_malformed_{len(cases)}",
                "text": "<tool_call>\n" + m.group(1) + "\n</tool_call>",
                "truncated": False,
            }
        )
V = '{"name": "terminal", "arguments": {"command": "ls /testbed"}}'
E = '{"name": "file_editor", "arguments": {"command": "view", "path": "/testbed/setup.py"}}'
synthetic = {
    "valid": f"<tool_call>\n{V}\n</tool_call>",
    "valid_no_newlines": f"<tool_call>{V}</tool_call>",
    "text_then_valid": f"Let me look at the repository.\n<tool_call>\n{V}\n</tool_call>",
    "two_valid": f"<tool_call>\n{V}\n</tool_call>\n<tool_call>\n{E}\n</tool_call>",
    "unknown_tool_grep": '<tool_call>\n{"name": "grep", "arguments": {"pattern": "def foo", "path": "/testbed"}}\n</tool_call>',
    "valid_then_malformed": f'<tool_call>\n{V}\n</tool_call>\n<tool_call>\n{{"name": "terminal", "arguments": {{"command": "echo "x""}}}}\n</tool_call>',
    "malformed_json": '<tool_call>\n{"name": "terminal", "arguments": {"command": "echo "hi""}}\n</tool_call>',
    "unclosed": f"<tool_call>\n{V}",
    "text_then_unclosed": f"I will run it.\n<tool_call>\n{V}",
    "parameters_key": '<tool_call>\n{"name": "terminal", "parameters": {"command": "ls"}}\n</tool_call>',
    "arguments_is_string": '<tool_call>\n{"name": "terminal", "arguments": "{\\"command\\": \\"ls\\"}"}\n</tool_call>',
    "missing_name": '<tool_call>\n{"arguments": {"command": "ls"}}\n</tool_call>',
    "json_list": '<tool_call>\n[{"name": "terminal", "arguments": {"command": "ls"}}]\n</tool_call>',
    "plain_text": "I have fixed the issue; the patch is in place.",
    "empty": "",
}
for k, t in synthetic.items():
    cases.append({"name": k, "text": t, "truncated": False})
cases.append(
    {
        "name": "truncated_complete_call",
        "text": f"<tool_call>\n{V}\n</tool_call>\nAnd now",
        "truncated": True,
    }
)
cases.append(
    {
        "name": "truncated_mid_call",
        "text": '<tool_call>\n{"name": "file_editor", "arguments": {"command": "create", "path": "/testbed/a.py", "file_text": "def f():',
        "truncated": True,
    }
)
json.dump(cases, open(sys.argv[2], "w"), indent=1)
print(len(cases), "cases")
