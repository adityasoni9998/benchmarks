"""Write the instance lists the ABCI SIF build drivers read.

  ids <file>             sorted instance ids, one per line (build_agent_sifs_node.sh,
                         launch_eval.sh shards)
  grading-images <file>  "<instance_id> docker://<official image>" per line, the image
                         the SWE-bench harness grades on (build_grading_sifs_node.sh)

usage: uv run python -m benchmarks.swebench.abci.instance_lists {ids,grading-images} FILE
       [--dataset princeton-nlp/SWE-bench_Verified] [--split test]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from benchmarks.swebench.apptainer_eval import image_uri


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write instance lists for ABCI builds."
    )
    parser.add_argument("kind", choices=["ids", "grading-images"])
    parser.add_argument("file", type=Path)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    from datasets import load_dataset

    instances = sorted(
        (cast(dict[str, Any], x) for x in load_dataset(args.dataset, split=args.split)),
        key=lambda x: x["instance_id"],
    )
    args.file.parent.mkdir(parents=True, exist_ok=True)
    with args.file.open("w") as f:
        for instance in instances:
            if args.kind == "ids":
                f.write(f"{instance['instance_id']}\n")
            else:
                f.write(f"{instance['instance_id']} {image_uri(instance)}\n")
    print(f"wrote {len(instances)} lines to {args.file}")


if __name__ == "__main__":
    main()
