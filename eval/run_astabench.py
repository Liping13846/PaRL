"""将 Agent 输出转换为 AstaBench 可评测格式的辅助脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_results(results_dir: Path, output_file: Path) -> None:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "astabench" in data:
            records.append(data["astabench"])
        elif "results" in data:
            records.append({
                "query_id": path.stem,
                "results": [
                    {
                        "paper_id": item.get("paper_id") or item.get("corpus_id"),
                        "markdown_evidence": item.get("markdown_evidence", ""),
                    }
                    for item in data["results"]
                ],
            })
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-file", default="results/astabench_submission.json")
    args = parser.parse_args()
    convert_results(Path(args.results_dir), Path(args.output_file))
    print(f"Converted results -> {args.output_file}")


if __name__ == "__main__":
    main()
