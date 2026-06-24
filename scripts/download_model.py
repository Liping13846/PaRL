#!/usr/bin/env python3
"""下载 HuggingFace 模型到本地 checkpoints 目录。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DIR = ROOT / "checkpoints" / "Qwen2.5-1.5B-Instruct"

PASA_MODELS = {
    "crawler": (
        "bytedance-research/pasa-7b-crawler",
        ROOT / "checkpoints" / "pasa-7b-crawler",
    ),
    "selector": (
        "bytedance-research/pasa-7b-selector",
        ROOT / "checkpoints" / "pasa-7b-selector",
    ),
}


def download_one(model_id: str, output_dir: Path, mirror: str) -> None:
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} -> {output_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(output_dir),
        resume_download=True,
        max_workers=4,
    )
    print(f"Done: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HF model to local checkpoints")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id")
    parser.add_argument("--output", default=str(DEFAULT_DIR), help="Local output directory")
    parser.add_argument(
        "--pasa",
        action="store_true",
        help="Download PaSa-7B crawler + selector (for configs/pasa.yaml)",
    )
    parser.add_argument(
        "--mirror",
        default=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),
        help="HuggingFace mirror endpoint (default: hf-mirror.com)",
    )
    args = parser.parse_args()

    if args.mirror:
        print(f"Using HF endpoint: {args.mirror}")

    if args.pasa:
        for role, (model_id, output_dir) in PASA_MODELS.items():
            print(f"\n=== PaSa {role} ===")
            download_one(model_id, output_dir, args.mirror)
        print("\nAll PaSa models ready. Run with: python run_agent.py --config pasa --query \"...\"")
        return

    download_one(args.model, Path(args.output), args.mirror)
    print(f"Config dev.yaml points to: checkpoints/Qwen2.5-1.5B-Instruct")


if __name__ == "__main__":
    main()
