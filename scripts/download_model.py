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


def clean_incomplete_cache(output_dir: Path) -> int:
    """删除冲突的 .incomplete 分片，避免多进程续传卡死。"""
    cache_dir = output_dir / ".cache" / "huggingface" / "download"
    if not cache_dir.exists():
        return 0
    removed = 0
    for path in cache_dir.glob("*.incomplete"):
        path.unlink(missing_ok=True)
        removed += 1
    for path in cache_dir.glob("*.lock"):
        path.unlink(missing_ok=True)
    return removed


def is_model_complete(output_dir: Path) -> bool:
    single = output_dir / "model.safetensors"
    if single.exists() and single.stat().st_size > 14_000_000_000:
        return True
    index = output_dir / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(output_dir.glob("model-*.safetensors"))
        return len(shards) >= 4
    return False


HF_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "tokenizer*",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "*.md",
    "LICENSE",
    ".gitattributes",
]

PASA_SELECTOR_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "model.safetensors",
]

PASA_CRAWLER_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
]


def _mirror_base(mirror: str, model_id: str) -> str:
    mirror = mirror.rstrip("/")
    return f"{mirror}/{model_id}/resolve/main"


def download_file_http(
    url: str,
    dest: Path,
    *,
    chunk_size: int = 4 * 1024 * 1024,
    max_retries: int = 100,
) -> None:
    import time

    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        downloaded = dest.stat().st_size if dest.exists() else 0
        headers: dict[str, str] = {}
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
        try:
            with requests.get(
                url,
                stream=True,
                headers=headers,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                mode = "ab" if downloaded and response.status_code == 206 else "wb"
                if mode == "wb":
                    downloaded = 0
                total = None
                content_range = response.headers.get("Content-Range", "")
                if content_range and "/" in content_range:
                    total = int(content_range.split("/")[-1])
                elif response.headers.get("Content-Length"):
                    total = downloaded + int(response.headers["Content-Length"])

                with dest.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 / total
                            print(
                                f"\r  {dest.name}: {downloaded / 1e9:.2f}/{total / 1e9:.2f} GB ({pct:.1f}%)",
                                end="",
                                flush=True,
                            )
                if total and downloaded < total - 1024:
                    raise requests.exceptions.ConnectionError("incomplete download")
                print(f"\n  saved {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
                return
        except (requests.exceptions.RequestException, OSError) as exc:
            print(f"\n  retry {attempt}/{max_retries} after error: {exc}")
            time.sleep(min(5 * attempt, 60))
    raise RuntimeError(f"Failed to download {dest.name} after {max_retries} retries")


def download_one_http(
    model_id: str,
    output_dir: Path,
    mirror: str,
    files: list[str],
    *,
    skip_if_complete: bool = True,
) -> None:
    if skip_if_complete and is_model_complete(output_dir):
        print(f"Already complete, skip: {output_dir}")
        return

    base = _mirror_base(mirror, model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"HTTP download {model_id} -> {output_dir}")
    print(f"Endpoint: {mirror}")
    for name in files:
        dest = output_dir / name
        if dest.exists():
            if name == "model.safetensors" and dest.stat().st_size > 14_000_000_000:
                print(f"  skip existing {name} ({dest.stat().st_size / 1e9:.2f} GB)")
                continue
            if name.endswith(".safetensors") and name != "model.safetensors" and dest.stat().st_size > 1_000_000_000:
                print(f"  skip existing {name}")
                continue
            if not name.endswith(".safetensors"):
                print(f"  skip existing {name}")
                continue
        download_file_http(f"{base}/{name}", dest)
    if not is_model_complete(output_dir):
        raise RuntimeError(f"HTTP download finished but model incomplete in {output_dir}")
    print(f"Done: {output_dir}")


def download_one_modelscope(model_id: str, output_dir: Path) -> None:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading via ModelScope {model_id} -> {output_dir}")
    ms_snapshot_download(model_id, local_dir=str(output_dir))
    if not is_model_complete(output_dir):
        raise RuntimeError(f"ModelScope download finished but model shards missing in {output_dir}")
    print(f"Done: {output_dir}")


def download_one(
    model_id: str,
    output_dir: Path,
    mirror: str,
    *,
    workers: int = 1,
    clean_incomplete: bool = False,
    skip_if_complete: bool = True,
) -> None:
    if skip_if_complete and is_model_complete(output_dir):
        print(f"Already complete, skip: {output_dir}")
        return

    if mirror:
        os.environ["HF_ENDPOINT"] = mirror

    if clean_incomplete:
        removed = clean_incomplete_cache(output_dir)
        if removed:
            print(f"Cleaned {removed} stale .incomplete file(s) in {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} -> {output_dir} (workers={workers})")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(output_dir),
        max_workers=workers,
        allow_patterns=HF_ALLOW_PATTERNS,
    )
    if not is_model_complete(output_dir):
        raise RuntimeError(f"Download finished but model shards missing in {output_dir}")
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
        "--selector-only",
        action="store_true",
        help="Only download PaSa-7B selector (crawler must exist)",
    )
    parser.add_argument(
        "--mirror",
        default=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),
        help="HuggingFace mirror endpoint (default: hf-mirror.com)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel download workers (default 1, avoid stuck on Windows)",
    )
    parser.add_argument(
        "--clean-incomplete",
        action="store_true",
        help="Remove stale .incomplete chunks before download",
    )
    parser.add_argument(
        "--use-modelscope",
        action="store_true",
        help="Use ModelScope (国内更稳定，推荐 selector 卡住时用)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use direct HTTP resume download (hf-mirror 大文件卡住时用)",
    )
    args = parser.parse_args()

    if args.mirror:
        print(f"Using HF endpoint: {args.mirror}")

    download_kwargs = {
        "workers": max(1, args.workers),
        "clean_incomplete": args.clean_incomplete,
    }

    if args.http:
        if args.selector_only:
            model_id, output_dir = PASA_MODELS["selector"]
            print("\n=== PaSa selector (HTTP) ===")
            download_one_http(model_id, output_dir, args.mirror, PASA_SELECTOR_FILES)
        elif args.pasa:
            for role, (model_id, output_dir) in PASA_MODELS.items():
                files = PASA_CRAWLER_FILES if role == "crawler" else PASA_SELECTOR_FILES
                print(f"\n=== PaSa {role} (HTTP) ===")
                download_one_http(model_id, output_dir, args.mirror, files)
        else:
            raise SystemExit("Use --http with --selector-only or --pasa")
        print("\nDownload finished.")
        return

    if args.use_modelscope:
        if args.selector_only:
            model_id, output_dir = PASA_MODELS["selector"]
            print("\n=== PaSa selector (ModelScope) ===")
            download_one_modelscope(model_id, output_dir)
        elif args.pasa:
            for role, (model_id, output_dir) in PASA_MODELS.items():
                if is_model_complete(output_dir):
                    print(f"Already complete, skip: {output_dir}")
                    continue
                print(f"\n=== PaSa {role} (ModelScope) ===")
                download_one_modelscope(model_id, output_dir)
        else:
            download_one_modelscope(args.model, Path(args.output))
        print("\nDownload finished.")
        return

    if args.selector_only:
        model_id, output_dir = PASA_MODELS["selector"]
        print("\n=== PaSa selector only ===")
        download_one(model_id, output_dir, args.mirror, **download_kwargs)
        print("\nSelector ready. Run: python run_agent.py --config pasa --query \"...\"")
        return

    if args.pasa:
        for role, (model_id, output_dir) in PASA_MODELS.items():
            print(f"\n=== PaSa {role} ===")
            download_one(model_id, output_dir, args.mirror, **download_kwargs)
        print("\nAll PaSa models ready. Run with: python run_agent.py --config pasa --query \"...\"")
        return

    download_one(args.model, Path(args.output), args.mirror, **download_kwargs)
    print(f"Config dev.yaml points to: checkpoints/Qwen2.5-1.5B-Instruct")


if __name__ == "__main__":
    main()
