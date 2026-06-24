"""多检索源结果去重与合并。"""

from __future__ import annotations

import re
from typing import Any


def normalize_title(title: str) -> str:
    text = title.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def paper_dedup_key(paper: dict[str, Any]) -> str:
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"

    corpus_id = str(paper.get("corpus_id", "")).strip()
    if corpus_id and not corpus_id.startswith("openalex:"):
        return f"s2:{corpus_id}"

    openalex_id = str(paper.get("openalex_id", "")).strip()
    if openalex_id:
        return f"openalex:{openalex_id}"

    title = normalize_title(paper.get("title", ""))
    year = paper.get("year")
    if title:
        return f"title:{title}:{year or ''}"
    return f"raw:{corpus_id}:{title}"


def merge_paper_records(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """合并两路检索结果，优先保留 primary（Semantic Scholar）记录。"""
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for paper in primary + secondary:
        key = paper_dedup_key(paper)
        if key in merged:
            existing = merged[key]
            if existing.get("source") != "semantic_scholar" and paper.get("source") == "semantic_scholar":
                merged[key] = paper
            continue
        merged[key] = paper
        order.append(key)

    return [merged[key] for key in order]
