"""智能引文扩展：引文邻居拉取 + 相似度预筛。"""

from __future__ import annotations

import re
from typing import Any

from apis.merge import merge_paper_records, paper_dedup_key


def extract_query_tokens(query: str) -> list[str]:
    english = re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    return english + chinese


def semantic_overlap_text(text: str, query: str) -> float:
    blob = text.lower()
    tokens = extract_query_tokens(query)
    if not tokens:
        return 0.4
    hits = sum(1 for token in tokens if token in blob)
    return hits / len(tokens)


def constraint_overlap_text(text: str, constraints: dict[str, Any]) -> float:
    if not constraints:
        return 1.0
    blob = text.lower()
    total = 0
    matched = 0
    for key, value in constraints.items():
        if value in (None, "") or key in {"year", "venue", "cites"}:
            continue
        total += 1
        if str(value).lower() in blob:
            matched += 1
    if total == 0:
        return 1.0
    return matched / total


def score_citation_candidate(
    paper: dict[str, Any],
    query: str,
    constraints: dict[str, Any] | None = None,
) -> float:
    constraints = constraints or {}
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    semantic = semantic_overlap_text(text, query)
    constraint = constraint_overlap_text(text, constraints)

    score = 0.7 * semantic + 0.3 * constraint

    year = constraints.get("year")
    paper_year = paper.get("year")
    if year not in (None, "") and paper_year == int(year):
        score += 0.08

    venue = (constraints.get("venue") or "").strip()
    paper_venue = (paper.get("venue") or "").lower()
    if venue and venue.lower() in paper_venue:
        score += 0.08

    return min(score, 1.0)


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for paper in candidates:
        key = paper_dedup_key(paper)
        if key in merged:
            continue
        merged[key] = paper
        order.append(key)
    return [merged[key] for key in order]


def merge_citation_neighbors(
    references: list[dict[str, Any]],
    cited_by: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for paper in references:
        paper["expand_direction"] = "reference"
    for paper in cited_by:
        paper["expand_direction"] = "cited_by"
    merged = merge_paper_records(references, cited_by)
    return dedupe_candidates(merged)


def select_citation_candidates(
    candidates: list[dict[str, Any]],
    query: str,
    constraints: dict[str, Any] | None = None,
    *,
    top_k: int = 8,
    min_score: float = 0.15,
    fallback_k: int = 3,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for paper in candidates:
        score = score_citation_candidate(paper, query, constraints)
        paper = dict(paper)
        paper["expand_prefilter_score"] = round(score, 4)
        scored.append((score, paper))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [paper for score, paper in scored if score >= min_score][:top_k]
    if selected:
        return selected

    return [paper for _, paper in scored[:fallback_k]]


def should_trigger_citation_expand(
    *,
    candidate_count: int,
    high_score_count: int,
    intent: str,
    min_recall: int = 8,
    min_high_score: int = 2,
) -> bool:
    """召回不足或高相关不足时触发引文扩展。"""
    if intent in {"metadata_search", "navigational"}:
        if candidate_count >= min_recall and high_score_count >= min_high_score:
            return False
    if candidate_count >= min_recall and high_score_count >= min_high_score + 1:
        return False
    return True
