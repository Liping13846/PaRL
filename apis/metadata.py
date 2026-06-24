"""Metadata 约束检索：将 QueryPlan.constraints 转为 OpenAlex filter。"""

from __future__ import annotations

from typing import Any

METADATA_INTENTS = frozenset({"metadata_search", "navigational"})
HARD_CONSTRAINT_KEYS = frozenset({"year", "venue", "cites"})
SEMANTIC_CONSTRAINT_KEYS = ("topic", "method", "dataset")

# 会议简称消歧（追加到 search 文本，避免 ACL 等缩写误匹配）
VENUE_SEARCH_BOOST: dict[str, str] = {
    "ACL": "computational linguistics",
    "EMNLP": "natural language processing",
    "NAACL": "computational linguistics",
}


def has_hard_constraints(constraints: dict[str, Any] | None) -> bool:
    if not constraints:
        return False
    return any(constraints.get(key) not in (None, "") for key in HARD_CONSTRAINT_KEYS)


def should_use_metadata_search(
    intent: str,
    constraints: dict[str, Any] | None,
    *,
    enabled: bool = True,
) -> bool:
    if not enabled or not constraints:
        return False
    if intent in METADATA_INTENTS and has_hard_constraints(constraints):
        return True
    return has_hard_constraints(constraints)


def build_metadata_search_text(query: str, constraints: dict[str, Any] | None) -> str:
    """保留原 query 语义，并补充约束中的 topic/method 及 venue 消歧词。"""
    base = query.strip()
    if not constraints:
        return base

    extras: list[str] = []
    base_lower = base.lower()
    for key in SEMANTIC_CONSTRAINT_KEYS:
        value = constraints.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text.lower() not in base_lower:
            extras.append(text)

    venue = (constraints.get("venue") or "").strip()
    if venue:
        boost = VENUE_SEARCH_BOOST.get(venue.upper(), "")
        if boost and boost.lower() not in base_lower:
            extras.append(boost)

    if extras:
        return f"{base} {' '.join(extras)}".strip()
    return base


def build_openalex_filters(constraints: dict[str, Any] | None) -> list[str]:
    """将硬约束转为 OpenAlex filter 片段（不含 cites，cites 需先解析 work id）。"""
    if not constraints:
        return []

    filters: list[str] = []
    year = constraints.get("year")
    if year not in (None, ""):
        try:
            filters.append(f"publication_year:{int(year)}")
        except (TypeError, ValueError):
            pass

    return filters
