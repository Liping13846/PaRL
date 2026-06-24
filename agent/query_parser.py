from __future__ import annotations

import re
from typing import Any

from agent.prompts import QUERY_PARSE_SYSTEM, QUERY_PARSE_USER
from agent.types import QueryPlan
from models.dual_agent import DualAgent


class QueryParser:
    def __init__(self, dual_agent: DualAgent) -> None:
        self.dual_agent = dual_agent

    def parse(self, query: str) -> QueryPlan:
        llm = self.dual_agent.crawler
        parsed = llm.parse_json(
            QUERY_PARSE_USER.format(query=query),
            system=QUERY_PARSE_SYSTEM,
        )
        if parsed:
            return QueryPlan(
                original_query=query,
                intent=parsed.get("intent", "semantic_search"),
                constraints=parsed.get("constraints", {}) or {},
                sub_queries=self._normalize_queries(parsed.get("sub_queries", []), query),
                expanded_terms=parsed.get("expanded_terms", []) or [],
            )
        return self._fallback_parse(query)

    @staticmethod
    def _normalize_queries(sub_queries: list[str], original: str) -> list[str]:
        cleaned = [q.strip() for q in sub_queries if isinstance(q, str) and q.strip()]
        if cleaned:
            return cleaned[:4]
        return [original.strip()]

    @staticmethod
    def _fallback_parse(query: str) -> QueryPlan:
        """无 LLM 或解析失败时的规则兜底。"""
        constraints: dict[str, Any] = {}
        year_match = re.search(r"\b(19|20)\d{2}\b", query)
        if year_match:
            constraints["year"] = int(year_match.group())

        venue_candidates = ["ACL", "EMNLP", "NeurIPS", "ICLR", "CVPR", "ICCV"]
        for venue in venue_candidates:
            if venue.lower() in query.lower():
                constraints["venue"] = venue
                break

        return QueryPlan(
            original_query=query,
            intent="semantic_search",
            constraints=constraints,
            sub_queries=[query.strip()],
            expanded_terms=[],
        )
