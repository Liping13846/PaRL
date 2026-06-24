from __future__ import annotations

import re
from typing import Any

from agent.types import Paper, QueryPlan


class Ranker:
    def __init__(self, score_threshold: float = 0.5) -> None:
        self.score_threshold = score_threshold

    def rank(
        self,
        papers: list[Paper],
        query_plan: QueryPlan,
        min_return: int = 5,
    ) -> list[Paper]:
        scored: list[Paper] = []
        for paper in papers:
            paper.relevance_score = self._fusion_score(paper, query_plan)
            paper.tier = self._to_tier(paper.relevance_score)
            paper.relevance_reason = self._build_reason(paper, query_plan)
            paper.evidence = self._extract_evidence(paper, query_plan)
            scored.append(paper)

        scored.sort(key=lambda p: p.relevance_score, reverse=True)
        above_threshold = [p for p in scored if p.relevance_score >= self.score_threshold]
        if above_threshold:
            return self._apply_mmr(above_threshold)

        fallback = scored[:min_return]
        for paper in fallback:
            if paper.tier == "filtered":
                paper.tier = "partially_relevant"
        return fallback

    def _fusion_score(self, paper: Paper, query_plan: QueryPlan) -> float:
        semantic = self._semantic_overlap(paper, query_plan.original_query)
        constraint = self._constraint_score(paper, query_plan.constraints)
        authority = min(paper.citation_count / 100.0, 1.0)
        recency = self._recency_score(paper.year)
        return 0.55 * semantic + 0.25 * constraint + 0.1 * authority + 0.1 * recency

    @staticmethod
    def _extract_query_tokens(query: str) -> list[str]:
        english = re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
        chinese = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        return english + chinese

    @staticmethod
    def _semantic_overlap(paper: Paper, query: str) -> float:
        text = f"{paper.title} {paper.abstract}".lower()
        tokens = Ranker._extract_query_tokens(query)
        if not tokens:
            return 0.4
        hits = sum(1 for token in tokens if token.lower() in text or token in text)
        return hits / len(tokens)

    @staticmethod
    def _constraint_score(paper: Paper, constraints: dict[str, Any]) -> float:
        if not constraints:
            return 1.0
        total = 0
        matched = 0
        for key, value in constraints.items():
            if value in (None, ""):
                continue
            total += 1
            if key == "year" and paper.year == int(value):
                matched += 1
            elif key == "venue" and value and str(value).lower() in paper.venue.lower():
                matched += 1
            else:
                blob = f"{paper.title} {paper.abstract}".lower()
                if str(value).lower() in blob:
                    matched += 1
        return matched / total if total else 1.0

    @staticmethod
    def _recency_score(year: int | None) -> float:
        if not year:
            return 0.5
        if year >= 2023:
            return 1.0
        if year >= 2020:
            return 0.7
        return 0.4

    def _to_tier(self, score: float) -> str:
        if score >= 0.8:
            return "highly_relevant"
        if score >= self.score_threshold:
            return "partially_relevant"
        return "filtered"

    @staticmethod
    def _build_reason(paper: Paper, query_plan: QueryPlan) -> str:
        return (
            f"Matched query '{query_plan.original_query}' with score components "
            f"from title/abstract overlap, constraints, authority and recency."
        )

    @staticmethod
    def _extract_evidence(paper: Paper, query_plan: QueryPlan) -> str:
        if not paper.abstract:
            return ""
        query_tokens = Ranker._extract_query_tokens(query_plan.original_query)
        sentences = re.split(r"(?<=[.!?])\s+", paper.abstract)
        for sentence in sentences:
            lower = sentence.lower()
            if any(token in lower for token in query_tokens):
                return sentence.strip()
        return paper.abstract[:300].strip()

    @staticmethod
    def _apply_mmr(papers: list[Paper], lambda_: float = 0.7, top_k: int = 20) -> list[Paper]:
        if len(papers) <= 1:
            return papers[:top_k]
        selected: list[Paper] = []
        candidates = papers[:]
        while candidates and len(selected) < top_k:
            best_idx = 0
            best_score = -1.0
            for idx, candidate in enumerate(candidates):
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        Ranker._title_similarity(candidate.title, chosen.title)
                        for chosen in selected
                    )
                mmr = lambda_ * candidate.relevance_score - (1 - lambda_) * redundancy
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            selected.append(candidates.pop(best_idx))
        return selected

    @staticmethod
    def _title_similarity(a: str, b: str) -> float:
        ta = set(re.findall(r"[a-z0-9]{3,}", a.lower()))
        tb = set(re.findall(r"[a-z0-9]{3,}", b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)
