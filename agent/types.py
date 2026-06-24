from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryPlan:
    original_query: str
    intent: str = "semantic_search"
    constraints: dict[str, Any] = field(default_factory=dict)
    sub_queries: list[str] = field(default_factory=list)
    expanded_terms: list[str] = field(default_factory=list)


@dataclass
class Paper:
    corpus_id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    citation_count: int = 0
    arxiv_id: str = ""
    doi: str = ""
    openalex_id: str = ""
    url: str = ""
    source: str = "search"
    relevance_score: float = 0.0
    tier: str = "filtered"
    relevance_reason: str = ""
    evidence: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any], source: str = "search") -> Paper:
        return cls(
            corpus_id=data.get("corpus_id", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            authors=data.get("authors", []),
            year=data.get("year"),
            venue=data.get("venue", ""),
            citation_count=data.get("citation_count", 0),
            arxiv_id=data.get("arxiv_id", ""),
            doi=data.get("doi", ""),
            openalex_id=data.get("openalex_id", ""),
            url=data.get("url", ""),
            source=source,
        )

    @property
    def is_s2_corpus_id(self) -> bool:
        return self.corpus_id.isdigit()

    def to_result_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.corpus_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "tier": self.tier,
            "relevance_score": round(self.relevance_score, 4),
            "relevance_reason": self.relevance_reason,
            "markdown_evidence": self.evidence or self._default_evidence(),
            "abstract": self.abstract,
            "citation_count": self.citation_count,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "openalex_id": self.openalex_id,
            "url": self.url,
            "source": self.source,
        }

    def _default_evidence(self) -> str:
        if not self.abstract:
            return ""
        return self.abstract[:300].strip()


@dataclass
class SearchOutput:
    query_id: str
    query: str
    query_plan: QueryPlan
    results: list[Paper]
    clusters: list[dict[str, Any]] = field(default_factory=list)
    citation_graph: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_astabench_format(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "results": [
                {
                    "paper_id": p.corpus_id,
                    "markdown_evidence": p.to_result_dict()["markdown_evidence"],
                }
                for p in self.results
                if p.tier in {"highly_relevant", "partially_relevant"} and p.is_s2_corpus_id
            ],
        }
