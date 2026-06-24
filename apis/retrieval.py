"""统一检索门面：OpenAlex 为主，Semantic Scholar 为可选辅助。"""

from __future__ import annotations

import logging
import os
from typing import Any

from apis.citation_expand import (
    merge_citation_neighbors,
    select_citation_candidates,
    should_trigger_citation_expand,
)
from apis.merge import merge_paper_records
from apis.metadata import (
    build_metadata_search_text,
    build_openalex_filters,
    should_use_metadata_search,
)
from apis.openalex import OpenAlexClient
from apis.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


class RetrievalClient:
    def __init__(self, search_cfg: dict[str, Any]) -> None:
        self.search_cfg = search_cfg
        self.primary_backend = search_cfg.get("primary_backend", "openalex")
        self.enable_s2 = search_cfg.get("enable_s2", True)
        self.enable_openalex = search_cfg.get("enable_openalex", True)
        self.enable_metadata_search = search_cfg.get("enable_metadata_search", True)
        self.resolve_s2_corpus_id = search_cfg.get("resolve_s2_corpus_id", True)
        self.resolve_s2_max = search_cfg.get("resolve_s2_max", 3)

        has_s2_key = bool(os.getenv("S2_API_KEY", "").strip())
        if not has_s2_key and self.enable_s2:
            logger.warning(
                "S2_API_KEY not set. Semantic Scholar will be used sparingly as fallback only."
            )

        self.s2 = (
            SemanticScholarClient(rate_limit=search_cfg.get("api_rate_limit", 1.0))
            if self.enable_s2
            else None
        )
        self.openalex = (
            OpenAlexClient(rate_limit=search_cfg.get("openalex_rate_limit", 0.2))
            if self.enable_openalex
            else None
        )

        self.openalex_per_query = search_cfg.get("openalex_per_query", 8)
        self.papers_per_query = search_cfg.get("papers_per_query", 10)
        self.papers_per_expand = search_cfg.get("papers_per_expand", 10)
        self.enable_smart_citation_expand = search_cfg.get("enable_smart_citation_expand", True)
        self.expand_fetch_per_seed = search_cfg.get("expand_fetch_per_seed", 25)
        self.expand_select_per_seed = search_cfg.get("expand_select_per_seed", 8)
        self.expand_include_cited_by = search_cfg.get("expand_include_cited_by", True)
        self.expand_cited_by_limit = search_cfg.get("expand_cited_by_limit", 12)
        self.expand_prefilter_min_score = search_cfg.get("expand_prefilter_min_score", 0.15)
        self.expand_min_recall = search_cfg.get("expand_min_recall", 8)
        self.expand_min_high_score = search_cfg.get("expand_min_high_score", 2)
        self._cited_work_cache: dict[str, str] = {}

    @property
    def api_call_count(self) -> int:
        total = 0
        if self.s2:
            total += self.s2.api_call_count
        if self.openalex:
            total += self.openalex.api_call_count
        return total

    def search_papers(
        self,
        query: str,
        *,
        constraints: dict[str, Any] | None = None,
        intent: str = "semantic_search",
    ) -> list[dict[str, Any]]:
        constraints = constraints or {}
        if (
            self.openalex
            and should_use_metadata_search(
                intent,
                constraints,
                enabled=self.enable_metadata_search,
            )
        ):
            metadata_results = self._search_with_metadata(query, constraints)
            if metadata_results:
                return self._finalize_results(metadata_results)

        openalex_results: list[dict[str, Any]] = []
        s2_results: list[dict[str, Any]] = []

        if self.primary_backend == "openalex" and self.openalex:
            openalex_results = self.openalex.search_works(query, limit=self.openalex_per_query)
            if self.s2 and len(openalex_results) < self.papers_per_query // 2:
                s2_results = self.s2.search_papers(query, limit=self.papers_per_query)
        elif self.s2:
            s2_results = self.s2.search_papers(query, limit=self.papers_per_query)
            if self.openalex:
                openalex_results = self.openalex.search_works(query, limit=self.openalex_per_query)
        elif self.openalex:
            openalex_results = self.openalex.search_works(query, limit=self.openalex_per_query)

        merged = merge_paper_records(s2_results, openalex_results)
        return self._finalize_results(merged)

    def _search_with_metadata(
        self,
        query: str,
        constraints: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not self.openalex:
            return []

        filters = build_openalex_filters(constraints)
        year = constraints.get("year")
        try:
            year_int = int(year) if year not in (None, "") else None
        except (TypeError, ValueError):
            year_int = None

        venue = (constraints.get("venue") or "").strip()
        venue_filter: str | None = None
        if venue and self.openalex:
            venue_filter = self.openalex.build_venue_filter(venue, year=year_int)
            if venue_filter:
                filters.append(venue_filter)
            else:
                logger.warning("Could not resolve venue constraint to OpenAlex source id: %s", venue)

        cites = (constraints.get("cites") or "").strip()
        if cites:
            cited_work_id = self._resolve_cited_work_id(cites)
            if cited_work_id:
                filters.append(f"cites:{cited_work_id}")
            else:
                logger.warning("Could not resolve cites constraint to OpenAlex work id: %s", cites)

        search_text = build_metadata_search_text(query, constraints)
        logger.info(
            "Metadata search: search=%r filters=%s",
            search_text,
            filters,
        )
        results = self.openalex.search_works_filtered(
            search_text,
            filters,
            limit=self.openalex_per_query,
        )
        if results:
            for paper in results:
                paper["source"] = "openalex_metadata"
            return results

        if venue_filter and len(filters) > 1:
            relaxed_filters = [item for item in filters if item != venue_filter]
            logger.info("Metadata search retry without venue filter: %s", relaxed_filters)
            results = self.openalex.search_works_filtered(
                search_text,
                relaxed_filters,
                limit=self.openalex_per_query,
            )
            if results:
                for paper in results:
                    paper["source"] = "openalex_metadata"
                return results

        if filters:
            logger.info("Metadata search returned 0 results, retrying without filters")
            return self.openalex.search_works(search_text, limit=self.openalex_per_query)
        return []

    def _resolve_cited_work_id(self, cites: str) -> str:
        if cites in self._cited_work_cache:
            return self._cited_work_cache[cites]
        if not self.openalex:
            return ""
        work_id = self.openalex.find_work_id_by_title(cites)
        self._cited_work_cache[cites] = work_id
        return work_id

    def _finalize_results(self, merged: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.resolve_s2_corpus_id and self.s2:
            resolved: list[dict[str, Any]] = []
            resolve_budget = self.resolve_s2_max
            for paper in merged:
                if resolve_budget <= 0 or str(paper.get("corpus_id", "")).isdigit():
                    resolved.append(paper)
                    continue
                resolved.append(self._maybe_resolve_s2_corpus_id(paper))
                resolve_budget -= 1
            merged = resolved
        return merged

    def get_references(self, paper: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        corpus_id = str(paper.get("corpus_id", ""))
        openalex_id = str(paper.get("openalex_id", ""))
        fetch_limit = limit or self.papers_per_expand

        if corpus_id.isdigit() and self.s2:
            refs = self.s2.get_references(corpus_id, limit=fetch_limit)
            for item in refs:
                item["source"] = "semantic_scholar_reference"
        elif openalex_id and self.openalex:
            refs = self.openalex.get_references(openalex_id, limit=fetch_limit)

        if self.resolve_s2_corpus_id and self.s2:
            refs = [self._maybe_resolve_s2_corpus_id(item) for item in refs]
        return refs

    def get_cited_by(self, paper: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
        cited_by: list[dict[str, Any]] = []
        corpus_id = str(paper.get("corpus_id", ""))
        openalex_id = str(paper.get("openalex_id", ""))
        fetch_limit = limit or self.expand_cited_by_limit

        if corpus_id.isdigit() and self.s2:
            cited_by = self.s2.get_citations(corpus_id, limit=fetch_limit)
            for item in cited_by:
                item["source"] = "semantic_scholar_cited_by"
        elif openalex_id and self.openalex:
            cited_by = self.openalex.get_cited_by(openalex_id, limit=fetch_limit)

        if self.resolve_s2_corpus_id and self.s2:
            cited_by = [self._maybe_resolve_s2_corpus_id(item) for item in cited_by]
        return cited_by

    def expand_citations_smart(
        self,
        paper: dict[str, Any],
        query: str,
        *,
        constraints: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """从 seed 论文拉取引文邻居，相似度预筛后返回候选。"""
        if not self.enable_smart_citation_expand:
            return self.get_references(paper)

        references = self.get_references(paper, limit=self.expand_fetch_per_seed)
        cited_by: list[dict[str, Any]] = []
        if self.expand_include_cited_by:
            cited_by = self.get_cited_by(paper, limit=self.expand_cited_by_limit)

        neighbors = merge_citation_neighbors(references, cited_by)
        selected = select_citation_candidates(
            neighbors,
            query,
            constraints,
            top_k=self.expand_select_per_seed,
            min_score=self.expand_prefilter_min_score,
        )
        for item in selected:
            direction = item.get("expand_direction", "reference")
            item["source"] = f"expand_{direction}"
        return self._finalize_results(selected)

    def should_expand_citations(
        self,
        *,
        candidate_count: int,
        high_score_count: int,
        intent: str = "semantic_search",
    ) -> bool:
        if not self.enable_smart_citation_expand:
            return True
        return should_trigger_citation_expand(
            candidate_count=candidate_count,
            high_score_count=high_score_count,
            intent=intent,
            min_recall=self.expand_min_recall,
            min_high_score=self.expand_min_high_score,
        )

    def _maybe_resolve_s2_corpus_id(self, paper: dict[str, Any]) -> dict[str, Any]:
        if not self.s2:
            return paper
        corpus_id = str(paper.get("corpus_id", ""))
        if corpus_id.isdigit():
            return paper

        resolved = None
        doi = (paper.get("doi") or "").strip()
        arxiv_id = (paper.get("arxiv_id") or "").strip()
        if doi:
            resolved = self.s2.get_paper_by_doi(doi)
        elif arxiv_id:
            resolved = self.s2.get_paper_by_arxiv(arxiv_id)

        if resolved and resolved.get("corpus_id"):
            merged = dict(paper)
            merged.update(resolved)
            merged["openalex_id"] = paper.get("openalex_id", "")
            merged["source"] = paper.get("source", "openalex")
            return merged
        return paper
