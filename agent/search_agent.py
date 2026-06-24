"""端到端论文搜索 Agent，基于 PaSa 思路改造。"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.prompts import (
    GENERATE_SEARCH_QUERIES,
    GENERATE_SEARCH_QUERIES_PASA,
    SELECT_PAPER,
    SELECT_PAPER_PASA,
)
from agent.query_parser import QueryParser
from agent.ranker import Ranker
from agent.reflector import Reflector
from agent.summarizer import Summarizer
from agent.types import Paper, QueryPlan, SearchOutput
from apis.merge import paper_dedup_key
from apis.retrieval import RetrievalClient
from models.dual_agent import DualAgent

logger = logging.getLogger(__name__)


class PaperSearchAgent:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        search_cfg = config["search"]
        budget_cfg = config["budget"]
        output_cfg = config["output"]

        self.retrieval = RetrievalClient(search_cfg)
        self.dual_agent = DualAgent(config["model"])
        self.query_parser = QueryParser(self.dual_agent)
        self.reflector = Reflector(self.dual_agent, budget_cfg.get("max_api_calls", 30))
        self.ranker = Ranker(score_threshold=search_cfg.get("score_threshold", 0.5))
        self.summarizer = Summarizer(
            self.dual_agent,
            include_clusters=output_cfg.get("include_cluster_summary", True),
            include_graph=output_cfg.get("include_citation_graph", True),
        )

        self.max_queries = search_cfg.get("max_queries", 3)
        self.max_expand_layers = search_cfg.get("max_expand_layers", 1)
        self.expand_seed_papers = search_cfg.get("expand_seed_papers", 3)
        self.expand_seed_min_score = search_cfg.get("expand_seed_min_score", 0.5)
        self.max_llm_calls = budget_cfg.get("max_llm_calls", 50)
        self.max_api_calls = budget_cfg.get("max_api_calls", 30)
        self.use_pasa_prompts = config.get("model", {}).get("use_pasa_prompts", False)

        self._seen_keys: set[str] = set()
        self._candidate_papers: dict[str, Paper] = {}
        self._query_plan: QueryPlan | None = None

    @property
    def api_call_count(self) -> int:
        return self.retrieval.api_call_count

    def run(self, query: str, query_id: str = "0") -> SearchOutput:
        logger.info("Running search for query_id=%s", query_id)
        self.current_query = query
        query_plan = self.query_parser.parse(query)
        self._query_plan = query_plan
        search_queries = self._generate_search_queries(query_plan)
        logger.info("Search queries: %s", search_queries)

        for round_idx in range(self.max_queries):
            for search_query in search_queries:
                if self.api_call_count >= self.max_api_calls:
                    break
                self._search_once(search_query)

            recall_count = len(self._candidate_papers)
            high_score_count = sum(
                1 for p in self._candidate_papers.values() if p.relevance_score >= 0.8
            )
            reflection = self.reflector.reflect(
                query_plan=query_plan,
                recall_count=recall_count,
                high_score_count=high_score_count,
                api_calls_used=self.api_call_count,
                round_idx=round_idx + 1,
                max_rounds=self.max_queries,
            )
            if not reflection["continue"]:
                break
            search_queries = reflection["new_queries"] or query_plan.sub_queries

        if self.max_expand_layers > 0:
            expand_added = self._expand_top_papers()
        else:
            expand_added = 0

        ranked = self.ranker.rank(list(self._candidate_papers.values()), query_plan)
        stats = {
            "api_calls": self.api_call_count,
            "s2_api_calls": self.retrieval.s2.api_call_count if self.retrieval.s2 else 0,
            "openalex_api_calls": (
                self.retrieval.openalex.api_call_count if self.retrieval.openalex else 0
            ),
            "llm_calls": self.dual_agent.llm_call_count,
            "candidate_count": len(self._candidate_papers),
            "result_count": len(ranked),
            "expand_candidates_scored": expand_added,
            "primary_backend": self.retrieval.primary_backend,
        }
        output = self.summarizer.build_output(query_id, query_plan, ranked, stats)
        self.dual_agent.unload_all()
        return output

    def _generate_search_queries(self, query_plan) -> list[str]:
        if self.dual_agent.llm_call_count >= self.max_llm_calls:
            return query_plan.sub_queries[: self.max_queries]

        llm = self.dual_agent.crawler
        if self.use_pasa_prompts:
            response = llm.chat(
                GENERATE_SEARCH_QUERIES_PASA.format(user_query=query_plan.original_query)
            )
        else:
            response = llm.chat(
                GENERATE_SEARCH_QUERIES.format(
                    query=query_plan.original_query,
                    sub_queries=query_plan.sub_queries,
                )
            )
        parsed = re.findall(r"\[Search\](.*?)\[", response, flags=re.DOTALL)
        queries = [q.strip() for q in parsed if q.strip()]
        if queries:
            return queries[: self.max_queries]
        merged = list(dict.fromkeys(query_plan.sub_queries + query_plan.expanded_terms))
        return merged[: self.max_queries]

    def _register_paper(self, raw: dict[str, Any], source: str) -> Paper | None:
        key = paper_dedup_key(raw)
        if key in self._seen_keys:
            return None
        paper = Paper.from_api(raw, source=source)
        if not paper.title:
            return None
        self._seen_keys.add(key)
        store_key = paper.corpus_id if paper.corpus_id else key
        self._candidate_papers[store_key] = paper
        return paper

    def _select_paper_prompt(self, title: str, abstract: str) -> str:
        template = SELECT_PAPER_PASA if self.use_pasa_prompts else SELECT_PAPER
        return template.format(
            user_query=self.current_query,
            title=title,
            abstract=abstract,
        )

    def _search_once(self, query: str) -> None:
        plan = self._query_plan
        raw_papers = self.retrieval.search_papers(
            query,
            constraints=plan.constraints if plan else None,
            intent=plan.intent if plan else "semantic_search",
        )
        prompts = []
        papers = []
        for raw in raw_papers:
            paper = self._register_paper(raw, source=raw.get("source", "search"))
            if paper is None:
                continue
            papers.append(paper)
            prompts.append(self._select_paper_prompt(paper.title, paper.abstract))
        if not prompts:
            return
        scores = self.dual_agent.selector.batch_infer_relevance(prompts)
        for paper, score in zip(papers, scores):
            paper.relevance_score = score

    def _expand_top_papers(self) -> int:
        plan = self._query_plan
        high_score_count = sum(
            1 for p in self._candidate_papers.values() if p.relevance_score >= 0.8
        )
        if not self.retrieval.should_expand_citations(
            candidate_count=len(self._candidate_papers),
            high_score_count=high_score_count,
            intent=plan.intent if plan else "semantic_search",
        ):
            logger.info("Skipping citation expand: recall sufficient")
            return 0

        ranked_seeds = sorted(
            self._candidate_papers.values(),
            key=lambda p: p.relevance_score,
            reverse=True,
        )
        seed_papers = [
            paper for paper in ranked_seeds if paper.relevance_score >= self.expand_seed_min_score
        ][: self.expand_seed_papers]
        if not seed_papers:
            seed_papers = ranked_seeds[: self.expand_seed_papers]

        scored_count = 0
        constraints = plan.constraints if plan else None
        for paper in seed_papers:
            if self.api_call_count >= self.max_api_calls:
                break

            raw_paper = {
                "corpus_id": paper.corpus_id,
                "openalex_id": paper.openalex_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
            }
            refs = self.retrieval.expand_citations_smart(
                raw_paper,
                self.current_query,
                constraints=constraints,
            )
            if not refs:
                continue

            prompts = []
            new_papers = []
            for raw in refs:
                ref = self._register_paper(raw, source=raw.get("source", "expand"))
                if ref is None:
                    continue
                new_papers.append(ref)
                prompts.append(self._select_paper_prompt(ref.title, ref.abstract))
            if not prompts:
                continue
            scores = self.dual_agent.selector.batch_infer_relevance(prompts)
            for ref, score in zip(new_papers, scores):
                ref.relevance_score = score
                scored_count += 1

        logger.info("Smart citation expand scored %s candidates from %s seeds", scored_count, len(seed_papers))
        return scored_count
