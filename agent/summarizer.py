from __future__ import annotations

from typing import Any

from agent.prompts import CLUSTER_SUMMARY
from agent.types import Paper, QueryPlan, SearchOutput
from models.dual_agent import DualAgent


class Summarizer:
    def __init__(
        self,
        dual_agent: DualAgent,
        include_clusters: bool = True,
        include_graph: bool = True,
    ) -> None:
        self.dual_agent = dual_agent
        self.include_clusters = include_clusters
        self.include_graph = include_graph

    def build_output(
        self,
        query_id: str,
        query_plan: QueryPlan,
        ranked_papers: list[Paper],
        stats: dict[str, Any],
    ) -> SearchOutput:
        clusters = self._build_clusters(query_plan, ranked_papers) if self.include_clusters else []
        graph = self._build_citation_graph(ranked_papers) if self.include_graph else {}
        return SearchOutput(
            query_id=query_id,
            query=query_plan.original_query,
            query_plan=query_plan,
            results=ranked_papers,
            clusters=clusters,
            citation_graph=graph,
            stats=stats,
        )

    def _build_clusters(self, query_plan: QueryPlan, papers: list[Paper]) -> list[dict[str, Any]]:
        if not papers:
            return []
        paper_lines = "\n".join(
            f"- {p.corpus_id}: {p.title}" for p in papers[:10]
        )
        llm = self.dual_agent.crawler
        parsed = llm.parse_json(
            CLUSTER_SUMMARY.format(query=query_plan.original_query, papers=paper_lines)
        )
        clusters = parsed.get("clusters", []) if parsed else []
        if clusters:
            return clusters
        return [{
            "name": "Core Results",
            "summary": f"Top {len(papers)} papers most relevant to the query.",
            "paper_ids": [p.corpus_id for p in papers[:5]],
        }]

    @staticmethod
    def _build_citation_graph(papers: list[Paper]) -> dict[str, Any]:
        nodes = [
            {
                "id": p.corpus_id,
                "label": p.title[:80],
                "score": p.relevance_score,
            }
            for p in papers[:10]
        ]
        edges = []
        if len(nodes) > 1:
            for idx in range(len(nodes) - 1):
                edges.append({
                    "source": nodes[idx]["id"],
                    "target": nodes[idx + 1]["id"],
                    "relation": "co_retrieved",
                })
        return {"nodes": nodes, "edges": edges}
