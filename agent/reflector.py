from __future__ import annotations

from typing import Any

from agent.prompts import REFLECT_SEARCH
from agent.types import QueryPlan
from models.dual_agent import DualAgent


class Reflector:
    def __init__(self, dual_agent: DualAgent, max_api_calls: int) -> None:
        self.dual_agent = dual_agent
        self.max_api_calls = max_api_calls

    def reflect(
        self,
        query_plan: QueryPlan,
        recall_count: int,
        high_score_count: int,
        api_calls_used: int,
        round_idx: int,
        max_rounds: int,
    ) -> dict[str, Any]:
        api_budget = max(self.max_api_calls - api_calls_used, 0)
        if round_idx >= max_rounds or api_budget <= 0:
            return {"continue": False, "reason": "budget_or_round_limit", "new_queries": []}
        if high_score_count >= 3 and recall_count >= 8:
            return {"continue": False, "reason": "enough_results", "new_queries": []}

        llm = self.dual_agent.crawler
        result = llm.parse_json(
            REFLECT_SEARCH.format(
                query=query_plan.original_query,
                sub_queries=query_plan.sub_queries,
                recall_count=recall_count,
                high_score_count=high_score_count,
                api_budget=api_budget,
            )
        )
        if result:
            return {
                "continue": bool(result.get("continue", False)),
                "reason": result.get("reason", ""),
                "new_queries": [
                    q.strip() for q in result.get("new_queries", []) if isinstance(q, str) and q.strip()
                ],
            }
        return {
            "continue": round_idx < max_rounds and api_budget > 0,
            "reason": "fallback_continue",
            "new_queries": query_plan.sub_queries,
        }
