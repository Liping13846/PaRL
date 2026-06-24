#!/usr/bin/env python3
"""PaRL 命令行入口 — Paper Agent for Research Literature。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.search_agent import PaperSearchAgent
from configs.loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_agent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaRL — Paper Agent for Research Literature")
    parser.add_argument("--config", default="dev", help="Config name: dev | pasa | prod")
    parser.add_argument("--query", type=str, help="Single query string")
    parser.add_argument("--query-id", default="0", help="Query id for output")
    parser.add_argument("--input-file", type=str, help="JSONL input file with question field")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument(
        "--mode",
        choices=["full", "lite"],
        default="full",
        help="lite: skip LLM, only test retrieval pipeline",
    )
    return parser.parse_args()


def save_output(output_dir: Path, query_id: str, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{query_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def run_single_query(agent: PaperSearchAgent, query: str, query_id: str, config: dict) -> dict:
    result = agent.run(query=query, query_id=query_id)
    payload = {
        "astabench": result.to_astabench_format(),
        "full_output": {
            "query_id": result.query_id,
            "query": result.query,
            "query_plan": {
                "intent": result.query_plan.intent,
                "constraints": result.query_plan.constraints,
                "sub_queries": result.query_plan.sub_queries,
                "expanded_terms": result.query_plan.expanded_terms,
            },
            "results": [p.to_result_dict() for p in result.results],
            "clusters": result.clusters,
            "citation_graph": result.citation_graph,
            "stats": result.stats,
        },
    }
    return payload


def run_lite_retrieval(query: str, config: dict) -> dict:
    """不加载 LLM，测试 OpenAlex 主检索 + S2 辅助链路。"""
    from apis.merge import paper_dedup_key
    from apis.retrieval import RetrievalClient
    from agent.query_parser import QueryParser
    from agent.ranker import Ranker
    from agent.types import Paper

    search_cfg = config["search"]
    retrieval = RetrievalClient(search_cfg)
    query_plan = QueryParser._fallback_parse(query)
    papers: list[Paper] = []
    seen: set[str] = set()
    for sub_query in query_plan.sub_queries:
        for raw in retrieval.search_papers(
            sub_query,
            constraints=query_plan.constraints,
            intent=query_plan.intent,
        ):
            key = paper_dedup_key(raw)
            if key in seen:
                continue
            seen.add(key)
            papers.append(Paper.from_api(raw, source=raw.get("source", "search")))

    ranker = Ranker(score_threshold=search_cfg["score_threshold"])
    ranked = ranker.rank(
        papers,
        query_plan,
        min_return=search_cfg.get("min_results", 8),
    )
    return {
        "mode": "lite",
        "query": query,
        "query_plan": query_plan.__dict__,
        "results": [p.to_result_dict() for p in ranked],
        "stats": {
            "api_calls": retrieval.api_call_count,
            "s2_api_calls": retrieval.s2.api_call_count if retrieval.s2 else 0,
            "openalex_api_calls": retrieval.openalex.api_call_count if retrieval.openalex else 0,
            "primary_backend": retrieval.primary_backend,
            "candidate_count": len(papers),
            "result_count": len(ranked),
        },
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir or config["output"]["results_dir"])

    if args.mode == "lite":
        if not args.query:
            raise SystemExit("--query is required in lite mode")
        payload = run_lite_retrieval(args.query, config)
        out = save_output(output_dir, args.query_id, payload)
        logger.info("Lite run finished. Saved to %s", out)
        return

    if args.query:
        agent = PaperSearchAgent(config)
        payload = run_single_query(agent, args.query, args.query_id, config)
        out = save_output(output_dir, args.query_id, payload)
        logger.info("Saved result to %s", out)
        return

    if args.input_file:
        input_path = Path(args.input_file)
        with open(input_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                query = data.get("question") or data.get("query")
                if not query:
                    continue
                query_id = str(data.get("query_id", idx))
                agent = PaperSearchAgent(config)
                payload = run_single_query(agent, query, query_id, config)
                out = save_output(output_dir, query_id, payload)
                logger.info("Processed %s -> %s", query_id, out)
        return

    raise SystemExit("Please provide --query or --input-file")


if __name__ == "__main__":
    main()
