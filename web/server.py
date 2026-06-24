"""简单 Web 前端后端服务。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from configs.loader import load_config
from run_agent import run_lite_retrieval, run_single_query
from agent.search_agent import PaperSearchAgent

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="PaRL", description="Paper Agent for Research Literature", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    mode: str = Field(default="lite", pattern="^(lite|full)$")
    config: str = Field(default="dev")


class SearchResponse(BaseModel):
    ok: bool
    data: dict


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "service": "parl", "name": "PaRL"}


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        config = load_config(request.config)
        if request.mode == "lite":
            payload = run_lite_retrieval(query, config)
        else:
            agent = PaperSearchAgent(config)
            payload = run_single_query(agent, query, query_id="web", config=config)
        return SearchResponse(ok=True, data=payload)
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
