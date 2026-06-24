"""简单 Web 前端后端服务。"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from configs.loader import load_config
from run_agent import run_lite_retrieval, run_single_query
from agent.search_agent import PaperSearchAgent
from apis.paper_display import fetch_arxiv_figures
from apis.paper_content import fetch_paper_conclusion, translate_text

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
NO_CACHE = "no-store, no-cache, must-revalidate"


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = NO_CACHE
        response.headers["Pragma"] = "no-cache"
        return response


app = FastAPI(title="PaRL", description="Paper Agent for Research Literature", version="0.1.0")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    mode: str = Field(default="lite", pattern="^(lite|full)$")
    config: str = Field(default="dev")


class SearchResponse(BaseModel):
    ok: bool
    data: dict


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": NO_CACHE, "Pragma": "no-cache"},
    )


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "service": "parl", "name": "PaRL"}


@app.get("/api/paper-figures")
async def paper_figures(arxiv_id: str) -> dict:
    arxiv_id = arxiv_id.strip().split("v")[0]
    if not re.fullmatch(r"\d{4}\.\d{4,5}", arxiv_id):
        raise HTTPException(status_code=400, detail="Invalid arXiv id")
    figures = fetch_arxiv_figures(arxiv_id, max_figures=2)
    return {"arxiv_id": arxiv_id, "figures": figures}


class ConclusionRequest(BaseModel):
    arxiv_id: str = ""
    abstract: str = ""
    openalex_id: str = ""
    doi: str = ""


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


@app.get("/api/paper-conclusion")
async def paper_conclusion_get(
    arxiv_id: str = "",
    abstract: str = "",
    openalex_id: str = "",
    doi: str = "",
) -> dict:
    return _build_conclusion_payload(arxiv_id, abstract, openalex_id, doi)


@app.post("/api/paper-conclusion")
async def paper_conclusion_post(request: ConclusionRequest) -> dict:
    return _build_conclusion_payload(
        request.arxiv_id,
        request.abstract,
        request.openalex_id,
        request.doi,
    )


def _build_conclusion_payload(
    arxiv_id: str,
    abstract: str,
    openalex_id: str = "",
    doi: str = "",
) -> dict:
    arxiv_id = arxiv_id.strip().split("v")[0]
    if arxiv_id and not re.fullmatch(r"\d{4}\.\d{4,5}", arxiv_id):
        raise HTTPException(status_code=400, detail="Invalid arXiv id")
    if not arxiv_id and not abstract.strip() and not openalex_id and not doi:
        raise HTTPException(status_code=400, detail="Need arxiv_id, openalex_id, doi, or abstract")
    return fetch_paper_conclusion(
        arxiv_id,
        abstract_fallback=abstract.strip(),
        openalex_id=openalex_id.strip(),
        doi=doi.strip(),
    )


@app.post("/api/translate")
async def translate(request: TranslateRequest) -> dict:
    return translate_text(request.text.strip())


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
