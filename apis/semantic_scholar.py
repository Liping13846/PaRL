"""Semantic Scholar API 同步封装，带限速与重试。"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
METADATA_FIELDS = (
    "title,abstract,corpusId,authors,year,venue,"
    "citationCount,referenceCount,externalIds,url"
)


class SemanticScholarClient:
    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("S2_API_KEY", "")
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self._last_request_at = 0.0
        self.api_call_count = 0

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{S2_BASE_URL}/{endpoint.lstrip('/')}"
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self._headers(),
                    params=params,
                    json=json_body,
                    timeout=20,
                )
                self._last_request_at = time.time()
                self.api_call_count += 1
                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("S2 rate limited, retry in %ss", wait)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    logger.warning("S2 API failed for %s: %s", endpoint, exc)
                    return {}
                time.sleep(2 ** attempt)
        return {}

    @staticmethod
    def _normalize_paper(raw: dict[str, Any]) -> dict[str, Any]:
        external_ids = raw.get("externalIds") or {}
        return {
            "corpus_id": str(raw.get("corpusId", "")),
            "title": raw.get("title", ""),
            "abstract": raw.get("abstract", "") or "",
            "authors": [
                a.get("name", "") for a in (raw.get("authors") or []) if a.get("name")
            ],
            "year": raw.get("year"),
            "venue": raw.get("venue", "") or "",
            "citation_count": raw.get("citationCount", 0) or 0,
            "arxiv_id": external_ids.get("ArXiv", ""),
            "url": raw.get("url", ""),
            "source": "semantic_scholar",
        }

    def _papers_from_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        papers = []
        for item in (data.get("data") or []):
            if not item.get("corpusId") or not item.get("title"):
                continue
            normalized = self._normalize_paper(item)
            if normalized["abstract"]:
                papers.append(normalized)
        return papers

    def search_papers(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        data = self._request(
            "GET",
            "paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": METADATA_FIELDS,
            },
        )
        if not data:
            return []
        papers = []
        for item in (data.get("data") or []):
            if not item.get("corpusId") or not item.get("title"):
                continue
            normalized = self._normalize_paper(item)
            if normalized["abstract"]:
                papers.append(normalized)
        return papers

    def get_paper(self, corpus_id: str) -> dict[str, Any] | None:
        data = self._request(
            "GET",
            f"paper/CorpusId:{corpus_id}",
            params={"fields": METADATA_FIELDS},
        )
        if not data.get("corpusId"):
            return None
        return self._normalize_paper(data)

    def get_paper_by_doi(self, doi: str) -> dict[str, Any] | None:
        doi = doi.replace("https://doi.org/", "").strip()
        if not doi:
            return None
        data = self._request(
            "GET",
            f"paper/DOI:{doi}",
            params={"fields": METADATA_FIELDS},
        )
        if not data.get("corpusId"):
            return None
        return self._normalize_paper(data)

    def get_paper_by_arxiv(self, arxiv_id: str) -> dict[str, Any] | None:
        arxiv_id = arxiv_id.split("v")[0].strip()
        if not arxiv_id:
            return None
        data = self._request(
            "GET",
            f"paper/ARXIV:{arxiv_id}",
            params={"fields": METADATA_FIELDS},
        )
        if not data.get("corpusId"):
            return None
        return self._normalize_paper(data)

    def get_references(self, corpus_id: str, limit: int = 20) -> list[dict[str, Any]]:
        data = self._request(
            "GET",
            f"paper/CorpusId:{corpus_id}/references",
            params={"fields": METADATA_FIELDS, "limit": limit},
        )
        papers = []
        for item in (data.get("data") or []):
            cited = item.get("citedPaper") or {}
            if not cited.get("corpusId") or not cited.get("title"):
                continue
            normalized = self._normalize_paper(cited)
            if normalized["abstract"]:
                papers.append(normalized)
        return papers

    def get_citations(self, corpus_id: str, limit: int = 20) -> list[dict[str, Any]]:
        data = self._request(
            "GET",
            f"paper/CorpusId:{corpus_id}/citations",
            params={"fields": METADATA_FIELDS, "limit": limit},
        )
        papers = []
        for item in (data.get("data") or []):
            citing = item.get("citingPaper") or {}
            if not citing.get("corpusId") or not citing.get("title"):
                continue
            normalized = self._normalize_paper(citing)
            if normalized["abstract"]:
                papers.append(normalized)
        return papers
