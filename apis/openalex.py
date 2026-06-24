"""OpenAlex API 同步封装，作为 Semantic Scholar 的辅助检索源。"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any
from urllib.parse import quote

import requests

from apis.arxiv_resolve import extract_arxiv_id_from_work

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org"

WORK_SELECT = (
    "id,display_name,abstract_inverted_index,authorships,"
    "publication_year,primary_location,cited_by_count,ids,locations,"
    "open_access,best_oa_location"
)

VENUE_SEARCH_TERMS: dict[str, str] = {
    "ACL": "Association for Computational Linguistics",
    "EMNLP": "Empirical Methods in Natural Language Processing",
    "NAACL": "North American Chapter of the Association for Computational Linguistics",
    "NeurIPS": "Neural Information Processing Systems",
    "ICLR": "International Conference on Learning Representations",
    "CVPR": "Computer Vision and Pattern Recognition",
    "ICCV": "International Conference on Computer Vision",
    "ICML": "International Conference on Machine Learning",
}


class OpenAlexClient:
    def __init__(
        self,
        mailto: str | None = None,
        rate_limit: float = 0.2,
        max_retries: int = 3,
    ) -> None:
        self.mailto = mailto or os.getenv("OPENALEX_MAILTO", "")
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self._last_request_at = 0.0
        self.api_call_count = 0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{OPENALEX_BASE_URL}/{endpoint.lstrip('/')}"
        query = dict(params or {})
        if self.mailto:
            query["mailto"] = self.mailto

        for attempt in range(self.max_retries):
            self._throttle()
            try:
                response = requests.get(url, params=query, timeout=20)
                self._last_request_at = time.time()
                self.api_call_count += 1
                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("OpenAlex rate limited, retry in %ss", wait)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"OpenAlex API failed: {endpoint}") from exc
                time.sleep(2 ** attempt)
        return {}

    @staticmethod
    def _extract_openalex_id(entity_id: str) -> str:
        if not entity_id:
            return ""
        match = re.search(r"([WS]\d+)", entity_id)
        return match.group(1) if match else entity_id

    @staticmethod
    def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
        if not inverted_index:
            return ""
        tokens: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                tokens.append((pos, word))
        tokens.sort(key=lambda item: item[0])
        return " ".join(word for _, word in tokens).strip()

    @classmethod
    def _normalize_work(cls, raw: dict[str, Any]) -> dict[str, Any]:
        openalex_id = cls._extract_openalex_id(raw.get("id", ""))
        ids = raw.get("ids") or {}
        doi = (ids.get("doi") or "").replace("https://doi.org/", "")
        venue = ""
        primary = raw.get("primary_location") or {}
        source = primary.get("source") or {}
        if source.get("display_name"):
            venue = source["display_name"]

        authors = []
        for authorship in raw.get("authorships") or []:
            author = authorship.get("author") or {}
            if author.get("display_name"):
                authors.append(author["display_name"])

        arxiv_id = extract_arxiv_id_from_work(raw)

        return {
            "corpus_id": f"openalex:{openalex_id}" if openalex_id else "",
            "openalex_id": openalex_id,
            "title": raw.get("display_name", "") or raw.get("title", ""),
            "abstract": cls._reconstruct_abstract(raw.get("abstract_inverted_index")),
            "authors": authors,
            "year": raw.get("publication_year"),
            "venue": venue,
            "citation_count": raw.get("cited_by_count", 0) or 0,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "url": raw.get("id", ""),
            "source": "openalex",
        }

    def _collect_works(self, data: dict[str, Any], *, require_abstract: bool = True) -> list[dict[str, Any]]:
        papers = []
        for item in data.get("results", []):
            normalized = self._normalize_work(item)
            if not normalized["title"]:
                continue
            if require_abstract and not normalized["abstract"]:
                continue
            papers.append(normalized)
        return papers

    def search_works(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "per_page": limit,
            "select": WORK_SELECT,
        }
        if query.strip():
            params["search"] = query.strip()
        data = self._request("works", params=params)
        return self._collect_works(data)

    def search_works_filtered(
        self,
        query: str,
        filters: list[str],
        limit: int = 10,
        *,
        require_abstract: bool = True,
    ) -> list[dict[str, Any]]:
        if not filters:
            return self.search_works(query, limit=limit)

        params: dict[str, Any] = {
            "filter": ",".join(filters),
            "per_page": limit,
            "select": WORK_SELECT,
        }
        if query.strip():
            params["search"] = query.strip()
        try:
            data = self._request("works", params=params)
        except RuntimeError as exc:
            logger.warning("OpenAlex filtered search failed: %s", exc)
            return self.search_works(query, limit=limit) if query.strip() else []
        return self._collect_works(data, require_abstract=require_abstract)

    def find_work_id_by_title(self, title: str, limit: int = 5) -> str:
        """按标题搜索并返回最匹配的 OpenAlex work id（W...）。"""
        title = title.strip()
        if not title:
            return ""

        data = self._request(
            "works",
            params={
                "search": title,
                "per_page": limit,
                "select": "id,display_name",
            },
        )
        target = self._normalize_title_for_match(title)
        best_id = ""
        best_score = 0.0
        for item in data.get("results", []):
            candidate_title = item.get("display_name") or item.get("title") or ""
            score = self._title_match_score(target, self._normalize_title_for_match(candidate_title))
            work_id = self._extract_openalex_id(item.get("id", ""))
            if score > best_score and work_id:
                best_score = score
                best_id = work_id

        if best_score >= 0.55:
            return best_id
        results = data.get("results") or []
        if results:
            return self._extract_openalex_id(results[0].get("id", ""))
        return ""

    @staticmethod
    def _normalize_title_for_match(title: str) -> str:
        text = title.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    @classmethod
    def _title_match_score(cls, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        if left in right or right in left:
            return 0.85
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        return overlap / max(len(left_tokens), len(right_tokens))

    def find_source_ids_for_venue(
        self,
        venue: str,
        year: int | None = None,
        limit: int = 3,
    ) -> list[str]:
        """将会议/期刊简称解析为 OpenAlex source id 列表。"""
        venue = venue.strip()
        if not venue:
            return []

        search_terms = []
        if year:
            search_terms.append(f"{venue} {year}")
        search_terms.append(VENUE_SEARCH_TERMS.get(venue.upper(), venue))

        venue_key = venue.lower()
        ranked: list[tuple[float, str]] = []
        seen_ids: set[str] = set()

        for search_term in search_terms:
            data = self._request(
                "sources",
                params={
                    "search": search_term,
                    "per_page": 20,
                    "select": "id,display_name,type",
                },
            )
            for item in data.get("results", []):
                name = (item.get("display_name") or "").lower()
                source_id = self._extract_openalex_id(item.get("id", ""))
                if not source_id or source_id in seen_ids:
                    continue
                seen_ids.add(source_id)
                score = self._venue_source_score(venue_key, name, item.get("type") or "", year)
                if score > 0:
                    ranked.append((score, source_id))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        ids: list[str] = []
        for _, source_id in ranked:
            if source_id not in ids:
                ids.append(source_id)
            if len(ids) >= limit:
                break
        return ids

    @staticmethod
    def _venue_source_score(
        venue_key: str,
        display_name: str,
        source_type: str,
        year: int | None = None,
    ) -> float:
        score = 0.0
        if venue_key in display_name:
            score += 2.0
        if source_type == "conference":
            score += 1.0
        if "proceedings" in display_name and venue_key in display_name:
            score += 0.5
        if venue_key == "acl" and "computational linguistics" in display_name:
            score += 1.5
        if year and str(year) in display_name and venue_key in display_name:
            score += 2.5
        return score

    def build_venue_filter(self, venue: str, year: int | None = None) -> str | None:
        source_ids = self.find_source_ids_for_venue(venue, year=year)
        if not source_ids:
            return None
        if len(source_ids) == 1:
            return f"primary_location.source.id:{source_ids[0]}"
        return f"primary_location.source.id:{'|'.join(source_ids)}"

    def get_work(self, openalex_id: str) -> dict[str, Any] | None:
        work_id = self._extract_openalex_id(openalex_id)
        if not work_id:
            return None
        data = self._request(
            f"works/{work_id}",
            params={
                "select": f"{WORK_SELECT},referenced_works",
            },
        )
        if not data.get("id"):
            return None
        return self._normalize_work(data)

    def get_references(self, openalex_id: str, limit: int = 20) -> list[dict[str, Any]]:
        work = self._request(
            f"works/{self._extract_openalex_id(openalex_id)}",
            params={"select": "referenced_works"},
        )
        ref_urls = work.get("referenced_works") or []
        if not ref_urls:
            return []

        ref_ids = [self._extract_openalex_id(url) for url in ref_urls[:limit]]
        ref_ids = [rid for rid in ref_ids if rid]
        if not ref_ids:
            return []

        filter_value = "|".join(ref_ids)
        data = self._request(
            "works",
            params={
                "filter": f"openalex:{filter_value}",
                "per_page": limit,
                "select": WORK_SELECT,
            },
        )
        papers = self._collect_works(data)
        for paper in papers:
            paper["source"] = "openalex_reference"
        return papers

    def get_cited_by(self, openalex_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """返回引用 seed 论文的后续工作（forward citations）。"""
        work_id = self._extract_openalex_id(openalex_id)
        if not work_id:
            return []
        data = self._request(
            "works",
            params={
                "filter": f"cites:{work_id}",
                "per_page": limit,
                "select": WORK_SELECT,
                "sort": "cited_by_count:desc",
            },
        )
        papers = self._collect_works(data)
        for paper in papers:
            paper["source"] = "openalex_cited_by"
        return papers
