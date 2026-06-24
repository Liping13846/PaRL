"""Resolve arXiv IDs from paper metadata for full-text section fetching."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import requests

logger = logging.getLogger(__name__)

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


def normalize_arxiv_id(arxiv_id: str) -> str:
    return (arxiv_id or "").split("v")[0].strip()


def extract_arxiv_id_from_work(raw: dict[str, Any]) -> str:
    """Extract arXiv id from an OpenAlex work payload."""
    ids = raw.get("ids") or {}
    for value in ids.values():
        if not value:
            continue
        match = _ARXIV_RE.search(str(value))
        if match:
            return match.group(1)
        if "arxiv" in str(value).lower():
            plain = _ARXIV_ID_RE.search(str(value))
            if plain:
                return plain.group(1)

    for loc in raw.get("locations") or []:
        for key in ("landing_page_url", "pdf_url"):
            url = loc.get(key) or ""
            match = _ARXIV_RE.search(url)
            if match:
                return match.group(1)

    oa_url = (raw.get("open_access") or {}).get("oa_url") or ""
    match = _ARXIV_RE.search(oa_url)
    if match:
        return match.group(1)

    best = raw.get("best_oa_location") or {}
    for key in ("landing_page_url", "pdf_url"):
        url = best.get(key) or ""
        match = _ARXIV_RE.search(url)
        if match:
            return match.group(1)

    return ""


@lru_cache(maxsize=256)
def _fetch_openalex_work(key: str) -> dict[str, Any]:
    if not key:
        return {}
    if key.startswith("W"):
        url = f"https://api.openalex.org/works/{key}"
    elif key.startswith("10."):
        url = f"https://api.openalex.org/works/https://doi.org/{key}"
    else:
        return {}
    try:
        response = requests.get(
            url,
            timeout=12,
            headers={"User-Agent": "PaRL/0.1 (mailto:parl@local)"},
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.debug("OpenAlex work fetch failed for %s: %s", key, exc)
        return {}


def resolve_arxiv_id(
    *,
    arxiv_id: str = "",
    openalex_id: str = "",
    doi: str = "",
) -> str:
    """Best-effort arXiv id resolution from metadata already known about a paper."""
    resolved = normalize_arxiv_id(arxiv_id)
    if resolved:
        return resolved

    if openalex_id:
        work = _fetch_openalex_work(openalex_id.strip())
        resolved = extract_arxiv_id_from_work(work)
        if resolved:
            return resolved

    doi = (doi or "").replace("https://doi.org/", "").strip()
    if doi:
        work = _fetch_openalex_work(doi)
        resolved = extract_arxiv_id_from_work(work)
        if resolved:
            return resolved

    return ""
