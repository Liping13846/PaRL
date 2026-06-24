"""Utilities for enriching paper results in the web UI."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urljoin

import requests

from apis.arxiv_resolve import resolve_arxiv_id
from apis.paper_content import fetch_paper_conclusion, translate_en_to_zh_cached

logger = logging.getLogger(__name__)

_AR5IV_BASE = "https://ar5iv.labs.arxiv.org/html/"
_CONCLUSION_PATTERNS = (
    r"\bwe conclude\b",
    r"\bin conclusion\b",
    r"\bto conclude\b",
    r"\bour results (?:show|demonstrate|indicate|suggest)\b",
    r"\bwe (?:show|demonstrate|find|observe) that\b",
    r"\bin summary\b",
    r"\boverall\b",
    r"\bthese results\b",
)


def resolve_paper_links(paper: dict[str, Any]) -> dict[str, str]:
    """Build canonical outbound links for a paper record."""
    arxiv_id = (paper.get("arxiv_id") or "").split("v")[0].strip()
    doi = (paper.get("doi") or "").strip()
    openalex_id = (paper.get("openalex_id") or "").strip()
    raw_url = (paper.get("url") or "").strip()

    links: dict[str, str] = {}
    if arxiv_id:
        links["arxiv"] = f"https://arxiv.org/abs/{arxiv_id}"
        links["pdf"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        links["ar5iv"] = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    if doi:
        links["doi"] = f"https://doi.org/{doi}"
    if openalex_id:
        links["openalex"] = f"https://openalex.org/{openalex_id}"
    elif raw_url.startswith("https://openalex.org/"):
        links["openalex"] = raw_url

    if arxiv_id:
        links["primary"] = links["arxiv"]
    elif doi:
        links["primary"] = links["doi"]
    elif links.get("openalex"):
        links["primary"] = links["openalex"]
    elif raw_url.startswith("http"):
        links["primary"] = raw_url

    return links


def _bibtex_escape(value: str) -> str:
    text = (value or "").strip()
    text = text.replace("\\", "\\textbackslash{}")
    text = text.replace("{", "\\{").replace("}", "\\}")
    return text


def _bibtex_key(paper: dict[str, Any]) -> str:
    authors = paper.get("authors") or []
    first_author = authors[0] if authors else "unknown"
    surname = re.sub(r"[^a-zA-Z]", "", first_author.split()[-1] if first_author else "unknown").lower()
    surname = surname or "unknown"
    year = str(paper.get("year") or "0000")
    title = paper.get("title") or "paper"
    title_token = re.sub(r"[^a-zA-Z0-9]", "", title.split()[0].lower() if title.split() else "paper")[:10]
    suffix = (paper.get("arxiv_id") or paper.get("openalex_id") or paper.get("paper_id") or "")
    suffix = re.sub(r"[^a-zA-Z0-9]", "", str(suffix).lower())[-4:]
    key = f"{surname}{year}{title_token}{suffix}"
    return re.sub(r"[^a-zA-Z0-9]", "", key) or "paper"


def _looks_like_conference(venue: str) -> bool:
    if not venue:
        return False
    lowered = venue.lower()
    keywords = (
        "conference", "proceedings", "workshop", "symposium", "acl", "emnlp",
        "naacl", "cvpr", "iccv", "eccv", "neurips", "nips", "icml", "iclr",
        "aaai", "ijcai", "kdd", "sigir", "www", "acl anthology",
    )
    return any(keyword in lowered for keyword in keywords)


def build_bibtex(paper: dict[str, Any]) -> str:
    """Build a BibTeX entry from normalized paper metadata."""
    title = (paper.get("title") or "Untitled").strip()
    authors = paper.get("authors") or []
    year = paper.get("year")
    venue = (paper.get("venue") or "").strip()
    doi = (paper.get("doi") or "").strip()
    arxiv_id = (paper.get("arxiv_id") or "").split("v")[0].strip()
    links = resolve_paper_links(paper)
    url = links.get("primary") or links.get("openalex") or (paper.get("url") or "").strip()
    cite_key = _bibtex_key(paper)

    author_field = " and ".join(_bibtex_escape(name) for name in authors) or "Unknown"
    lines = [f"@{'misc' if arxiv_id and not venue else 'inproceedings' if _looks_like_conference(venue) else 'article'}{{{cite_key},"]
    lines.append(f"  title = {{{_bibtex_escape(title)}}},")
    lines.append(f"  author = {{{author_field}}},")
    if year:
        lines.append(f"  year = {{{year}}},")

    if arxiv_id and not venue:
        lines.append(f"  eprint = {{{arxiv_id}}},")
        lines.append("  archivePrefix = {arXiv},")
    elif _looks_like_conference(venue):
        lines.append(f"  booktitle = {{{_bibtex_escape(venue)}}},")
    elif venue:
        lines.append(f"  journal = {{{_bibtex_escape(venue)}}},")

    if doi:
        lines.append(f"  doi = {{{_bibtex_escape(doi)}}},")
    if url:
        lines.append(f"  url = {{{_bibtex_escape(url)}}},")

    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("}")
    return "\n".join(lines)


def extract_conclusion_snippet(abstract: str) -> str:
    """Heuristically pick a conclusion-like sentence from the abstract."""
    text = (abstract or "").strip()
    if not text:
        return ""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return ""

    for sentence in reversed(sentences):
        lower = sentence.lower()
        if any(re.search(pattern, lower) for pattern in _CONCLUSION_PATTERNS):
            return sentence

    if len(sentences) >= 2:
        return " ".join(sentences[-2:])
    return sentences[-1]


def fetch_arxiv_figures(arxiv_id: str, *, max_figures: int = 2, timeout: float = 8.0) -> list[dict[str, str]]:
    """Fetch the first few figure images from ar5iv HTML (best effort)."""
    arxiv_id = (arxiv_id or "").split("v")[0].strip()
    if not arxiv_id:
        return []

    page_url = f"{_AR5IV_BASE}{arxiv_id}"
    try:
        response = requests.get(page_url, timeout=timeout, headers={"User-Agent": "PaRL/0.1"})
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("ar5iv figure fetch failed for %s: %s", arxiv_id, exc)
        return []

    html = response.text
    if "html fallback" in html.lower() or "not available" in html.lower():
        return []

    figures: list[dict[str, str]] = []
    seen: set[str] = set()

    for match in re.finditer(
        r"<figure[^>]*>(.*?)</figure>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        block = match.group(1)
        img_match = re.search(
            r'<img[^>]+src=["\']([^"\']+)["\']',
            block,
            flags=re.IGNORECASE,
        )
        if not img_match:
            continue

        src = img_match.group(1).strip()
        if not src or src.startswith("data:"):
            continue

        absolute = urljoin(page_url, src)
        if absolute in seen:
            continue
        seen.add(absolute)

        caption_match = re.search(
            r'<figcaption[^>]*>(.*?)</figcaption>',
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        caption = ""
        if caption_match:
            caption = re.sub(r"<[^>]+>", " ", caption_match.group(1))
            caption = re.sub(r"\s+", " ", caption).strip()

        figures.append({"url": absolute, "caption": caption[:240]})
        if len(figures) >= max_figures:
            break

    return figures


def enrich_result_dict(result: dict[str, Any]) -> dict[str, Any]:
    """Add display-oriented fields for the web UI."""
    abstract = (result.get("abstract") or "").strip()
    evidence = (result.get("markdown_evidence") or "").strip()
    links = resolve_paper_links(result)

    conclusion = ""
    if not (result.get("arxiv_id") or "").strip():
        conclusion = extract_conclusion_snippet(abstract)
        if not conclusion and evidence and evidence != abstract[: len(evidence)]:
            conclusion = extract_conclusion_snippet(evidence) or evidence[:400]

    enriched = dict(result)
    enriched["abstract"] = abstract
    enriched["abstract_zh"] = translate_en_to_zh_cached(abstract) if abstract else ""
    enriched["conclusion"] = conclusion
    enriched["conclusion_en"] = ""
    enriched["conclusion_zh"] = ""
    enriched["conclusion_section_title"] = ""
    enriched["conclusion_source"] = "pending" if (result.get("arxiv_id") or "").strip() else "abstract_fallback"
    enriched["links"] = links
    enriched["citation_count"] = result.get("citation_count", 0)
    enriched["doi"] = (result.get("doi") or "").strip()
    enriched["openalex_id"] = (result.get("openalex_id") or "").strip()
    enriched["arxiv_id"] = resolve_arxiv_id(
        arxiv_id=(result.get("arxiv_id") or "").strip(),
        openalex_id=enriched["openalex_id"],
        doi=enriched["doi"],
    )
    enriched["conclusion_source"] = "pending" if enriched["arxiv_id"] else "abstract_fallback"
    enriched["bibtex"] = build_bibtex(enriched)
    return enriched


def _enrich_with_conclusion(enriched: dict[str, Any]) -> dict[str, Any]:
    conc = fetch_paper_conclusion(
        enriched.get("arxiv_id", ""),
        abstract_fallback=enriched.get("abstract", ""),
        openalex_id=enriched.get("openalex_id", ""),
        doi=enriched.get("doi", ""),
    )
    enriched["conclusion_en"] = conc["conclusion_en"]
    enriched["conclusion_zh"] = conc["conclusion_zh"]
    enriched["conclusion_source"] = conc["source"]
    enriched["conclusion_section_title"] = conc.get("section_title", "")
    if conc.get("arxiv_id"):
        enriched["arxiv_id"] = conc["arxiv_id"]
    return enriched


def enrich_results_batch(results: list[dict[str, Any]], *, max_workers: int = 4) -> list[dict[str, Any]]:
    """Enrich search results with metadata, conclusions, and pre-translated Chinese text."""
    if not results:
        return []

    staged = [enrich_result_dict(item) for item in results]
    if len(staged) == 1:
        return [_enrich_with_conclusion(staged[0])]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(staged))) as pool:
        return list(pool.map(_enrich_with_conclusion, staged))
