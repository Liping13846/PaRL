"""Fetch full-text sections (Conclusion) from ar5iv and translate for bilingual display."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import requests

from apis.arxiv_resolve import normalize_arxiv_id, resolve_arxiv_id

logger = logging.getLogger(__name__)

_HTML_SOURCES = (
    "https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
    "https://arxiv.org/html/{arxiv_id}",
)
_USER_AGENT = "PaRL/0.1"
_CONCLUSION_TITLE = re.compile(
    r"(^|\b)(conclusions?|concluding remarks|summary and conclusions?|"
    r"discussion and conclusions?|conclusion and future work|"
    r"concluding discussion)(\b|$)",
    re.IGNORECASE,
)
_CONCLUSION_ZH = re.compile(r"结论|总结与展望|结语")
_STOP_SECTION = re.compile(
    r"\b(references|bibliography|acknowledgments?|appendix|supplementary)\b|参考文献|致谢|附录",
    re.IGNORECASE,
)
_MAX_CONCLUSION_CHARS = 4000
_MAX_TRANSLATE_CHARS = 3500
_CHUNK_SIZE = 450
_HTML_CACHE: dict[str, str] = {}
_CONCLUSION_CACHE: dict[str, tuple[str, str, str]] = {}


def _extract_section_plaintext(section) -> str:
    if section is None:
        return ""

    from bs4 import BeautifulSoup

    fragment = BeautifulSoup(str(section), "lxml")
    root = fragment.find("section") or fragment
    for tag in root.find_all(["figure", "cite", "script", "style", "nav"]):
        tag.decompose()
    for tag in root.find_all(["h1", "h2", "h3", "h4", "h5"]):
        tag.decompose()

    text = root.get_text(" ", strip=True)
    text = re.sub(r"~\s*\\cite\{.*?\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_CONCLUSION_CHARS]


def _collect_headings(soup) -> list[tuple[Any, str, int]]:
    headings: list[tuple[Any, str, int]] = []
    for index, tag in enumerate(soup.find_all(["h1", "h2", "h3", "h4", "h5"])):
        title = tag.get_text(" ", strip=True)
        if title:
            headings.append((tag, title, index))
    return headings


def _is_conclusion_title(title: str) -> bool:
    title = re.sub(r"^\d+(\.\d+)*\s*", "", title.strip())
    if not title or len(title) > 120:
        return False
    if _STOP_SECTION.search(title):
        return False
    return bool(_CONCLUSION_TITLE.search(title) or _CONCLUSION_ZH.search(title))


def extract_conclusion_from_html(html: str) -> tuple[str, str]:
    """Return (section_title, conclusion_text) parsed from ar5iv HTML."""
    if not html or "html fallback" in html.lower():
        return "", ""

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed; cannot parse ar5iv HTML")
        return "", ""

    soup = BeautifulSoup(html, "lxml")
    headings = _collect_headings(soup)
    if not headings:
        return "", ""

    stop_index = len(headings)
    conclusion_index: int | None = None

    for idx, (_tag, title, _order) in enumerate(headings):
        if _STOP_SECTION.search(title):
            stop_index = min(stop_index, idx)
            break

    for idx, (tag, title, _order) in enumerate(headings):
        if idx >= stop_index:
            break
        if not _is_conclusion_title(title):
            continue
        conclusion_index = idx

    if conclusion_index is None:
        return "", ""

    tag, title, _order = headings[conclusion_index]
    section = tag.find_parent("section")
    if section is None:
        section = tag.parent
    text = _extract_section_plaintext(section)
    if len(text) < 25:
        return "", ""
    return title, text


def _fetch_arxiv_html(arxiv_id: str, *, timeout: float = 12.0) -> str:
    arxiv_id = normalize_arxiv_id(arxiv_id)
    if not arxiv_id:
        return ""
    if arxiv_id in _HTML_CACHE:
        return _HTML_CACHE[arxiv_id]

    for template in _HTML_SOURCES:
        url = template.format(arxiv_id=arxiv_id)
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
            response.raise_for_status()
            html = response.text
            if html and "html fallback" not in html.lower() and len(html) > 1000:
                _HTML_CACHE[arxiv_id] = html
                return html
        except requests.RequestException as exc:
            logger.debug("HTML fetch failed for %s via %s: %s", arxiv_id, url, exc)
    return ""


def _load_conclusion_from_arxiv(arxiv_id: str) -> tuple[str, str, str]:
    arxiv_id = normalize_arxiv_id(arxiv_id)
    if not arxiv_id:
        return "", "", ""

    cached = _CONCLUSION_CACHE.get(arxiv_id)
    if cached:
        return cached

    html = _fetch_arxiv_html(arxiv_id)
    title, text = extract_conclusion_from_html(html)
    bundle = (title, text, "ar5iv_conclusion" if text else "")
    if text:
        _CONCLUSION_CACHE[arxiv_id] = bundle
    return bundle


def _chunk_text(text: str, size: int = _CHUNK_SIZE) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]

    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence[:size]
    if current:
        chunks.append(current)
    return chunks or [text[:size]]


def _translate_chunk_mymemory(text: str) -> str:
    response = requests.get(
        "https://api.mymemory.translated.net/get",
        params={"q": text, "langpair": "en|zh-CN"},
        timeout=10,
        headers={"User-Agent": _USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    translated = (payload.get("responseData") or {}).get("translatedText") or ""
    return translated.strip()


def _translate_chunk_deep_translator(text: str) -> str:
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source="en", target="zh-CN").translate(text).strip()


def translate_en_to_zh(text: str) -> str:
    """Translate English academic text to Chinese with lightweight free APIs."""
    text = (text or "").strip()
    if not text:
        return ""

    clipped = text[:_MAX_TRANSLATE_CHARS]
    chunks = _chunk_text(clipped)
    translated_parts: list[str] = []

    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            translated_parts.append(_translate_chunk_deep_translator(chunk))
            continue
        except Exception as exc:
            logger.debug("Google translate failed: %s", exc)

        try:
            translated_parts.append(_translate_chunk_mymemory(chunk))
        except Exception as exc:
            logger.warning("Translation failed for chunk: %s", exc)
            translated_parts.append("")

    result = " ".join(part for part in translated_parts if part).strip()
    return result or "翻译暂不可用，请查看英文原文。"


@lru_cache(maxsize=512)
def translate_en_to_zh_cached(text: str) -> str:
    return translate_en_to_zh(text)


def translate_text(text: str) -> dict[str, str]:
    """Translate English text to Chinese on demand."""
    text_en = (text or "").strip()
    return {
        "text_en": text_en,
        "text_zh": translate_en_to_zh_cached(text_en) if text_en else "",
    }


def fetch_paper_conclusion(
    arxiv_id: str = "",
    *,
    abstract_fallback: str = "",
    openalex_id: str = "",
    doi: str = "",
) -> dict[str, str]:
    """Fetch conclusion text for a paper, resolving arXiv id when possible."""
    resolved_arxiv = resolve_arxiv_id(
        arxiv_id=arxiv_id,
        openalex_id=openalex_id,
        doi=doi,
    )
    section_title, conclusion_en, source = "", "", ""

    if resolved_arxiv:
        section_title, conclusion_en, source = _load_conclusion_from_arxiv(resolved_arxiv)

    if not conclusion_en and abstract_fallback:
        from apis.paper_display import extract_conclusion_snippet

        conclusion_en = extract_conclusion_snippet(abstract_fallback)
        source = "abstract_fallback" if conclusion_en else "unavailable"

    if not conclusion_en:
        source = "unavailable"
    elif source != "ar5iv_conclusion":
        source = source or "abstract_fallback"

    conclusion_zh = translate_en_to_zh_cached(conclusion_en) if conclusion_en else ""

    return {
        "arxiv_id": resolved_arxiv,
        "section_title": section_title,
        "conclusion_en": conclusion_en,
        "conclusion_zh": conclusion_zh,
        "source": source,
    }
