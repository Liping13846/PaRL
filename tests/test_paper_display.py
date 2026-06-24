"""Tests for paper display enrichment."""

from apis.paper_display import (
    build_bibtex,
    enrich_result_dict,
    extract_conclusion_snippet,
    resolve_paper_links,
)


def test_resolve_paper_links_prefers_arxiv():
    links = resolve_paper_links(
        {
            "arxiv_id": "2301.00001",
            "doi": "10.1234/example",
            "openalex_id": "W123",
        }
    )
    assert links["primary"] == "https://arxiv.org/abs/2301.00001"
    assert links["pdf"].endswith("2301.00001.pdf")
    assert links["doi"] == "https://doi.org/10.1234/example"


def test_extract_conclusion_snippet():
    abstract = (
        "We study transformers for VQA. "
        "Our method improves accuracy by 5 points. "
        "In conclusion, the proposed model generalizes well."
    )
    conclusion = extract_conclusion_snippet(abstract)
    assert "In conclusion" in conclusion


def test_build_bibtex_arxiv():
    bib = build_bibtex(
        {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "arxiv_id": "1706.03762",
        }
    )
    assert "@misc{" in bib
    assert "eprint = {1706.03762}" in bib
    assert "archivePrefix = {arXiv}" in bib
    assert "title = {Attention Is All You Need}" in bib


def test_build_bibtex_conference():
    bib = build_bibtex(
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Jacob Devlin"],
            "year": 2019,
            "venue": "Proceedings of NAACL",
            "doi": "10.18653/v1/N19-1423",
        }
    )
    assert "@inproceedings{" in bib
    assert "booktitle = {Proceedings of NAACL}" in bib
    assert "doi = {10.18653/v1/N19-1423}" in bib


def test_enrich_result_dict_adds_display_fields():
    enriched = enrich_result_dict(
        {
            "title": "Test Paper",
            "abstract": "We show that X works. In conclusion, it is effective.",
            "arxiv_id": "2301.00001",
            "markdown_evidence": "We show that X works.",
        }
    )
    assert enriched["links"]["arxiv"]
    assert enriched["conclusion"]
    assert enriched["abstract"]
    assert enriched["bibtex"]
    assert "@misc{" in enriched["bibtex"]
