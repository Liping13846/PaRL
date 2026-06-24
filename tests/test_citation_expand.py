"""智能引文扩展单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apis.citation_expand import (
    score_citation_candidate,
    select_citation_candidates,
    should_trigger_citation_expand,
)


def test_score_citation_candidate_prefers_relevant_title():
    paper = {
        "title": "Contrastive learning for medical image segmentation",
        "abstract": "We propose a contrastive framework for segmentation.",
        "year": 2023,
        "venue": "MICCAI",
    }
    query = "contrastive learning medical image segmentation"
    score = score_citation_candidate(paper, query, {})
    noise = score_citation_candidate(
        {
            "title": "Quantum chemistry benchmark datasets",
            "abstract": "We release molecular property datasets.",
            "year": 2020,
            "venue": "Nature",
        },
        query,
        {},
    )
    assert score > noise


def test_select_citation_candidates_filters_noise():
    candidates = [
        {
            "title": "Transformer for visual question answering",
            "abstract": "We study VQA with transformer encoders.",
        },
        {
            "title": "Soil moisture retrieval using satellite imagery",
            "abstract": "Remote sensing application.",
        },
        {
            "title": "Attention-based VQA model",
            "abstract": "Visual question answering with attention.",
        },
    ]
    selected = select_citation_candidates(
        candidates,
        "transformer visual question answering",
        top_k=2,
        min_score=0.2,
    )
    titles = [paper["title"] for paper in selected]
    assert any("VQA" in title or "visual question" in title.lower() for title in titles)
    assert "Soil moisture" not in titles


def test_should_trigger_citation_expand():
    assert should_trigger_citation_expand(
        candidate_count=3,
        high_score_count=0,
        intent="semantic_search",
    )
    assert not should_trigger_citation_expand(
        candidate_count=10,
        high_score_count=3,
        intent="metadata_search",
    )


if __name__ == "__main__":
    import unittest

    unittest.main()
