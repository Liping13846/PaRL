"""Metadata 检索路径单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apis.metadata import (
    build_metadata_search_text,
    build_openalex_filters,
    has_hard_constraints,
    should_use_metadata_search,
)


def test_has_hard_constraints_year_venue():
    assert has_hard_constraints({"year": 2024, "topic": "VQA"})
    assert has_hard_constraints({"venue": "ACL"})
    assert not has_hard_constraints({"topic": "VQA", "method": "EMD"})


def test_should_use_metadata_search():
    constraints = {"year": 2024, "venue": "ACL"}
    assert should_use_metadata_search("metadata_search", constraints)
    assert should_use_metadata_search("semantic_search", constraints)
    assert not should_use_metadata_search("semantic_search", {"topic": "VQA"})
    assert not should_use_metadata_search("metadata_search", constraints, enabled=False)


def test_build_openalex_filters():
    filters = build_openalex_filters({"year": 2024, "venue": "ACL", "topic": "VQA"})
    assert filters == ["publication_year:2024"]


def test_build_metadata_search_text_prefers_semantic_fields():
    text = build_metadata_search_text(
        "ACL 2024 papers on visual question answering",
        {"topic": "visual question answering", "method": "EMD", "year": 2024, "venue": "ACL"},
    )
    assert "visual question answering" in text
    assert "computational linguistics" in text
    assert "EMD" in text


def test_build_metadata_search_text_fallback():
    assert build_metadata_search_text("raw query", {}) == "raw query"


if __name__ == "__main__":
    import unittest

    unittest.main()
