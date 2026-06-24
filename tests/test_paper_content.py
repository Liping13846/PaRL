"""Tests for ar5iv conclusion extraction and translation helpers."""

from apis.paper_content import (
    _is_conclusion_title,
    extract_conclusion_from_html,
    translate_en_to_zh,
)


SAMPLE_HTML = """
<html><body>
<section id="s1"><h2>Introduction</h2><p>Intro text here with enough words.</p></section>
<section id="s2"><h2>Method</h2><p>Method details here with enough words for testing.</p></section>
<section id="s3"><h2>Conclusion</h2><p>
We presented a new approach for visual question answering.
Our experiments show consistent improvements over strong baselines.
Future work will extend the model to multimodal reasoning tasks.
</p></section>
<section id="s4"><h2>References</h2><p>bib content</p></section>
</body></html>
"""


def test_is_conclusion_title():
    assert _is_conclusion_title("Conclusion")
    assert _is_conclusion_title("5 Conclusions and Future Work")
    assert not _is_conclusion_title("References")


def test_extract_conclusion_from_html():
    title, text = extract_conclusion_from_html(SAMPLE_HTML)
    assert title == "Conclusion"
    assert "visual question answering" in text
    assert "Future work" in text


def test_translate_en_to_zh_fallback_message():
    result = translate_en_to_zh("")
    assert result == ""
