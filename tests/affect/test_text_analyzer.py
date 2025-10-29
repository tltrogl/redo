from __future__ import annotations

import pytest

from diaremot.affect.analyzers.common import GOEMOTIONS_LABELS
from diaremot.affect.analyzers.text import TextEmotionAnalyzer


def test_text_analyzer_normalizes_scores() -> None:
    issues: list[str] = []
    analyzer = TextEmotionAnalyzer(
        onnx_path="missing.onnx",
        model_dir=".",
        disable_downloads=True,
        record_issue=issues.append,
    )
    analyzer._onnx_model = lambda text: {"Joy": 0.25, "neutral": 0.75}

    result = analyzer.analyze("hello world")

    assert result.top5[0]["label"] == "neutral"
    assert pytest.approx(sum(result.full.values()), rel=1e-6) == 1.0
    assert set(result.full) >= set(GOEMOTIONS_LABELS)


def test_text_analyzer_records_issue_when_backend_missing() -> None:
    issues: list[str] = []
    analyzer = TextEmotionAnalyzer(
        onnx_path="missing.onnx",
        model_dir=".",
        disable_downloads=True,
        record_issue=issues.append,
    )

    analyzer.analyze("some text")

    assert any("Text emotion model unavailable" in msg for msg in issues)
