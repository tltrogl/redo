from __future__ import annotations

import numpy as np
import pytest

from diaremot.affect.analyzers.vad import VadEmotionAnalyzer


def test_vad_analyzer_returns_values() -> None:
    issues: list[str] = []
    analyzer = VadEmotionAnalyzer(
        onnx_path="missing.onnx",
        record_issue=issues.append,
    )
    analyzer._model = lambda wav, sr: (0.2, 0.5, -0.1)

    result = analyzer.analyze(np.zeros(8), 16000)

    assert result.valence == pytest.approx(0.2)
    assert result.arousal == pytest.approx(0.5)
    assert result.dominance == pytest.approx(-0.1)


def test_vad_analyzer_records_issue_when_missing_backend() -> None:
    issues: list[str] = []
    analyzer = VadEmotionAnalyzer(
        onnx_path="missing.onnx",
        record_issue=issues.append,
    )

    result = analyzer.analyze(np.zeros(8), 16000)

    assert result.valence == 0.0
    assert any("Valence/arousal/dominance model unavailable" in msg for msg in issues)
