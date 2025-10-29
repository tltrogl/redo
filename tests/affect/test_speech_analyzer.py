from __future__ import annotations

import numpy as np
import pytest

from diaremot.affect.analyzers.common import SER8_LABELS
from diaremot.affect.analyzers.speech import SpeechEmotionAnalyzer


def test_speech_analyzer_normalizes_and_low_confidence() -> None:
    issues: list[str] = []
    analyzer = SpeechEmotionAnalyzer(
        onnx_path="missing.onnx",
        labels=SER8_LABELS,
        record_issue=issues.append,
    )
    analyzer._model = lambda wav, sr: ("Happy", {"Happy": 10.0, "Neutral": 0.1})

    result = analyzer.analyze(np.ones(160), 16000)

    assert result.top == "happy"
    assert pytest.approx(sum(result.scores.values()), rel=1e-6) == 1.0
    assert not result.low_confidence


def test_speech_analyzer_records_issue_when_missing_backend() -> None:
    issues: list[str] = []
    analyzer = SpeechEmotionAnalyzer(
        onnx_path="missing.onnx",
        labels=SER8_LABELS,
        record_issue=issues.append,
    )

    result = analyzer.analyze(np.ones(10), 16000)

    assert result.top == "neutral"
    assert any("Speech emotion model unavailable" in msg for msg in issues)
