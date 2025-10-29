from __future__ import annotations

from typing import Any

import pytest

from diaremot.affect.analyzers import intent as intent_module
from diaremot.affect.analyzers.intent import IntentAnalyzer


def test_intent_analyzer_pipeline_scores_sorted() -> None:
    issues: list[str] = []
    analyzer = IntentAnalyzer(labels=["status", "action"], backend="torch", record_issue=issues.append)

    def fake_pipeline(text: str, candidate_labels: list[str], multi_label: bool) -> dict[str, Any]:
        return {"labels": ["action", "status"], "scores": [0.8, 0.2]}

    analyzer._lazy_prepare = lambda: None
    analyzer._intent_pipeline = fake_pipeline

    result = analyzer.infer("schedule a meeting")

    assert result.top == "action"
    assert result.top3[0]["score"] >= result.top3[1]["score"]


def test_intent_analyzer_records_issue_when_pipeline_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    issues: list[str] = []
    analyzer = IntentAnalyzer(labels=["status"], backend="torch", record_issue=issues.append)

    monkeypatch.setattr(intent_module, "_maybe_import_transformers_pipeline", lambda: None)

    analyzer.infer("status update")

    assert any("Transformers pipeline unavailable" in msg for msg in issues)


def test_intent_analyzer_blank_text_uses_default() -> None:
    analyzer = IntentAnalyzer(labels=["status", "opinion"], backend="auto")

    result = analyzer.infer("")

    assert result.top == "status"
    assert len(result.top3) == 2
