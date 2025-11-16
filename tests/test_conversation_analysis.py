import pytest

from diaremot.summaries import conversation_analysis as ca

SEGMENTS = [
    {"speaker_id": "A", "start": 0.0, "end": 10.0, "arousal": 0.5, "text": "hello topic one"},
    {"speaker_id": "B", "start": 10.0, "end": 20.0, "arousal": 0.7, "text": "topic one again"},
    {"speaker_id": "A", "start": 19.0, "end": 25.0, "arousal": 0.2, "text": "new words here"},
    {"speaker_id": "B", "start": 25.5, "end": 30.0, "arousal": 0.6, "text": "closing words"},
]

TOTAL_DURATION = 30.0
DOMINANCE = {"A": 52.459016393442624, "B": 47.540983606557376}
TURN_BALANCE = 0.9982545693874831
TURN_RATE = 8.0
INTERRUPTION_RATE = 2.0
INTERRUPTIONS_PER_SPEAKER = {"A": 2.0}
AVG_TURN = 7.625
RESPONSE_AVG = 0.25
RESPONSE_MEDIAN = 0.25


@pytest.mark.skipif(ca.pd is None, reason="pandas is required for the vectorised path test")
def test_analyze_conversation_flow_vectorised_metrics():
    metrics = ca.analyze_conversation_flow(list(SEGMENTS), TOTAL_DURATION)

    assert metrics.turn_taking_balance == pytest.approx(TURN_BALANCE, rel=1e-6)
    assert metrics.conversation_pace_turns_per_min == pytest.approx(TURN_RATE)
    assert metrics.interruption_rate_per_min == pytest.approx(INTERRUPTION_RATE)
    assert metrics.avg_turn_duration_sec == pytest.approx(AVG_TURN)
    assert metrics.silence_ratio == pytest.approx(0.0)
    assert metrics.interruptions_per_speaker == pytest.approx(INTERRUPTIONS_PER_SPEAKER)

    assert metrics.response_latency_stats["avg_sec"] == pytest.approx(RESPONSE_AVG)
    assert metrics.response_latency_stats["median_sec"] == pytest.approx(RESPONSE_MEDIAN)
    assert metrics.response_latency_stats["count"] == 2

    assert metrics.speaker_dominance == pytest.approx(DOMINANCE)


def test_analyze_conversation_flow_fallback(monkeypatch):
    monkeypatch.setattr(ca, "pd", None)

    metrics = ca.analyze_conversation_flow(list(SEGMENTS), TOTAL_DURATION)

    assert metrics.turn_taking_balance == pytest.approx(TURN_BALANCE, rel=1e-6)
    assert metrics.conversation_pace_turns_per_min == pytest.approx(TURN_RATE)
    assert metrics.interruption_rate_per_min == pytest.approx(INTERRUPTION_RATE)
    assert metrics.avg_turn_duration_sec == pytest.approx(AVG_TURN)
    assert metrics.interruptions_per_speaker == pytest.approx(INTERRUPTIONS_PER_SPEAKER)
    assert metrics.response_latency_stats["avg_sec"] == pytest.approx(RESPONSE_AVG)
    assert metrics.response_latency_stats["median_sec"] == pytest.approx(RESPONSE_MEDIAN)
    assert metrics.response_latency_stats["count"] == 2
    assert metrics.speaker_dominance == pytest.approx(DOMINANCE)

    # Silence ratio uses total speech time and should clamp to 0 even when speech overlaps
    assert metrics.silence_ratio == pytest.approx(0.0)
