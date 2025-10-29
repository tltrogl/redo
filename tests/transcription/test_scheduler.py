import sys
import types

import numpy as np
import pytest

_librosa_stub = types.SimpleNamespace(
    resample=lambda data, orig_sr, target_sr: np.asarray(data, dtype=np.float32),
    __version__="0.0-test",
)
sys.modules.setdefault("librosa", _librosa_stub)
sys.modules.setdefault(
    "diaremot.pipeline.runtime_env",
    types.SimpleNamespace(iter_model_roots=lambda: []),
)

from diaremot.pipeline.transcription.models import TranscriptionSegment
from diaremot.pipeline.transcription.postprocess import distribute_batch_results
from diaremot.pipeline.transcription.scheduler import BatchingConfig, create_batch_groups


def test_create_batch_groups_batches_short_segments():
    segments = [
        {"start_time": i * 2.0, "end_time": i * 2.0 + 2.0} for i in range(6)
    ]
    config = BatchingConfig(
        enabled=True,
        min_segments_threshold=4,
        short_segment_max_sec=3.0,
        max_segments_per_batch=3,
    )

    groups = create_batch_groups(segments, config)
    assert "batch_0" in groups
    assert len(groups["batch_0"]) <= config.max_segments_per_batch
    assert groups["individual"] == []


def test_distribute_batch_results_with_word_boundaries():
    batch_segment = TranscriptionSegment(
        start_time=0.0,
        end_time=6.0,
        text="hello world again",
        confidence=0.9,
        words=[
            {"word": "hello", "start": 0.1, "end": 0.5},
            {"word": "world", "start": 2.1, "end": 2.5},
            {"word": "again", "start": 4.1, "end": 4.6},
        ],
        model_used="faster",
    )
    boundaries = [
        {"original": {"start_time": 0.0, "end_time": 2.0}, "concat_start": 0.0, "concat_end": 2.0},
        {"original": {"start_time": 2.0, "end_time": 4.0}, "concat_start": 2.0, "concat_end": 4.0},
        {"original": {"start_time": 4.0, "end_time": 6.0}, "concat_start": 4.0, "concat_end": 6.0},
    ]

    distributed = distribute_batch_results(batch_segment, boundaries)
    assert [seg.text for seg in distributed] == ["hello", "world", "again"]
    assert distributed[0].words[0]["start"] == pytest.approx(0.1)
    assert distributed[-1].start_time == 4.0
    assert all(seg.words for seg in distributed)
