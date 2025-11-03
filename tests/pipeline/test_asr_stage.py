from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from diaremot.pipeline.stages import asr
from diaremot.pipeline.stages.base import PipelineState


class _GuardStub:
    def __init__(self) -> None:
        self.progress_calls: list[str] = []
        self.done_calls: list[dict[str, int]] = []

    def progress(self, message: str) -> None:
        self.progress_calls.append(message)

    def done(self, **kwargs: int) -> None:
        self.done_calls.append(kwargs)


class _CheckpointStub:
    def __init__(self) -> None:
        self.last_args: tuple[str, object, list[dict[str, object]], float] | None = None

    def create_checkpoint(self, input_audio_path, stage, payload, progress) -> None:  # noqa: ANN001
        self.last_args = (input_audio_path, stage, payload, progress)


class _CoreLogStub:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, dict[str, str]]] = []

    def stage(self, name: str, level: str, *, message: str) -> None:
        self.messages.append((name, level, {"message": message}))


class _AsyncTxStub:
    def __init__(self) -> None:
        self.async_calls = 0
        self.sync_calls = 0

    async def transcribe_segments_async(self, audio, sr, diar_segments):  # noqa: ANN001
        self.async_calls += 1
        await asyncio.sleep(0)
        reversed_segments = list(reversed(diar_segments))
        return [
            SimpleNamespace(
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                text=f"seg-{idx}",
                speaker_id=seg["speaker_id"],
                speaker_name=seg["speaker_name"],
                asr_logprob_avg=None,
                snr_db=None,
            )
            for idx, seg in enumerate(reversed_segments)
        ]

    def transcribe_segments(self, *args, **kwargs):  # noqa: ANN001, D401 - guard misuse
        self.sync_calls += 1
        raise AssertionError("synchronous path should not be used when async enabled")


class _SyncTxStub:
    def __init__(self) -> None:
        self.sync_calls = 0

    def transcribe_segments(self, audio, sr, diar_segments):  # noqa: ANN001
        self.sync_calls += 1
        return [
            SimpleNamespace(
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                text=f"seg-{idx}",
                speaker_id=seg["speaker_id"],
                speaker_name=seg["speaker_name"],
                asr_logprob_avg=None,
                snr_db=None,
            )
            for idx, seg in enumerate(diar_segments)
        ]


class _PipelineStub:
    def __init__(self, tx, enable_async: bool) -> None:  # noqa: ANN001
        self.tx = tx
        self.pipeline_config = SimpleNamespace(
            enable_async_transcription=enable_async
        )
        self.corelog = _CoreLogStub()
        self.checkpoints = _CheckpointStub()
        self.cache_version = "test"


def _build_state(tmp_path: Path) -> PipelineState:
    state = PipelineState(input_audio_path="input.wav", out_dir=tmp_path)
    state.y = np.zeros(3200, dtype=np.float32)
    state.sr = 16000
    state.turns = [
        {"start_time": 0.0, "end_time": 1.0, "speaker": "A", "speaker_name": "Alpha"},
        {"start_time": 1.5, "end_time": 2.5, "speaker": "B", "speaker_name": "Beta"},
    ]
    state.cache_dir = tmp_path
    state.audio_sha16 = "abc123"
    state.pp_sig = {"pp": "sig"}
    return state


def test_asr_stage_async_orders_results(tmp_path: Path) -> None:
    pipeline = _PipelineStub(_AsyncTxStub(), enable_async=True)
    state = _build_state(tmp_path)
    guard = _GuardStub()

    asr.run(pipeline, state, guard)

    assert pipeline.tx.async_calls == 1
    assert pipeline.tx.sync_calls == 0

    starts = [segment.start_time for segment in state.tx_out]
    assert starts == sorted(starts)
    assert [entry["start"] for entry in state.norm_tx] == sorted(starts)

    assert guard.progress_calls[-1].endswith("(async)")
    assert guard.done_calls[-1]["segments"] == len(state.turns)

    checkpoint = pipeline.checkpoints.last_args
    assert checkpoint is not None
    _, _, payload, progress = checkpoint
    assert progress == 60.0
    assert [row["start"] for row in payload] == sorted(starts)


def test_asr_stage_sync_fallback(tmp_path: Path) -> None:
    pipeline = _PipelineStub(_SyncTxStub(), enable_async=False)
    state = _build_state(tmp_path)
    guard = _GuardStub()

    asr.run(pipeline, state, guard)

    assert pipeline.tx.sync_calls == 1
    assert all("(async)" not in call for call in guard.progress_calls)
    assert [segment.start_time for segment in state.tx_out] == [0.0, 1.5]
