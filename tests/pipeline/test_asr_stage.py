from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


PipelineState: type | None = None
asr = None


class _FakeOrtSession:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001
        pass

    def get_inputs(self) -> list[SimpleNamespace]:  # noqa: D401 - simple stub
        return [SimpleNamespace(name="input")]

    def get_outputs(self) -> list[SimpleNamespace]:  # noqa: D401 - simple stub
        return [SimpleNamespace(name="output")]


@pytest.fixture(scope="module", autouse=True)
def _stub_dependencies() -> None:
    patch = pytest.MonkeyPatch()

    patch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            InferenceSession=_FakeOrtSession,
            SessionOptions=lambda: SimpleNamespace(
                intra_op_num_threads=0,
                inter_op_num_threads=0,
                graph_optimization_level=None,
            ),
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=0),
            get_available_providers=lambda: ["CPUExecutionProvider"],
        ),
    )

    patch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(
            feature=SimpleNamespace(
                melspectrogram=lambda *args, **kwargs: np.zeros((1, 1))  # noqa: ANN001
            ),
            power_to_db=lambda *args, **kwargs: np.zeros((1, 1)),  # noqa: ANN001
        ),
    )

    for mod in [
        "scipy",
        "scipy.signal",
        "soundfile",
        "torchaudio",
        "torch",
        "numba",
        "sklearn",
        "webrtcvad",
    ]:
        patch.setitem(sys.modules, mod, ModuleType(mod))

    global asr, PipelineState
    asr = importlib.import_module("diaremot.pipeline.stages.asr")
    PipelineState = importlib.import_module("diaremot.pipeline.stages.base").PipelineState

    yield

    patch.undo()

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
        self.load_calls: list[tuple[str, object]] = []
        self._queued_loads: list[tuple[object, object]] = []

    def queue_load(self, data: object, metadata: object | None = None) -> None:
        self._queued_loads.append((data, metadata))

    def create_checkpoint(self, input_audio_path, stage, payload, progress) -> None:  # noqa: ANN001
        self.last_args = (input_audio_path, stage, payload, progress)

    def load_checkpoint(self, input_audio_path, stage, file_hash=None):  # noqa: ANN001
        self.load_calls.append((input_audio_path, stage))
        if self._queued_loads:
            return self._queued_loads.pop(0)
        return None, None


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
        self.cache_root = Path(".")


def _build_state(tmp_path: Path) -> PipelineState:
    assert PipelineState is not None
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


def test_asr_resume_with_digest_cache(tmp_path: Path) -> None:
    pipeline = _PipelineStub(_SyncTxStub(), enable_async=False)
    checkpoint_segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker_id": "A",
            "speaker_name": "Alpha",
            "text": "hello there",
            "asr_logprob_avg": -0.25,
            "snr_db": 12.0,
            "error_flags": "",
        },
        {
            "start": 1.5,
            "end": 2.5,
            "speaker_id": "B",
            "speaker_name": "Beta",
            "text": "general kenobi",
            "asr_logprob_avg": -0.30,
            "snr_db": 11.5,
            "error_flags": "",
        },
    ]
    pipeline.checkpoints.queue_load(checkpoint_segments, None)

    state = _build_state(tmp_path)
    state.resume_tx = True
    state.tx_cache = {
        "segments": [asr._cache_metadata(seg) for seg in checkpoint_segments],  # noqa: SLF001
    }
    guard = _GuardStub()

    asr.run(pipeline, state, guard)

    assert pipeline.tx.sync_calls == 0
    assert len(state.norm_tx) == 2
    assert [seg["text"] for seg in state.norm_tx] == ["hello there", "general kenobi"]
    assert any("resume (tx cache)" in call for call in guard.progress_calls)
    assert pipeline.checkpoints.load_calls  # checkpoint consulted


def test_asr_resume_digest_mismatch_triggers_transcription(tmp_path: Path) -> None:
    pipeline = _PipelineStub(_SyncTxStub(), enable_async=False)
    checkpoint_segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker_id": "A",
            "speaker_name": "Alpha",
            "text": "hello",
            "asr_logprob_avg": -0.2,
            "snr_db": 10.0,
            "error_flags": "",
        }
    ]
    pipeline.checkpoints.queue_load(checkpoint_segments, None)

    state = _build_state(tmp_path)
    state.resume_tx = True
    bad_entry = asr._cache_metadata(checkpoint_segments[0])  # noqa: SLF001
    bad_entry["digest"] = "deadbeefdeadbeefdeadbeefdeadbeef"
    state.tx_cache = {"segments": [bad_entry]}
    guard = _GuardStub()

    asr.run(pipeline, state, guard)

    assert pipeline.tx.sync_calls == 1
    assert [seg["text"] for seg in state.norm_tx] == ["seg-0", "seg-1"]
    assert any("digest mismatch" in msg[2]["message"] for msg in pipeline.corelog.messages)


def test_asr_stage_writes_digest_cache(tmp_path: Path) -> None:
    pipeline = _PipelineStub(_SyncTxStub(), enable_async=False)
    state = _build_state(tmp_path)
    guard = _GuardStub()

    asr.run(pipeline, state, guard)

    cache_path = tmp_path / "tx.json"
    assert cache_path.exists()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))

    assert payload["segment_count"] == len(state.norm_tx)
    assert all("digest" in entry for entry in payload["segments"])
    assert all("text" not in entry for entry in payload["segments"])
