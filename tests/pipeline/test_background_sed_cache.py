import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


def _noop(*_args, **_kwargs):  # noqa: ANN001
    return None


for _mod_name in [
    "diaremot.pipeline.stages.diarize",
    "diaremot.pipeline.stages.affect",
    "diaremot.pipeline.stages.asr",
    "diaremot.pipeline.stages.dependency_check",
    "diaremot.pipeline.stages.paralinguistics",
]:
    if _mod_name not in sys.modules:
        stub = ModuleType(_mod_name)
        stub.run = _noop  # type: ignore[attr-defined]
        sys.modules[_mod_name] = stub

if "diaremot.pipeline.stages.summaries" not in sys.modules:
    summaries_stub = ModuleType("diaremot.pipeline.stages.summaries")

    def _set(attr: str) -> None:
        setattr(summaries_stub, attr, _noop)

    for attr_name in ("run_overlap", "run_conversation", "run_speaker_rollups", "run_outputs"):
        _set(attr_name)
    sys.modules["diaremot.pipeline.stages.summaries"] = summaries_stub

from diaremot.pipeline.stages.base import PipelineState
from diaremot.pipeline.stages import preprocess


class _GuardStub:
    def __init__(self) -> None:
        self.progress_calls: list[str] = []
        self.done_calls: list[dict[str, int]] = []

    def progress(self, message: str) -> None:
        self.progress_calls.append(message)

    def done(self, **kwargs: int) -> None:
        self.done_calls.append(kwargs)


class _CoreLogStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    def event(self, name: str, action: str, **payload: object) -> None:
        self.events.append((name, action, payload))

    def stage(self, name: str, level: str, *, message: str) -> None:
        self.events.append((name, level, {"message": message}))


class _StatsStub:
    def __init__(self) -> None:
        self.config_snapshot: dict[str, dict[str, object]] = {}
        self.file_id = "test.wav"


class _TaggerStub:
    def tag(self, audio, sr):  # noqa: ANN001
        return {
            "top": [{"label": "music", "score": 0.91}],
            "dominant_label": "music",
            "noise_score": 0.6,
        }


class _PipelineStub:
    def __init__(self, cache_root):  # noqa: ANN001
        self.cache_root = cache_root
        self.cache_version = "cache-test"
        self.cfg = {
            "enable_sed": True,
            "sed_mode": "timeline",
        }
        self.stats = _StatsStub()
        self.corelog = _CoreLogStub()
        self.sed_tagger = _TaggerStub()


def _build_state(tmp_path) -> PipelineState:  # noqa: ANN001
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = PipelineState(input_audio_path="input.wav", out_dir=out_dir)
    state.y = np.ones(1600, dtype=np.float32)
    state.sr = 16000
    state.pp_sig = {"pp": "sig"}
    return state


def test_background_sed_persists_timeline_sidecar(tmp_path, monkeypatch):  # noqa: ANN001
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    pipeline = _PipelineStub(cache_root)
    state = _build_state(tmp_path)
    guard = _GuardStub()

    module = ModuleType("diaremot.affect.sed_timeline")

    def _fake_run_sed_timeline(audio, *, sr, cfg, out_dir, file_id, model_paths=None, labels=None):  # noqa: ANN001
        csv_path = out_dir / "events_timeline.csv"
        csv_path.write_text("start,end,label,score\n", encoding="utf-8")
        events = [{"start": 0.0, "end": 0.5, "label": "music", "score": 0.9}]
        return SimpleNamespace(csv=csv_path, jsonl=None, events=events, mode="waveform")

    module.run_sed_timeline = _fake_run_sed_timeline
    monkeypatch.setitem(sys.modules, "diaremot.affect.sed_timeline", module)

    preprocess.run_background_sed(pipeline, state, guard)

    assert guard.done_calls, "stage should mark completion"
    assert state.sed_info is not None
    assert "timeline_events" not in state.sed_info
    assert state.sed_info.get("timeline_event_count") == 1
    events_path = state.sed_info.get("timeline_events_path")
    assert events_path is not None

    cache_dir = pipeline.cache_root / state.audio_sha16
    sed_cache = cache_dir / "sed.json"
    assert sed_cache.exists()

    payload = json.loads(sed_cache.read_text(encoding="utf-8"))
    sed_info = payload["sed_info"]
    assert "timeline_events" not in sed_info
    assert sed_info["timeline_event_count"] == 1
    assert sed_info["timeline_events_path"] == events_path

    events_payload = json.loads((cache_dir / "sed.timeline_events.json").read_text(encoding="utf-8"))
    assert events_payload["events"] == [{"start": 0.0, "end": 0.5, "label": "music", "score": 0.9}]


def test_background_sed_upgrades_inlined_events(tmp_path):  # noqa: ANN001
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    pipeline = _PipelineStub(cache_root)
    state = _build_state(tmp_path)
    state.audio_sha16 = "abc123"
    cache_dir = pipeline.cache_root / state.audio_sha16
    cache_dir.mkdir()
    state.cache_dir = cache_dir
    guard = _GuardStub()

    sed_cache = cache_dir / "sed.json"
    legacy_payload = {
        "version": pipeline.cache_version,
        "audio_sha16": state.audio_sha16,
        "pp_signature": state.pp_sig,
        "sed_signature": preprocess.compute_sed_signature(pipeline.cfg),
        "out_dir": str(state.out_dir),
        "sed_info": {
            "enabled": True,
            "top": [],
            "dominant_label": None,
            "noise_score": 0.4,
            "timeline_events": [{"start": 0.0, "end": 0.2, "label": "speech"}],
        },
    }
    sed_cache.write_text(json.dumps(legacy_payload), encoding="utf-8")

    preprocess.run_background_sed(pipeline, state, guard)

    assert guard.done_calls, "cache hit should still mark completion"
    assert state.sed_info is not None
    assert "timeline_events" not in state.sed_info
    assert state.sed_info.get("timeline_event_count") == 1

    payload = json.loads(sed_cache.read_text(encoding="utf-8"))
    sed_info = payload["sed_info"]
    assert "timeline_events" not in sed_info
    assert sed_info.get("timeline_event_count") == 1
    events_path = sed_info.get("timeline_events_path")
    assert events_path
    events_payload = json.loads((cache_dir / "sed.timeline_events.json").read_text(encoding="utf-8"))
    assert events_payload["events"] == [{"start": 0.0, "end": 0.2, "label": "speech"}]
