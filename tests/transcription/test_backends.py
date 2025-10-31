import sys
import types

import numpy as np

_librosa_stub = types.SimpleNamespace(
    resample=lambda data, orig_sr, target_sr: np.asarray(data, dtype=np.float32),
    __version__="0.0-test",
)
sys.modules.setdefault("librosa", _librosa_stub)
sys.modules.setdefault(
    "diaremot.pipeline.runtime_env",
    types.SimpleNamespace(iter_model_roots=lambda: []),
)

from diaremot.pipeline.transcription.backends import ModelManager, backends


def test_model_manager_prefers_faster_whisper(monkeypatch):
    monkeypatch.setattr(backends, "has_faster_whisper", True, raising=False)
    monkeypatch.setattr(backends, "has_openai_whisper", True, raising=False)

    manager = ModelManager()
    sentinel = object()

    def fake_faster(config):
        return sentinel

    def fake_openai(config):
        raise AssertionError("openai backend should not be used when faster is available")

    monkeypatch.setattr(manager, "_load_faster_whisper", fake_faster)
    monkeypatch.setattr(manager, "_load_openai_whisper", fake_openai)

    config = {
        "model_size": "tiny",
        "compute_type": "int8",
        "cpu_threads": 1,
        "asr_backend": "auto",
        "local_first": True,
    }

    import asyncio

    asyncio.run(manager._load_model("primary", config))
    assert manager._models["primary"] is sentinel


def test_model_manager_falls_back_to_openai(monkeypatch):
    monkeypatch.setattr(backends, "has_faster_whisper", False, raising=False)
    monkeypatch.setattr(backends, "has_openai_whisper", True, raising=False)

    manager = ModelManager()
    sentinel = object()
    calls = []

    def fake_faster(config):  # pragma: no cover - safety
        calls.append("faster")
        raise RuntimeError("no faster")

    def fake_openai(config):
        return sentinel

    monkeypatch.setattr(manager, "_load_faster_whisper", fake_faster)
    monkeypatch.setattr(manager, "_load_openai_whisper", fake_openai)

    config = {
        "model_size": "tiny",
        "compute_type": "int8",
        "cpu_threads": 1,
        "asr_backend": "auto",
        "local_first": True,
    }

    import asyncio

    asyncio.run(manager._load_model("primary", config))
    assert manager._models["primary"] is sentinel
    assert calls == []  # no faster attempt when unavailable
