from __future__ import annotations

import os
import socket
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .logger import logger


def bool_env(name: str) -> bool | None:
    val = os.getenv(name)
    if val is None:
        return None
    norm = val.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    return None


def can_reach_host(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def torch_repo_cached() -> bool:
    try:
        import torch.hub as hub

        hub_dir = Path(hub.get_dir())
    except Exception:
        return False
    candidates = [
        hub_dir / "snakers4_silero-vad_master",
        hub_dir / "snakers4_silero-vad_main",
        hub_dir / "silero-vad",
    ]
    return any(path.exists() for path in candidates)


def resolve_state_shape(shape: tuple[Any, ...] | None) -> tuple[int, ...]:
    default = (2, 1, 128)
    if not shape:
        return default
    resolved: list[int] = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(int(dim))
        else:
            resolved.append(default[idx] if idx < len(default) else 1)
    if len(resolved) < len(default):
        resolved.extend(default[len(resolved) :])
    return tuple(resolved[: len(default)])


def merge_regions(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def energy_vad_fallback(
    wav: np.ndarray, sr: int, gate_db: float, hop_sec: float
) -> list[tuple[float, float]]:
    if wav.size == 0:
        return []
    hop = max(1, int(hop_sec * sr))
    if len(wav) < hop * 2:
        return []
    try:
        frames = librosa.util.frame(wav, frame_length=hop * 2, hop_length=hop).T
    except Exception:
        return []
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)
    speech = rms_db > gate_db
    regions = []
    start = None
    for i, is_speech in enumerate(speech):
        t = i * hop_sec
        if is_speech and start is None:
            start = t
        elif not is_speech and start is not None:
            regions.append((start, t))
            start = None
    if start is not None:
        regions.append((start, len(speech) * hop_sec))
    padded = [(max(0.0, s - 0.1), e + 0.1) for s, e in regions]
    if not padded:
        return []
    return merge_regions(padded, gap=0.5)


def download_silero_torch(timeout: float = 30.0) -> tuple[Any, Any] | None:
    def _hub_load():
        import torch.hub as hub  # type: ignore[attr-defined]

        return hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            trust_repo=True,
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_hub_load)
            return future.result(timeout=timeout)
    except FuturesTimeoutError:
        logger.warning("Silero VAD TorchHub load timed out after %.1fs", timeout)
    except Exception as exc:
        logger.warning("Silero VAD TorchHub unavailable: %s", exc)
    return None


__all__ = [
    "bool_env",
    "can_reach_host",
    "torch_repo_cached",
    "resolve_state_shape",
    "merge_regions",
    "energy_vad_fallback",
    "download_silero_torch",
]
