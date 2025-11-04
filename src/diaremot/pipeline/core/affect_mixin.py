"""Affect helpers shared across pipeline executors."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..outputs import default_affect


def _normalize_waveform_input(wav: Any) -> np.ndarray | None:
    """Convert arbitrary audio buffers into NumPy without needless copies."""

    if wav is None:
        return None

    if isinstance(wav, np.ndarray):
        return wav.astype(np.float32, copy=False)

    if isinstance(wav, memoryview):
        return np.frombuffer(wav, dtype=np.float32)

    as_array = getattr(wav, "as_array", None)
    if callable(as_array):
        arr = as_array(copy=False)
        if isinstance(arr, np.ndarray):
            return arr.astype(np.float32, copy=False)

    if hasattr(wav, "__array__"):
        return np.asarray(wav, dtype=np.float32)

    if isinstance(wav, Iterable):
        return np.fromiter((float(value) for value in wav), dtype=np.float32)

    return np.asarray(wav, dtype=np.float32)


class AffectMixin:
    def _affect_hint(self, v, a, d, intent):
        try:
            if a is None or v is None:
                return "neutral-status"
            if a > 0.5 and v < 0:
                return "agitated-negative"
            if a < 0.3 and v > 0.2:
                return "calm-positive"
            return f"neutral-{intent}"
        except Exception:  # pragma: no cover - defensive
            return "neutral-status"

    def _affect_unified(self, wav: Any, sr: int, text: str) -> dict[str, Any]:
        try:
            normalized = _normalize_waveform_input(wav)
            if hasattr(self.affect, "analyze"):
                res = self.affect.analyze(wav=normalized, sr=sr, text=text)
                if getattr(self.affect, "issues", None):
                    for issue in self.affect.issues:
                        if issue not in self.stats.issues:
                            self.stats.issues.append(issue)
                return res or default_affect()
            return default_affect()
        except Exception as exc:  # pragma: no cover - best effort fallback
            self.corelog.warn(f"Affect analysis failed: {exc}")
            return default_affect()
