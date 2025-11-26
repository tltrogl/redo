from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .common import (
    IssueRecorder,
    VadEmotionResult,
    default_vad_result,
    ort_session,
    target_sample_rate,
)

logger = logging.getLogger(__name__)


class OnnxVADEmotion:
    def __init__(self, model_path: str):
        self.sess = ort_session(model_path)

    def __call__(self, y: np.ndarray, sr: int) -> tuple[float, float, float]:
        if y.size == 0:
            return 0.0, 0.0, 0.0
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        target_sr = target_sample_rate()
        try:
            import librosa  # type: ignore
        except ImportError:  # pragma: no cover
            librosa = None  # type: ignore
        if librosa is not None and sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        x = y.astype(np.float32)[None, :]
        inp_name = self.sess.get_inputs()[0].name
        out = self.sess.run(None, {inp_name: x})
        arr = np.array(out[0]).astype(np.float32).ravel()
        if arr.size >= 3:
            v, a, d = arr[:3]
        else:
            v = a = d = 0.0
        v = float(np.clip(v, -1.0, 1.0))
        a = float(np.clip(a, -1.0, 1.0))
        d = float(np.clip(d, -1.0, 1.0))
        return v, a, d


class VadEmotionAnalyzer:
    def __init__(
        self,
        *,
        onnx_path: str,
        record_issue: IssueRecorder | None = None,
    ) -> None:
        self._onnx_path = os.fspath(Path(onnx_path).expanduser())
        self._record_issue: IssueRecorder = record_issue or (lambda _: None)
        self._model: Callable[[np.ndarray, int], tuple[float, float, float]] | None = None
        self._attempted = False

    def _ensure_model(self) -> None:
        if self._model is not None or self._attempted:
            return
        self._attempted = True
        try:
            self._model = OnnxVADEmotion(self._onnx_path)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("V/A/D ONNX unavailable: %s", exc)
            self._record_issue(
                f"Valence/arousal/dominance model unavailable under {self._onnx_path}"
            )
            self._model = None

    def analyze(self, wav: np.ndarray | None, sr: int | None) -> VadEmotionResult:
        if wav is None or sr is None:
            return default_vad_result()

        self._ensure_model()
        if self._model is None:
            self._record_issue("Valence/arousal/dominance model unavailable; using zeros")
            return default_vad_result()

        try:
            v, a, d = self._model(np.asarray(wav, dtype=np.float32), int(sr))
        except Exception as exc:  # noqa: BLE001
            logger.warning("V/A/D inference failed: %s", exc)
            self._record_issue(f"VAD emotion inference failed: {exc}")
            return default_vad_result()

        return VadEmotionResult(valence=float(v), arousal=float(a), dominance=float(d))


__all__ = [
    "VadEmotionAnalyzer",
    "OnnxVADEmotion",
]
