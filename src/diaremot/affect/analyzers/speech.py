from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .common import (
    SER8_LABELS,
    IssueRecorder,
    SpeechEmotionResult,
    default_speech_result,
    ensure_16k_mono,
    normalize_scores,
    ort_session,
    ser_low_confidence,
    target_sample_rate,
    trim_max_len,
)

logger = logging.getLogger(__name__)

try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    librosa = None  # type: ignore


class OnnxAudioEmotion:
    def __init__(self, model_path: str, labels: list[str] = SER8_LABELS):
        path = Path(model_path).expanduser()
        self.model_path = os.fspath(path)
        self.labels = list(labels)
        self.sess = ort_session(self.model_path)
        input_meta = self.sess.get_inputs()[0]
        self._input_name = input_meta.name
        shape = input_meta.shape or []
        self._input_rank = len(shape)
        get_providers = getattr(self.sess, "get_providers", None)
        providers = get_providers() if callable(get_providers) else ["CPUExecutionProvider"]
        self._providers = list(providers)
        logger.info(
            "Audio SER ONNX ready via providers=%s path=%s",
            self._providers,
            self.model_path,
        )

    def _as_waveform_input(self, y: np.ndarray) -> dict[str, np.ndarray]:
        arr = y.astype(np.float32)
        if self._input_rank <= 1:
            return {self._input_name: arr}
        if self._input_rank == 2:
            return {self._input_name: arr[None, :]}
        if self._input_rank == 3:
            return {self._input_name: arr[None, None, :]}
        raise RuntimeError("Waveform input unsupported for rank>=4")

    def _as_mel_input(self, y: np.ndarray) -> dict[str, np.ndarray] | None:
        if librosa is None:
            return None
        sr = target_sample_rate()
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
        x = mel_db.astype(np.float32)[None, None, :, :]
        return {self._input_name: x}

    def __call__(self, y: np.ndarray, sr: int) -> tuple[str, dict[str, float]]:
        y = ensure_16k_mono(y, sr)
        y = trim_max_len(y, sr=target_sample_rate(), max_seconds=20.0)
        out = None
        if self._input_rank < 4:
            try:
                inputs = self._as_waveform_input(y)
                out = self.sess.run(None, inputs)
            except Exception:  # noqa: BLE001 - onnxruntime errors vary by build
                out = None
        if out is None:
            mel_inputs = self._as_mel_input(y)
            if mel_inputs is None:
                raise RuntimeError("Mel spectrogram path unavailable (librosa missing)")
            out = self.sess.run(None, mel_inputs)

        logits = np.asarray(out[0], dtype=np.float32).reshape(-1)
        probs = np.exp(logits - np.max(logits))
        probs = probs / (np.sum(probs) + 1e-6)
        label_count = len(self.labels)
        if probs.shape[0] != label_count:
            logger.warning(
                "Audio SER logits mismatch (got=%s expected=%s) for %s",
                probs.shape[0],
                label_count,
                self.model_path,
            )
            adjusted = np.zeros(label_count, dtype=np.float32)
            limit = min(label_count, probs.shape[0])
            adjusted[:limit] = probs[:limit]
            probs = adjusted / (np.sum(adjusted) + 1e-6)
        full = {self.labels[i]: float(probs[i]) for i in range(label_count)}
        scores = normalize_scores(full, self.labels)
        top_label = max(scores.items(), key=lambda kv: kv[1])[0]
        return top_label, scores


class SpeechEmotionAnalyzer:
    def __init__(
        self,
        *,
        onnx_path: str,
        labels: Sequence[str] | None = None,
        record_issue: IssueRecorder | None = None,
    ) -> None:
        self._onnx_path = os.fspath(Path(onnx_path).expanduser())
        self._labels = list(labels or SER8_LABELS)
        self._record_issue: IssueRecorder = record_issue or (lambda _: None)

        self._model: Callable[[np.ndarray, int], tuple[str, dict[str, float]]] | None = None
        self._attempted = False

    def _ensure_model(self) -> None:
        if self._model is not None or self._attempted:
            return
        self._attempted = True
        try:
            self._model = OnnxAudioEmotion(self._onnx_path, labels=self._labels)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Audio SER ONNX unavailable: %s", exc)
            self._record_issue(f"Speech emotion ONNX unavailable at {self._onnx_path}")
            self._model = None

    def analyze(self, wav: np.ndarray | None, sr: int | None) -> SpeechEmotionResult:
        if wav is None or sr is None:
            return default_speech_result()

        self._ensure_model()
        if self._model is None:
            self._record_issue("Speech emotion model unavailable; using neutral distribution")
            return default_speech_result()

        try:
            top_label, raw_scores = self._model(np.asarray(wav, dtype=np.float32), int(sr))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Speech emotion inference failed: %s", exc)
            self._record_issue(f"Speech emotion inference failed: {exc}")
            return default_speech_result()

        normalized = normalize_scores(raw_scores, self._labels)
        low_conf = ser_low_confidence(normalized)
        if top_label not in normalized:
            top_label = max(normalized.items(), key=lambda kv: kv[1])[0]
        return SpeechEmotionResult(top=top_label, scores=normalized, low_confidence=low_conf)


__all__ = [
    "SpeechEmotionAnalyzer",
    "OnnxAudioEmotion",
]
