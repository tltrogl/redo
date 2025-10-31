from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..intent_defaults import INTENT_LABELS_DEFAULT
from ..io.onnx_utils import create_onnx_session
from .analyzers.common import (
    GOEMOTIONS_LABELS,
    SER8_LABELS,
    EmotionOutputs,
    IntentResult,
    SpeechEmotionResult,
    TextEmotionResult,
    VadEmotionResult,
    default_intent_result,
    normalize_backend,
    normalize_intent_label,
    resolve_component_dir,
    resolve_model_dir,
    select_first_existing,
)
from .analyzers.intent import IntentAnalyzer, resolve_intent_model_dir
from .analyzers.speech import SpeechEmotionAnalyzer
from .analyzers.text import TextEmotionAnalyzer
from .analyzers.vad import VadEmotionAnalyzer


class EmotionAnalyzer:
    """High-level orchestration for affect analysis across text, audio, and VAD."""

    def __init__(
        self,
        model_dir: str | None = None,
        disable_downloads: bool | None = None,
        *,
        text_model_dir: str | None = None,
        ser_model_dir: str | None = None,
        vad_model_dir: str | None = None,
    ) -> None:
        base_dir = Path(model_dir).expanduser() if model_dir else resolve_model_dir()
        self.model_dir = str(base_dir)
        self.disable_downloads = bool(disable_downloads or False)
        self.issues: list[str] = []

        def _record_issue(message: str) -> None:
            if message not in self.issues:
                self.issues.append(message)

        self._record_issue = _record_issue

        self.text_model_dir = resolve_component_dir(
            text_model_dir, "DIAREMOT_TEXT_EMO_MODEL_DIR", "text_emotions"
        )
        self.ser_model_dir = resolve_component_dir(
            ser_model_dir, "AFFECT_SER_MODEL_DIR", "affect", "ser8"
        )
        self.vad_model_dir = resolve_component_dir(
            vad_model_dir, "AFFECT_VAD_DIM_MODEL_DIR", "affect", "vad_dim"
        )

        self.path_text_onnx = os.fspath(
            select_first_existing(
                self.text_model_dir,
                (
                    "model.int8.onnx",
                    "model.onnx",
                    "roberta-base-go_emotions.onnx",
                ),
            )
        )
        self.path_ser8_onnx = os.fspath(
            select_first_existing(
                self.ser_model_dir,
                (
                    "model.int8.onnx",
                    "ser8.int8.onnx",
                    "model.onnx",
                    "ser_8class.onnx",
                ),
            )
        )
        self.path_vad_onnx = os.fspath(
            select_first_existing(
                self.vad_model_dir,
                ("model.onnx", "vad_model.onnx"),
            )
        )

        env_ser = os.getenv("DIAREMOT_SER_ONNX")
        if env_ser:
            self.path_ser8_onnx = os.fspath(Path(env_ser).expanduser())

        self._text_analyzer = TextEmotionAnalyzer(
            onnx_path=self.path_text_onnx,
            model_dir=os.fspath(self.text_model_dir),
            disable_downloads=self.disable_downloads,
            record_issue=self._record_issue,
        )
        self._speech_analyzer = SpeechEmotionAnalyzer(
            onnx_path=self.path_ser8_onnx,
            labels=SER8_LABELS,
            record_issue=self._record_issue,
        )
        self._vad_analyzer = VadEmotionAnalyzer(
            onnx_path=self.path_vad_onnx,
            record_issue=self._record_issue,
        )

    def analyze_text(self, text: str) -> TextEmotionResult:
        return self._text_analyzer.analyze(text)

    def analyze_audio(self, y: np.ndarray | None, sr: int | None) -> SpeechEmotionResult:
        return self._speech_analyzer.analyze(y, sr)

    def analyze_vad_emotion(self, y: np.ndarray | None, sr: int | None) -> VadEmotionResult:
        return self._vad_analyzer.analyze(y, sr)

    def _make_affect_hint(self, vad: VadEmotionResult, intent_top: str) -> str:
        intent = normalize_intent_label(intent_top)
        v = float(vad.valence)
        a = float(vad.arousal)
        if not np.isfinite(v) or not np.isfinite(a):
            return f"neutral-{intent}"
        if a > 0.55 and v < 0.0:
            return "agitated-negative"
        if a < 0.30 and v > 0.25:
            return "calm-positive"
        if v > 0.50:
            return f"bright-{intent}"
        if v < -0.35:
            return f"low-{intent}"
        return f"neutral-{intent}"

    def _build_affect_payload(
        self,
        *,
        text_res: TextEmotionResult,
        speech_res: SpeechEmotionResult,
        vad_res: VadEmotionResult,
        intent_res: IntentResult,
    ) -> dict[str, Any]:
        from ..pipeline.outputs import default_affect

        payload = default_affect()
        payload["vad"] = {
            "valence": vad_res.valence,
            "arousal": vad_res.arousal,
            "dominance": vad_res.dominance,
        }
        payload["speech_emotion"] = {
            "top": speech_res.top,
            "scores_8class": dict(speech_res.scores),
            "low_confidence_ser": speech_res.low_confidence,
        }
        payload["text_emotions"] = {
            "top5": [dict(item) for item in text_res.top5],
            "full_28class": dict(text_res.full),
        }
        payload["intent"] = {
            "top": intent_res.top,
            "top3": [dict(item) for item in intent_res.top3],
        }
        payload["affect_hint"] = self._make_affect_hint(vad_res, intent_res.top)
        return payload

    def _analyze_components(
        self,
        *,
        wav: np.ndarray | None,
        sr: int | None,
        text: str,
    ) -> tuple[TextEmotionResult, SpeechEmotionResult, VadEmotionResult]:
        text_res = self.analyze_text(text)
        speech_res = self.analyze_audio(wav, sr)
        vad_res = self.analyze_vad_emotion(wav, sr)
        return text_res, speech_res, vad_res

    def analyze(
        self,
        *,
        wav: np.ndarray | None,
        sr: int | None,
        text: str,
    ) -> dict[str, Any]:
        text_res, speech_res, vad_res = self._analyze_components(wav=wav, sr=sr, text=text)
        intent_res = default_intent_result()
        return self._build_affect_payload(
            text_res=text_res,
            speech_res=speech_res,
            vad_res=vad_res,
            intent_res=intent_res,
        )

    def analyze_segment(self, text: str, audio_wave: np.ndarray | None, sr: int | None) -> EmotionOutputs:
        payload = self.analyze(wav=audio_wave, sr=sr, text=text or "")
        return EmotionOutputs.from_affect(payload)


class EmotionIntentAnalyzer(EmotionAnalyzer):
    def __init__(
        self,
        *,
        text_emotion_model: str = "SamLowe/roberta-base-go_emotions",
        intent_labels: Sequence[str] | None = None,
        affect_backend: str | None = None,
        affect_text_model_dir: str | None = None,
        affect_ser_model_dir: str | None = None,
        affect_vad_model_dir: str | None = None,
        affect_intent_model_dir: str | None = None,
        analyzer_threads: int | None = None,
        disable_downloads: bool | None = None,
        model_dir: str | None = None,
    ) -> None:
        super().__init__(
            model_dir=model_dir,
            disable_downloads=disable_downloads,
            text_model_dir=affect_text_model_dir,
            ser_model_dir=affect_ser_model_dir,
            vad_model_dir=affect_vad_model_dir,
        )

        self.text_emotion_model = text_emotion_model
        labels = intent_labels or INTENT_LABELS_DEFAULT
        self.intent_labels: list[str] = [str(label) for label in labels]
        self.affect_backend = normalize_backend(affect_backend)
        self.analyzer_threads = analyzer_threads or 1

        self.affect_text_model_dir = os.fspath(self.text_model_dir)
        self.affect_ser_model_dir = os.fspath(self.ser_model_dir)
        self.affect_vad_model_dir = os.fspath(self.vad_model_dir)

        self.affect_intent_model_dir = resolve_intent_model_dir(affect_intent_model_dir)

        self._intent_analyzer = IntentAnalyzer(
            labels=self.intent_labels,
            backend=self.affect_backend,
            model_dir=self.affect_intent_model_dir,
            analyzer_threads=self.analyzer_threads,
            disable_downloads=self.disable_downloads,
            record_issue=self._record_issue,
        )

    def analyze(
        self,
        *,
        wav: np.ndarray | None,
        sr: int | None,
        text: str,
    ) -> dict[str, Any]:
        text_res, speech_res, vad_res = self._analyze_components(wav=wav, sr=sr, text=text)
        intent_res = self._intent_analyzer.infer(text)
        return self._build_affect_payload(
            text_res=text_res,
            speech_res=speech_res,
            vad_res=vad_res,
            intent_res=intent_res,
        )


__all__ = (
    "EmotionAnalyzer",
    "EmotionOutputs",
    "GOEMOTIONS_LABELS",
    "SER8_LABELS",
    "EmotionIntentAnalyzer",
    "create_onnx_session",
)
