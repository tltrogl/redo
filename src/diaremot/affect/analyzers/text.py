from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .common import (
    GOEMOTIONS_LABELS,
    IssueRecorder,
    TextEmotionResult,
    default_text_result,
    normalize_scores,
    ort_session,
    softmax,
    topk_distribution,
)

logger = logging.getLogger(__name__)


class OnnxTextEmotion:
    def __init__(
        self,
        model_path: str,
        labels: list[str] = GOEMOTIONS_LABELS,
        *,
        tokenizer_source: str | os.PathLike[str] | None = None,
        disable_downloads: bool = False,
    ):
        self.labels = labels
        self.sess = ort_session(model_path)
        self.tokenizer = self._load_tokenizer(
            model_path,
            tokenizer_source=tokenizer_source,
            disable_downloads=disable_downloads,
        )

    def _load_tokenizer(
        self,
        model_path: str,
        *,
        tokenizer_source: str | os.PathLike[str] | None,
        disable_downloads: bool,
    ):
        from transformers import AutoTokenizer  # type: ignore

        candidates: list[tuple[str, dict[str, object]]] = []
        errors: list[str] = []

        if tokenizer_source:
            local_dir = Path(tokenizer_source).expanduser()
        else:
            local_dir = Path(model_path).expanduser().parent

        local_dir_str = os.fspath(local_dir)
        candidates.append((local_dir_str, {"local_files_only": True}))
        if not disable_downloads:
            candidates.append((local_dir_str, {"local_files_only": False}))
            candidates.append(("SamLowe/roberta-base-go_emotions", {}))

        for identifier, kwargs in candidates:
            try:
                return AutoTokenizer.from_pretrained(identifier, **kwargs)
            except Exception as exc:  # noqa: BLE001 - HF backend specific
                errors.append(f"{identifier}: {exc}")

        details = "; ".join(errors)
        raise RuntimeError(
            "Unable to load text emotion tokenizer; attempted candidates: " + details
        )

    def __call__(self, text: str) -> dict[str, float]:
        enc = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        inputs = {self.sess.get_inputs()[0].name: enc["input_ids"].astype(np.int64)}

        if len(self.sess.get_inputs()) > 1 and "attention_mask" in enc:
            inputs[self.sess.get_inputs()[1].name] = enc["attention_mask"].astype(np.int64)
        if len(self.sess.get_inputs()) > 2 and "token_type_ids" in enc:
            inputs[self.sess.get_inputs()[2].name] = enc["token_type_ids"].astype(np.int64)

        out = self.sess.run(None, inputs)
        logits = out[0]
        if logits.ndim == 2:
            logits = logits[0]
        probs = softmax(logits.astype(np.float32))
        return {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}


class HfTextEmotionFallback:
    def __init__(self):
        pipeline = _maybe_import_transformers_pipeline()
        if pipeline is None:
            raise RuntimeError("transformers pipeline() unavailable for fallback")
        self.pipe = pipeline(
            task="text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            truncation=True,
        )

    def __call__(self, text: str) -> dict[str, float]:
        out = self.pipe(text)[0]
        full = {d["label"].lower(): float(d["score"]) for d in out}
        for lab in GOEMOTIONS_LABELS:
            full.setdefault(lab, 0.0)
        arr = np.array([full[lab] for lab in GOEMOTIONS_LABELS], dtype=np.float32)
        arr = arr / (arr.sum() + 1e-8)
        return {lab: float(arr[i]) for i, lab in enumerate(GOEMOTIONS_LABELS)}


def _maybe_import_transformers_pipeline():
    try:
        from transformers import pipeline  # type: ignore

        return pipeline
    except Exception:  # pragma: no cover - optional dependency
        return None


class TextEmotionAnalyzer:
    def __init__(
        self,
        *,
        onnx_path: str,
        model_dir: str,
        disable_downloads: bool,
        record_issue: IssueRecorder | None = None,
    ) -> None:
        self._onnx_path = os.fspath(Path(onnx_path).expanduser())
        self._model_dir = os.fspath(Path(model_dir).expanduser())
        self._disable_downloads = disable_downloads
        self._record_issue: IssueRecorder = record_issue or (lambda _: None)

        self._onnx_model: Callable[[str], dict[str, float]] | None = None
        self._fallback_model: Callable[[str], dict[str, float]] | None = None
        self._attempted_fallback = False

    def _ensure_backend(self) -> None:
        if self._onnx_model is not None:
            return
        try:
            self._onnx_model = OnnxTextEmotion(
                self._onnx_path,
                tokenizer_source=self._model_dir,
                disable_downloads=self._disable_downloads,
            )
            return
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Text emotion ONNX unavailable: %s", exc)
            self._record_issue(f"Text emotion ONNX missing under {self._model_dir}")
            self._onnx_model = None

        if self._disable_downloads or self._attempted_fallback:
            return

        self._attempted_fallback = True
        try:
            self._fallback_model = HfTextEmotionFallback()
            logger.warning("Using HuggingFace fallback for text emotion.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("HF fallback unavailable: %s", exc)
            self._fallback_model = None
            self._record_issue("Text emotion fallback unavailable; outputs neutral")

    def analyze(self, text: str) -> TextEmotionResult:
        clean = (text or "").strip()
        if not clean:
            return default_text_result()

        self._ensure_backend()
        backend = self._onnx_model or self._fallback_model
        if backend is None:
            self._record_issue("Text emotion model unavailable; using neutral distribution")
            return default_text_result()

        try:
            raw_scores = backend(clean)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Text emotion inference failed: %s", exc)
            self._record_issue(f"Text emotion inference failed: {exc}")
            return default_text_result()

        normalized = normalize_scores(raw_scores, GOEMOTIONS_LABELS)
        top5 = topk_distribution(normalized, k=5)
        return TextEmotionResult(top5=top5, full=normalized)


__all__ = [
    "TextEmotionAnalyzer",
    "OnnxTextEmotion",
    "HfTextEmotionFallback",
]
