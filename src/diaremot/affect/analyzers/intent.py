from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...io.onnx_utils import create_onnx_session
from ..intent_defaults import INTENT_LABELS_DEFAULT
from .common import IntentResult, IssueRecorder, normalize_backend, softmax

logger = logging.getLogger(__name__)

DEFAULT_INTENT_MODEL = "facebook/bart-large-mnli"


def _intent_dir_has_assets(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    has_onnx = any(
        (path / name).exists() for name in ("model_uint8.onnx", "model_int8.onnx", "model.onnx")
    )

    cfg_path = path / "config.json"
    if not cfg_path.exists():
        return False

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        model_type = cfg.get("model_type")
        if not isinstance(model_type, str) or not model_type:
            return False
    except Exception:
        return False

    tokenizer_present = (path / "tokenizer.json").exists() or (
        (path / "vocab.json").exists() and (path / "merges.txt").exists()
    )
    if not tokenizer_present:
        return False

    if has_onnx:
        return True

    return True


def _intent_candidate_dirs(explicit: str | None) -> Iterable[Path]:
    candidates: list[Path] = []

    def _add(candidate: str | Path | None) -> None:
        if not candidate:
            return
        candidates.append(Path(candidate).expanduser())

    _add(explicit)
    _add(os.getenv("DIAREMOT_INTENT_MODEL_DIR"))

    model_root = os.getenv("DIAREMOT_MODEL_DIR")
    if model_root:
        root = Path(model_root).expanduser()
        _add(root)
        _add(root / "intent")
        _add(root / "bart")
        _add(root / "bart-large-mnli")
        _add(root / "facebook" / "bart-large-mnli")
        _add(root / "bart" / "facebook" / "bart-large-mnli")

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def resolve_intent_model_dir(explicit: str | None) -> str | None:
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists() and _intent_dir_has_assets(path):
            return str(path)
        return None

    env_override = os.getenv("DIAREMOT_INTENT_MODEL_DIR")
    if env_override:
        env_path = Path(env_override).expanduser()
        if _intent_dir_has_assets(env_path):
            return str(env_path)
        return None

    for candidate in _intent_candidate_dirs(None):
        if candidate.exists() and _intent_dir_has_assets(candidate):
            return str(candidate)

    return None


def _find_label_index(id2label: dict[int, str], target: str) -> int | None:
    target_lower = target.lower()
    for idx, label in id2label.items():
        if label.lower() == target_lower:
            return int(idx)
    return None


def _maybe_import_transformers_pipeline():
    try:
        from transformers import pipeline  # type: ignore

        return pipeline
    except Exception:  # pragma: no cover - optional dependency
        return None


class IntentAnalyzer:
    def __init__(
        self,
        *,
        labels: Sequence[str] | None = None,
        backend: str | None = None,
        model_dir: str | None = None,
        analyzer_threads: int | None = None,
        disable_downloads: bool = False,
        record_issue: IssueRecorder | None = None,
    ) -> None:
        self.labels: list[str] = [str(label) for label in (labels or INTENT_LABELS_DEFAULT)]
        self.backend = normalize_backend(backend)
        self.model_dir = model_dir
        self.analyzer_threads = analyzer_threads or 1
        self.disable_downloads = disable_downloads
        self._record_issue: IssueRecorder = record_issue or (lambda _: None)

        self._intent_session: object | None = None
        self._intent_tokenizer: Callable[..., dict[str, np.ndarray]] | None = None
        self._intent_config: object | None = None
        self._intent_pipeline: Callable[..., object] | None = None
        self._intent_entail_idx: int | None = None
        self._intent_contra_idx: int | None = None
        self._intent_hypothesis_template: str = "This example is {}."

    def _lazy_prepare(self) -> None:
        if self.backend == "onnx":
            self._ensure_intent_onnx(strict=True)
        elif self.backend == "torch":
            self._ensure_intent_pipeline()
        else:
            if not self._ensure_intent_onnx(strict=False):
                self._ensure_intent_pipeline()

    def _select_onnx_model(self, model_dir: Path) -> Path | None:
        for name in ("model_int8.onnx", "model_uint8.onnx", "model.onnx"):
            candidate = model_dir / name
            if candidate.exists():
                return candidate
        remaining = list(model_dir.glob("*.onnx"))
        return remaining[0] if remaining else None

    def _ensure_intent_onnx(self, *, strict: bool) -> bool:
        if self._intent_session is not None and self._intent_tokenizer is not None:
            return True

        if not self.model_dir:
            if strict:
                logger.warning("Intent ONNX backend requested but no model directory is configured")
            self._record_issue("Intent model directory not configured")
            return False

        model_dir = Path(self.model_dir)
        model_path = self._select_onnx_model(model_dir)
        if model_path is None:
            if strict:
                logger.warning("Intent ONNX backend missing model.onnx in %s", model_dir)
            self._record_issue(f"Intent ONNX model missing under {model_dir}")
            return False

        try:
            self._intent_session = create_onnx_session(model_path, threads=self.analyzer_threads)
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.warning("Intent ONNX session unavailable: %s", exc)
            self._intent_session = None
            self._record_issue(f"Intent ONNX session unavailable: {exc}")
            return False

        try:
            from transformers import AutoConfig, AutoTokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            logger.warning("transformers unavailable for intent tokenizer: %s", exc)
            self._intent_session = None
            self._record_issue("Transformers package missing for intent tokenizer")
            return False

        try:
            self._intent_tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, local_files_only=True
            )
            self._intent_config = AutoConfig.from_pretrained(self.model_dir, local_files_only=True)
        except Exception as exc_local:  # noqa: BLE001
            logger.warning("Intent tokenizer/config not found locally: %s", exc_local)
            self._record_issue(
                f"Intent tokenizer/config unavailable under {self.model_dir}: {exc_local}"
            )
            if self.disable_downloads:
                self._intent_session = None
                self._intent_tokenizer = None
                self._intent_config = None
                return False
            try:
                self._intent_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_INTENT_MODEL)
                self._intent_config = AutoConfig.from_pretrained(DEFAULT_INTENT_MODEL)
                logger.warning(
                    "Using HuggingFace fallback for intent tokenizer/config: %s",
                    DEFAULT_INTENT_MODEL,
                )
            except Exception as exc_remote:  # noqa: BLE001
                logger.warning("Intent fallback tokenizer/config unavailable: %s", exc_remote)
                self._intent_session = None
                self._intent_tokenizer = None
                self._intent_config = None
                self._record_issue(f"Intent tokenizer/config fallback failed: {exc_remote}")
                return False

        id2label_raw = getattr(self._intent_config, "id2label", {})
        if isinstance(id2label_raw, dict):
            id2label = {int(k): str(v) for k, v in id2label_raw.items()}
        else:
            id2label = {int(idx): str(label) for idx, label in enumerate(id2label_raw)}
        self._intent_entail_idx = _find_label_index(id2label, "entailment")
        self._intent_contra_idx = _find_label_index(id2label, "contradiction")

        template = getattr(self._intent_config, "hypothesis_template", None)
        if isinstance(template, str) and "{}" in template:
            self._intent_hypothesis_template = template
        else:
            self._intent_hypothesis_template = "This example is {}."

        return True

    def _ensure_intent_pipeline(self) -> bool:
        if self._intent_pipeline is not None:
            return True
        pipeline = _maybe_import_transformers_pipeline()
        if pipeline is None:
            self._record_issue("Transformers pipeline unavailable for intent analysis")
            return False
        try:
            self._intent_pipeline = pipeline(
                task="zero-shot-classification",
                model=DEFAULT_INTENT_MODEL,
                multi_label=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Intent pipeline unavailable: %s", exc)
            self._intent_pipeline = None
            self._record_issue(f"Intent transformers pipeline unavailable: {exc}")
            return False
        return True

    def _intent_default_prediction(self) -> IntentResult:
        if not self.labels:
            return IntentResult(top="", top3=[])
        topn = min(3, len(self.labels))
        default_labels = self.labels[:topn]
        score = 1.0 / topn if topn else 0.0
        entries = [{"label": label, "score": score} for label in default_labels]
        return IntentResult(top=default_labels[0], top3=entries)

    def _infer_intent_with_onnx(self, text: str) -> IntentResult:
        if self._intent_session is None or self._intent_tokenizer is None:
            return self._intent_default_prediction()

        entail_idx = self._intent_entail_idx
        contra_idx = self._intent_contra_idx

        # Batch all hypotheses for this text
        hypotheses = [self._intent_hypothesis_template.format(label) for label in self.labels]
        texts = [text] * len(self.labels)

        encoded = self._intent_tokenizer(
            texts,
            hypotheses,
            return_tensors="np",
            truncation=True,
            padding=True,
        )
        inputs = {name: np.asarray(value, dtype=np.int64) for name, value in encoded.items()}

        # Run inference for all labels in one batch
        logits_batch = self._intent_session.run(None, inputs)[0]

        results: list[dict[str, Any]] = []
        for i, label in enumerate(self.labels):
            arr = np.array(logits_batch[i], dtype=np.float32).ravel()

            if (
                entail_idx is not None
                and contra_idx is not None
                and 0 <= entail_idx < arr.size
                and 0 <= contra_idx < arr.size
            ):
                pair = np.array([arr[contra_idx], arr[entail_idx]], dtype=np.float32)
                score = float(softmax(pair)[-1])
            elif entail_idx is not None and 0 <= entail_idx < arr.size:
                probs = softmax(arr)
                score = float(probs[entail_idx])
            else:
                probs = softmax(arr)
                score = float(np.max(probs))

            results.append({"label": label, "score": score})

        results.sort(key=lambda item: item["score"], reverse=True)
        top3 = results[: min(3, len(results))]
        top_label = top3[0]["label"] if top3 else ""
        return IntentResult(top=top_label, top3=top3)

    def _infer_intent_with_pipeline(self, text: str) -> IntentResult:
        if self._intent_pipeline is None:
            return self._intent_default_prediction()

        candidates = self.labels or ["other"]
        raw = self._intent_pipeline(text, candidate_labels=candidates, multi_label=True)
        entries: list[dict[str, float]]
        if isinstance(raw, dict) and "labels" in raw and "scores" in raw:
            entries = [
                {"label": str(label), "score": float(score)}
                for label, score in zip(raw["labels"], raw["scores"])
            ]
        else:
            entries = []
            for item in raw:
                if isinstance(item, dict):
                    label = str(item.get("label", ""))
                    score = float(item.get("score", 0.0))
                else:
                    label = str(getattr(item, "label", ""))
                    score = float(getattr(item, "score", 0.0))
                entries.append({"label": label, "score": score})

        entries.sort(key=lambda item: item["score"], reverse=True)
        top3 = entries[: min(3, len(entries))]
        top_label = top3[0]["label"] if top3 else ""
        return IntentResult(top=top_label, top3=top3)

    def infer(self, text: str) -> IntentResult:
        clean_text = (text or "").strip()
        if not clean_text:
            return self._intent_default_prediction()

        self._lazy_prepare()

        if self._intent_session is not None and self._intent_tokenizer is not None:
            try:
                return self._infer_intent_with_onnx(clean_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Intent ONNX inference failed: %s", exc)

        if self._intent_pipeline is not None:
            try:
                return self._infer_intent_with_pipeline(clean_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Intent pipeline inference failed: %s", exc)

        return self._intent_default_prediction()


__all__ = [
    "IntentAnalyzer",
    "resolve_intent_model_dir",
    "DEFAULT_INTENT_MODEL",
]
