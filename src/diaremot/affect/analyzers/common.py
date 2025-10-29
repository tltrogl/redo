from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...pipeline.runtime_env import DEFAULT_MODELS_ROOT

__all__ = [
    "EmotionOutputs",
    "IntentResult",
    "TextEmotionResult",
    "SpeechEmotionResult",
    "VadEmotionResult",
    "GOEMOTIONS_LABELS",
    "SER8_LABELS",
    "default_text_result",
    "default_speech_result",
    "default_vad_result",
    "default_intent_result",
    "IssueRecorder",
    "json_dumps",
    "softmax",
    "topk_distribution",
    "target_sample_rate",
    "ensure_16k_mono",
    "trim_max_len",
    "normalize_scores",
    "ser_low_confidence",
    "normalize_intent_label",
    "resolve_model_dir",
    "resolve_component_dir",
    "select_first_existing",
    "normalize_backend",
    "resolve_component_aliases",
    "ort_session",
]

def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@dataclass
class EmotionOutputs:
    """Serialized affect outputs for storage layers (CSV/JSON)."""

    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    emotion_top: str = "neutral"
    emotion_scores_json: str = json_dumps({})
    low_confidence_ser: bool = True
    text_emotions_top5_json: str = json_dumps([])
    text_emotions_full_json: str = json_dumps({})
    intent_top: str = "status_update"
    intent_top3_json: str = json_dumps([])
    affect_hint: str = "neutral-status"

    @classmethod
    def from_affect(cls, payload: Mapping[str, Any]) -> EmotionOutputs:
        def _safe_float(value: Any) -> float:
            try:
                num = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(num):
                return 0.0
            return float(num)

        vad = payload.get("vad", {}) if isinstance(payload, Mapping) else {}
        ser = payload.get("speech_emotion", {}) if isinstance(payload, Mapping) else {}
        text = payload.get("text_emotions", {}) if isinstance(payload, Mapping) else {}
        intent = payload.get("intent", {}) if isinstance(payload, Mapping) else {}
        return cls(
            valence=_safe_float(vad.get("valence", 0.0)),
            arousal=_safe_float(vad.get("arousal", 0.0)),
            dominance=_safe_float(vad.get("dominance", 0.0)),
            emotion_top=str(ser.get("top", "neutral")),
            emotion_scores_json=json_dumps(ser.get("scores_8class", {})),
            low_confidence_ser=bool(ser.get("low_confidence_ser", False)),
            text_emotions_top5_json=json_dumps(text.get("top5", [])),
            text_emotions_full_json=json_dumps(text.get("full_28class", {})),
            intent_top=str(intent.get("top", "status_update")),
            intent_top3_json=json_dumps(intent.get("top3", [])),
            affect_hint=str(payload.get("affect_hint", "neutral-status")),
        )

    def to_affect(self) -> dict[str, Any]:
        def _loads(data: str, default: Any) -> Any:
            try:
                return json.loads(data)
            except (TypeError, json.JSONDecodeError):
                return default

        return {
            "vad": {
                "valence": self.valence,
                "arousal": self.arousal,
                "dominance": self.dominance,
            },
            "speech_emotion": {
                "top": self.emotion_top,
                "scores_8class": _loads(self.emotion_scores_json, {}),
                "low_confidence_ser": bool(self.low_confidence_ser),
            },
            "text_emotions": {
                "top5": _loads(self.text_emotions_top5_json, []),
                "full_28class": _loads(self.text_emotions_full_json, {}),
            },
            "intent": {
                "top": self.intent_top,
                "top3": _loads(self.intent_top3_json, []),
            },
            "affect_hint": self.affect_hint,
        }


@dataclass
class TextEmotionResult:
    top5: list[dict[str, float]]
    full: dict[str, float]


@dataclass
class SpeechEmotionResult:
    top: str
    scores: dict[str, float]
    low_confidence: bool


@dataclass
class VadEmotionResult:
    valence: float
    arousal: float
    dominance: float


@dataclass
class IntentResult:
    top: str
    top3: list[dict[str, float]]


GOEMOTIONS_LABELS: list[str] = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

SER8_LABELS: list[str] = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


IssueRecorder = Callable[[str], None]


def default_text_result() -> TextEmotionResult:
    base = {label: 0.0 for label in GOEMOTIONS_LABELS}
    base["neutral"] = 1.0
    return TextEmotionResult(
        top5=[{"label": "neutral", "score": 1.0}],
        full=base,
    )


def default_speech_result() -> SpeechEmotionResult:
    base = {label: 0.0 for label in SER8_LABELS}
    base["neutral"] = 1.0
    return SpeechEmotionResult(top="neutral", scores=base, low_confidence=True)


def default_vad_result() -> VadEmotionResult:
    return VadEmotionResult(valence=0.0, arousal=0.0, dominance=0.0)


def default_intent_result() -> IntentResult:
    return IntentResult(
        top="status_update",
        top3=[
            {"label": "status_update", "score": 1.0},
            {"label": "small_talk", "score": 0.0},
            {"label": "opinion", "score": 0.0},
        ],
    )


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.asarray([], dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    denom = np.sum(e)
    if denom <= 0:
        return np.asarray([], dtype=np.float32)
    return e / denom


def topk_distribution(scores: Mapping[str, float], *, k: int = 5) -> list[dict[str, float]]:
    items = [
        (str(label), float(score))
        for label, score in scores.items()
        if isinstance(score, (int, float)) and math.isfinite(float(score))
    ]
    items.sort(key=lambda item: item[1], reverse=True)
    limited = items[: max(0, min(k, len(items)))]
    return [
        {"label": label, "score": float(score)}
        for label, score in limited
    ]


def target_sample_rate() -> int:
    return 16000


def ensure_16k_mono(y: np.ndarray, sr: int) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=-1)
    target_sr = target_sample_rate()
    try:
        import librosa  # type: ignore
    except ImportError:  # pragma: no cover
        librosa = None  # type: ignore
    if librosa is not None and sr != target_sr:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    return arr.astype(np.float32)


def trim_max_len(y: np.ndarray, *, sr: int, max_seconds: float = 20.0) -> np.ndarray:
    if max_seconds <= 0:
        return y
    limit = int(max_seconds * sr)
    if limit <= 0 or y.size <= limit:
        return y
    return np.asarray(y[:limit], dtype=np.float32)


def entropy(probs: Sequence[float]) -> float:
    arr = np.asarray([float(p) for p in probs if float(p) > 0.0], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(-np.sum(arr * np.log(arr)))


def normalized_entropy(probs: Mapping[str, float]) -> float:
    values = [float(v) for v in probs.values() if float(v) > 0.0]
    if not values:
        return 0.0
    ent = entropy(values)
    max_ent = math.log(len(values)) if len(values) > 1 else 0.0
    if max_ent <= 0.0:
        return 0.0
    return float(ent / max_ent)


def top_margin(scores: Mapping[str, float]) -> float:
    values = [float(v) for v in scores.values() if math.isfinite(float(v))]
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    arr = np.asarray(values, dtype=np.float32)
    idx = np.argpartition(arr, -2)[-2:]
    top_two = arr[idx]
    return float(np.max(top_two) - np.min(top_two))


def canonical_label(label: str, labels: Sequence[str]) -> str:
    if not label:
        return labels[0] if labels else ""
    lower = label.lower()
    for candidate in labels:
        if candidate.lower() == lower:
            return candidate
    return labels[0] if labels else label


def normalize_scores(scores: Mapping[str, float], labels: Sequence[str]) -> dict[str, float]:
    base: dict[str, float] = {label: 0.0 for label in labels}
    for key, value in scores.items():
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score) or score < 0.0:
            continue
        canonical = canonical_label(str(key), labels)
        base[canonical] = score
    total = float(sum(base.values()))
    if total > 0.0:
        return {label: float(score / total) for label, score in base.items()}
    if labels:
        base[labels[0]] = 1.0
    return base


def ser_low_confidence(scores: Mapping[str, float]) -> bool:
    if not scores:
        return True
    values = [float(v) for v in scores.values() if math.isfinite(float(v))]
    if not values:
        return True
    top_score = max(values)
    margin = top_margin(scores)
    entropy_val = normalized_entropy({k: float(v) for k, v in scores.items()})
    return top_score < 0.55 or margin < 0.18 or entropy_val > 0.85


def normalize_intent_label(label: str) -> str:
    clean = (label or "status_update").strip().lower().replace(" ", "_")
    if clean == "status_update":
        clean = "status"
    clean = clean.replace("_", "-")
    return clean or "status"


def resolve_model_dir() -> Path:
    d = os.environ.get("DIAREMOT_MODEL_DIR")
    if d:
        return Path(d).expanduser()
    return Path(DEFAULT_MODELS_ROOT)


_COMPONENT_ALIASES: dict[tuple[str, ...], tuple[tuple[str, ...], ...]] = {
    ("text_emotions",): (("goemotions-onnx",),),
    ("affect", "ser8"): (("ser8-onnx",), ("ser8",)),
    ("affect", "vad_dim"): (("VAD_dim",), ("vad_dim",)),
    ("affect", "sed_panns"): (("panns",),),
    ("intent",): (("bart",),),
}


def resolve_component_aliases() -> dict[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    return dict(_COMPONENT_ALIASES)


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = os.fspath(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _descend_casefold(base: Path, parts: Sequence[str]) -> list[Path]:
    nodes: list[Path] = [base]
    for part in parts:
        next_nodes: list[Path] = []
        folded = part.casefold()
        for node in nodes:
            next_nodes.append(node / part)
            if not node.exists():
                continue
            try:
                children = list(node.iterdir())
            except OSError:
                continue
            for child in children:
                if child.name.casefold() == folded:
                    next_nodes.append(child)
        nodes = _dedupe_paths(next_nodes)
    return nodes


def resolve_component_dir(
    cli_value: str | None, env_key: str, *default_subpath: str
) -> Path:
    candidates: list[Path] = []
    if cli_value:
        candidates.append(Path(cli_value).expanduser())
    env_value = os.getenv(env_key)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    model_root = resolve_model_dir()
    if default_subpath:
        candidates.extend(_descend_casefold(model_root, tuple(default_subpath)))
        alias_paths = _COMPONENT_ALIASES.get(tuple(default_subpath), ())
        for alias in alias_paths:
            candidates.extend(_descend_casefold(model_root, alias))
    else:
        candidates.append(model_root)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return candidates[0]


def select_first_existing(directory: Path, names: Sequence[str]) -> Path:
    if directory.is_file() or directory.suffix.lower() == ".onnx":
        return directory
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return directory / names[0]


def normalize_backend(value: str | None) -> str:
    if not value:
        return "auto"
    normalized = value.lower()
    if normalized not in {"auto", "onnx", "torch"}:
        return "auto"
    return normalized


def ort_session(path: str):
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"onnxruntime not available: {exc}") from exc

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=sess_options, providers=providers)
