"""Utility helpers shared across pipeline stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def compute_audio_sha16(y: np.ndarray) -> str:
    hasher = hashlib.blake2s(digest_size=16)
    try:
        arr = np.asarray(y, dtype=np.float32)
    except Exception:
        hasher.update(b"")
        return hasher.hexdigest()

    if not isinstance(arr, np.ndarray):
        try:
            arr = np.asarray(arr, dtype=np.float32)
        except Exception:
            hasher.update(b"")
            return hasher.hexdigest()

    if arr.size == 0:
        hasher.update(b"")
        return hasher.hexdigest()

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    hasher.update(arr.view(np.uint8))
    return hasher.hexdigest()


def compute_audio_sha16_from_file(path: str) -> str:
    """Compute hash from raw file bytes (fast, no audio decode)."""
    try:
        hasher = hashlib.blake2s(digest_size=16)
        with open(path, "rb") as f:
            # Read in 1MB chunks
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def _normalise_signature_value(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(k): _normalise_signature_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_normalise_signature_value(v) for v in value]
    return value


def compute_pp_signature(pp_conf: Any) -> str:
    """Compute a deterministic string signature of preprocessing config."""

    if pp_conf is None:
        return json.dumps(None)

    if is_dataclass(pp_conf):
        payload: Any = asdict(pp_conf)
    elif isinstance(pp_conf, dict):
        payload = dict(pp_conf)
    elif hasattr(pp_conf, "__dict__"):
        payload = {
            key: getattr(pp_conf, key)
            for key in vars(pp_conf).keys()
            if not key.startswith("_")
        }
    else:
        payload = {}
        for key in dir(pp_conf):
            if key.startswith("_"):
                continue
            try:
                value = getattr(pp_conf, key)
            except Exception:
                continue
            if callable(value):
                continue
            payload[key] = value

    normalised = _normalise_signature_value(payload)
    return json.dumps(normalised, sort_keys=True)


def compute_sed_signature(cfg: Any) -> str:
    """Compute a deterministic signature for background SED configuration."""

    def _get(source: Any, key: str) -> Any:
        if source is None:
            return None
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    keys = [
        "enable_sed",
        "sed_mode",
        "sed_window_sec",
        "sed_hop_sec",
        "sed_enter",
        "sed_exit",
        "sed_min_dur",
        "sed_default_min_dur",
        "sed_merge_gap",
        "sed_classmap_csv",
        "sed_timeline_jsonl",
        "sed_median_k",
        "sed_batch_size",
        "sed_max_windows",
    ]
    sig: dict[str, Any] = {}
    for key in keys:
        sig[key] = _get(cfg, key)
    return json.dumps(sig, sort_keys=True)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json_safe(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        corrupt_path = path.with_suffix(path.suffix + ".corrupt")
        try:
            path.replace(corrupt_path)
        except Exception:
            try:
                path.unlink()
            except Exception:
                pass
        return None
    except Exception:
        return None
    return None


__all__ = [
    "atomic_write_json",
    "compute_audio_sha16",
    "compute_audio_sha16_from_file",
    "compute_pp_signature",
    "compute_sed_signature",
    "read_json_safe",
]
