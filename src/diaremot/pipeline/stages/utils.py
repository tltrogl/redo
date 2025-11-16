"""Utility helpers shared across pipeline stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np


def compute_audio_sha16(y: np.ndarray) -> str:
    try:
        arr = np.array(y, dtype=np.float32)
    except Exception:
        return hashlib.blake2s(b"", digest_size=16).hexdigest()

    if not isinstance(arr, np.ndarray):
        try:
            arr = np.array(arr, dtype=np.float32, copy=True)
        except Exception:
            return hashlib.blake2s(b"", digest_size=16).hexdigest()
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    return hashlib.blake2s(arr.tobytes(), digest_size=16).hexdigest()


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


def compute_pp_signature(pp_conf: Any) -> str:
    """Compute a deterministic string signature of preprocessing config."""
    keys = ["target_sr", "denoise", "loudness_mode"]
    sig: dict[str, Any] = {}
    for key in keys:
        try:
            sig[key] = getattr(pp_conf, key)
        except Exception:
            sig[key] = None
    return json.dumps(sig, sort_keys=True)


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
    except Exception:
        return None
    return None


def matches_pipeline_cache(
    payload: Mapping[str, Any] | None,
    *,
    version: Any = None,
    audio_sha16: str | None = None,
    pp_signature: Any = None,
    extra: Mapping[str, Any] | None = None,
    require_version: bool = True,
    require_audio_sha: bool = True,
    require_signature: bool = True,
) -> bool:
    """Validate cache metadata against expected pipeline identifiers."""

    if not payload:
        return False

    if require_version and (version is not None) and payload.get("version") != version:
        return False

    if require_audio_sha and audio_sha16 and payload.get("audio_sha16") != audio_sha16:
        return False

    if require_signature and (pp_signature is not None) and payload.get("pp_signature") != pp_signature:
        return False

    if extra:
        for key, expected in extra.items():
            if payload.get(key) != expected:
                return False

    return True


def build_cache_payload(
    *,
    version: Any,
    audio_sha16: str | None,
    pp_signature: Any,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose a cache payload with standard metadata fields."""

    payload: dict[str, Any] = {
        "version": version,
        "audio_sha16": audio_sha16,
        "pp_signature": pp_signature,
    }
    if extra:
        payload.update(extra)
    return payload


__all__ = [
    "atomic_write_json",
    "compute_audio_sha16",
    "compute_audio_sha16_from_file",
    "compute_pp_signature",
    "compute_sed_signature",
    "build_cache_payload",
    "matches_pipeline_cache",
    "read_json_safe",
]
