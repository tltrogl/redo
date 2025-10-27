from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .logger import logger


class SpeakerRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._speakers: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._use_wrapped_format = False
        self._load()

    def _iso_now(self) -> str:
        return datetime.now(tz=UTC).isoformat(timespec="seconds")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and ("speakers" in data or "metadata" in data):
                self._speakers = {
                    k: v for k, v in (data.get("speakers") or {}).items() if isinstance(k, str)
                }
                self._metadata = {
                    k: v for k, v in data.items() if k != "speakers" and not k.startswith("_")
                }
                self._use_wrapped_format = True
            elif isinstance(data, dict):
                self._speakers = {k: v for k, v in data.items() if isinstance(k, str)}
            else:
                logger.warning("Registry load expected a JSON object at %s", self.path)
        except Exception as exc:
            logger.warning("Registry load failed: %s", exc)
            self._speakers = {}
            self._metadata = {}
            self._use_wrapped_format = False

    def _touch_metadata(self) -> None:
        if not (self._use_wrapped_format or self._metadata):
            return
        now = self._iso_now()
        if "created_at" not in self._metadata:
            self._metadata["created_at"] = now
        self._metadata["updated_at"] = now
        if "total_speakers" in self._metadata:
            self._metadata["total_speakers"] = len(self._speakers)
        meta_block = self._metadata.get("metadata")
        if isinstance(meta_block, dict):
            if "total_speakers" in meta_block or meta_block:
                meta_block["total_speakers"] = len(self._speakers)
                self._metadata["metadata"] = meta_block
        self._use_wrapped_format = True

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_suffix(".tmp")
            if self._use_wrapped_format or self._metadata:
                self._touch_metadata()
                payload = {**self._metadata, "speakers": self._speakers}
            else:
                payload = self._speakers
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temp_path.replace(self.path)
        except Exception as exc:
            logger.warning("Registry save failed: %s", exc)
            temp_path = self.path.with_suffix(".tmp")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def has(self, name: str) -> bool:
        return name in self._speakers

    def enroll(self, name: str, centroid: np.ndarray) -> None:
        stamp = self._iso_now()
        self._speakers[name] = {
            "centroid": centroid.tolist() if isinstance(centroid, np.ndarray) else centroid,
            "samples": 1,
            "last_seen": stamp,
        }

    def update_centroid(self, name: str, centroid: np.ndarray, alpha: float = 0.30) -> None:
        if not self.has(name):
            self.enroll(name, centroid)
            return
        existing = self._speakers.get(name) or {}
        old = np.asarray(existing.get("centroid", []), dtype=np.float32)
        if old.size == 0:
            old = np.asarray(centroid, dtype=np.float32)
        new = np.asarray(centroid, dtype=np.float32)
        ema = (1 - alpha) * old + alpha * new
        ema = ema / (np.linalg.norm(ema) + 1e-9)
        existing["centroid"] = ema.tolist()
        existing["samples"] = int(existing.get("samples", 1)) + 1
        existing["last_seen"] = self._iso_now()
        self._speakers[name] = existing

    def match(self, centroid: np.ndarray) -> tuple[str | None, float]:
        if not self._speakers:
            return None, 0.0
        c = np.asarray(centroid, dtype=np.float32).reshape(-1)
        best, best_sim = None, -1.0
        for name, rec in self._speakers.items():
            ref = np.asarray(rec.get("centroid", []), dtype=np.float32).reshape(-1)
            if ref.size == 0:
                continue
            sim = float(np.dot(c, ref) / (np.linalg.norm(c) * np.linalg.norm(ref) + 1e-9))
            if sim > best_sim:
                best, best_sim = name, sim
        return best, best_sim


__all__ = ["SpeakerRegistry"]
