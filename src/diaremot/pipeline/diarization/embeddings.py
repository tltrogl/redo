from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np

from ..io.onnx_utils import create_onnx_session
from .logger import logger
from .paths import iter_model_subpaths


class ECAPAEncoder:
    def __init__(self, model_path: Path | None = None) -> None:
        self.session = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.model_path = Path(model_path) if model_path else None
        self._load()

    def _load(self) -> None:
        try:
            if self.model_path and Path(self.model_path).exists():
                model_path = Path(self.model_path)
            else:
                env_path = os.getenv("ECAPA_ONNX_PATH")
                if env_path:
                    model_path = Path(env_path)
                else:
                    candidate_paths = list(
                        iter_model_subpaths(Path("ecapa_onnx") / "ecapa_tdnn.onnx")
                    )
                    candidate_paths.extend(
                        list(
                            iter_model_subpaths(
                                Path("Diarization") / "ecapa-onnx" / "ecapa_tdnn.onnx"
                            )
                        )
                    )
                    candidate_paths.extend(list(iter_model_subpaths("ecapa_tdnn.onnx")))
                    unique_candidates: list[Path] = []
                    seen: set[str] = set()
                    for cand in candidate_paths:
                        resolved = Path(cand)
                        key = str(resolved)
                        if key in seen:
                            continue
                        seen.add(key)
                        unique_candidates.append(resolved)
                    model_path = next((cand for cand in unique_candidates if cand.exists()), None)
                    if model_path is None and unique_candidates:
                        model_path = unique_candidates[0]
                    if model_path is None:
                        model_path = Path.cwd() / "models" / "ecapa_tdnn.onnx"
            self.session = create_onnx_session(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info("ECAPA ONNX model loaded: %s", model_path)
        except Exception as exc:
            logger.error("ECAPA ONNX model unavailable: %s", exc)
            self.session = None

    def embed_batch(self, batch: list[np.ndarray], sr: int) -> list[np.ndarray | None]:
        if self.session is None or not batch:
            return [None] * len(batch)
        try:
            mel_specs = []
            max_frames = 0
            for x in batch:
                mel = librosa.feature.melspectrogram(
                    y=x,
                    sr=sr,
                    n_fft=400,
                    hop_length=160,
                    n_mels=80,
                    fmin=20,
                    fmax=sr / 2,
                )
                mel = librosa.power_to_db(mel, ref=1.0).T
                try:
                    m = mel.mean(axis=0, keepdims=True)
                    s = mel.std(axis=0, keepdims=True) + 1e-8
                    mel = (mel - m) / s
                except Exception as exc:
                    logger.warning("CMVN normalization failed for clip %d: %s", len(mel_specs), exc)
                mel_specs.append(mel)
                if mel.shape[0] > max_frames:
                    max_frames = mel.shape[0]
            padded = []
            for mel in mel_specs:
                pad_width = max(0, max_frames - mel.shape[0])
                if pad_width:
                    mel = np.pad(mel, ((0, pad_width), (0, 0)), mode="edge")
                padded.append(mel)
            inputs = np.stack(padded, axis=0).astype(np.float32)
            inputs = np.transpose(inputs, (0, 2, 1))
            ort_inputs = {self.input_name: inputs}
            outputs = self.session.run([self.output_name], ort_inputs)
            embeddings = []
            for arr in outputs[0]:
                vec = np.asarray(arr, dtype=np.float32).reshape(-1)
                norm = np.linalg.norm(vec)
                embeddings.append(vec / (norm + 1e-8))
            return embeddings
        except Exception as exc:
            logger.error("ECAPA embedding inference failed: %s", exc)
            return [None] * len(batch)


__all__ = ["ECAPAEncoder"]
