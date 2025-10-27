from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.signal

from ..io.onnx_utils import create_onnx_session
from .logger import logger
from .paths import MODEL_ROOTS, iter_model_subpaths
from .utils import (
    bool_env,
    can_reach_host,
    download_silero_torch,
    merge_regions,
    resolve_state_shape,
    torch_repo_cached,
)


class SileroVAD:
    def __init__(self, threshold: float, speech_pad_sec: float = 0.05, backend: str = "auto"):
        self.threshold = float(threshold)
        self.speech_pad_sec = float(speech_pad_sec)
        self.backend_preference = (backend or "auto").lower()
        self.model = None
        self.get_speech_timestamps = None
        self.session = None
        self.input_name = None
        self.output_name = None
        self._onnx_input_name: str | None = None
        self._onnx_state_name: str | None = None
        self._onnx_sr_name: str | None = None
        self._onnx_state_output_index: int | None = None
        self._onnx_state_shape: tuple[int, ...] = (2, 1, 128)
        self._onnx_state_cache: np.ndarray | None = None
        self._onnx_context_cache: np.ndarray | None = None
        self._onnx_last_sr: int | None = None
        self._load()

    def _load(self) -> None:
        def _load_torch() -> bool:
            override = bool_env("SILERO_VAD_TORCH")
            if override is False:
                logger.info("Silero VAD Torch backend disabled via SILERO_VAD_TORCH")
                return False
            if not torch_repo_cached():
                if override is not True and not can_reach_host("github.com", timeout=3.0):
                    logger.info(
                        "Silero VAD TorchHub repo not cached and GitHub unreachable; falling back to energy VAD"
                    )
                    return False
            timeout_env = os.getenv("SILERO_TORCH_LOAD_TIMEOUT")
            try:
                timeout = float(timeout_env) if timeout_env else 30.0
            except ValueError:
                timeout = 30.0
            timeout = max(5.0, timeout)
            try:
                result = download_silero_torch(timeout)
            except ModuleNotFoundError:
                logger.info("Silero VAD Torch backend unavailable (torch not installed)")
                return False
            except Exception as exc:  # pragma: no cover
                logger.warning("Silero VAD Torch import failed: %s", exc)
                return False
            if not result:
                return False
            self.model, utils = result
            (
                self.get_speech_timestamps,
                self.save_audio,
                self.read_audio,
                self.VADIterator,
                self.collect_chunks,
            ) = utils
            try:
                self.model.eval()
            except Exception:
                pass
            logger.info("Silero VAD PyTorch model loaded (TorchHub)")
            return True

        def _load_onnx() -> bool:
            try:
                onnx_path = os.getenv("SILERO_VAD_ONNX_PATH")
                if not onnx_path:
                    candidate_paths = list(iter_model_subpaths("silero_vad.onnx"))
                    candidate_paths.extend(list(iter_model_subpaths(Path("silero") / "vad.onnx")))
                    for root in MODEL_ROOTS:
                        root_path = Path(root)
                        try:
                            for cand in root_path.glob("**/silero_vad.onnx"):
                                candidate_paths.append(cand)
                        except OSError:
                            continue
                    unique: list[Path] = []
                    seen: set[str] = set()
                    for cand in candidate_paths:
                        resolved = Path(cand)
                        key = str(resolved)
                        if key in seen:
                            continue
                        seen.add(key)
                        unique.append(resolved)
                    for cand in unique:
                        if cand.exists():
                            onnx_path = str(cand)
                            break
                    if not onnx_path and unique:
                        onnx_path = str(unique[0])
                if onnx_path and Path(onnx_path).exists():
                    self.session = create_onnx_session(onnx_path, threads=1)
                    inputs = self.session.get_inputs()
                    outputs = self.session.get_outputs()
                    self.input_name = inputs[0].name if inputs else None
                    self.output_name = outputs[0].name if outputs else None
                    self._onnx_input_name = None
                    self._onnx_state_name = None
                    self._onnx_sr_name = None
                    self._onnx_state_output_index = None
                    self._onnx_state_shape = (2, 1, 128)
                    for inp in inputs:
                        lower = inp.name.lower()
                        if self._onnx_input_name is None and (
                            lower == "input" or lower.endswith("/input") or "input" in lower
                        ):
                            self._onnx_input_name = inp.name
                        elif self._onnx_state_name is None and "state" in lower:
                            self._onnx_state_name = inp.name
                            try:
                                self._onnx_state_shape = resolve_state_shape(
                                    tuple(getattr(inp, "shape", ()))
                                )
                            except Exception:
                                self._onnx_state_shape = (2, 1, 128)
                        elif self._onnx_sr_name is None and (
                            lower == "sr" or "sample_rate" in lower or "samplerate" in lower
                        ):
                            self._onnx_sr_name = inp.name
                    if self._onnx_input_name is None and inputs:
                        self._onnx_input_name = inputs[0].name
                    if self._onnx_state_name is None:
                        for inp in inputs:
                            if "state" in inp.name.lower():
                                self._onnx_state_name = inp.name
                                self._onnx_state_shape = resolve_state_shape(
                                    tuple(getattr(inp, "shape", ()))
                                )
                                break
                    if self._onnx_sr_name is None:
                        for inp in inputs:
                            lower = inp.name.lower()
                            if lower.startswith("sr") or lower == "sr":
                                self._onnx_sr_name = inp.name
                                break
                    for idx, out in enumerate(outputs):
                        lower = out.name.lower()
                        if idx == 0:
                            self.output_name = out.name
                        if self._onnx_state_output_index is None and "state" in lower:
                            self._onnx_state_output_index = idx
                    if self._onnx_state_output_index is None and len(outputs) > 1:
                        self._onnx_state_output_index = 1
                    self._onnx_state_cache = None
                    self._onnx_context_cache = None
                    self._onnx_last_sr = None
                    logger.info("Silero VAD ONNX model loaded: %s", onnx_path)
                    return True
            except Exception as exc:
                logger.info("Silero VAD ONNX unavailable: %s", exc)
            self.session = None
            self.input_name = None
            self.output_name = None
            return False

        pref = self.backend_preference
        if pref == "onnx":
            if not _load_onnx():
                raise RuntimeError("Silero VAD ONNX requested but unavailable")
            return
        if pref == "torch":
            if not _load_torch():
                raise RuntimeError("Silero VAD Torch backend requested but unavailable")
            return
        if _load_onnx():
            return
        override = bool_env("SILERO_VAD_TORCH")
        should_try_torch = override is True
        if override is None:
            should_try_torch = torch_repo_cached() or can_reach_host("github.com", timeout=3.0)
        if should_try_torch and _load_torch():
            return
        if should_try_torch:
            logger.info("Silero VAD Torch backend unavailable; proceeding without it")
        else:
            logger.info("Silero VAD Torch backend skipped (offline/disabled); using fallbacks")

    def _detect_with_onnx(
        self,
        wav: np.ndarray,
        sr: int,
        *,
        min_speech_sec: float,
        min_silence_sec: float,
    ) -> list[tuple[float, float]]:
        if self.session is None or self._onnx_input_name is None:
            return []
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if sr != 16000:
            audio = scipy.signal.resample_poly(audio, 16000, sr).astype(np.float32, copy=False)
            sr = 16000
        if audio.size == 0:
            return []
        chunk_size = 512
        context_size = 64
        batch_size = 1
        orig_samples = audio.shape[0]
        pad = (-orig_samples) % chunk_size
        if pad:
            audio = np.pad(audio, (0, pad), mode="constant")
        num_chunks = audio.shape[0] // chunk_size
        if num_chunks == 0:
            return []
        state_shape = list(self._onnx_state_shape or (2, 1, 128))
        if len(state_shape) < 3:
            state_shape = [2, batch_size, 128]
        else:
            state_shape = state_shape[:3]
            state_shape[1] = batch_size
        state = np.zeros(tuple(state_shape), dtype=np.float32)
        context = np.zeros((batch_size, context_size), dtype=np.float32)
        self._onnx_state_cache = state
        self._onnx_context_cache = context
        self._onnx_last_sr = sr
        sr_array = np.array(sr, dtype=np.int64)
        chunk_probs: list[float] = []
        offset = 0
        for _ in range(num_chunks):
            chunk = audio[offset : offset + chunk_size]
            offset += chunk_size
            chunk = chunk.reshape(batch_size, -1)
            window = np.concatenate([context, chunk], axis=1).astype(np.float32, copy=False)
            feeds: dict[str, np.ndarray] = {self._onnx_input_name: window}
            if self._onnx_state_name:
                feeds[self._onnx_state_name] = state
            if self._onnx_sr_name:
                feeds[self._onnx_sr_name] = sr_array
            ort_outs = self.session.run(None, feeds)
            if not isinstance(ort_outs, list | tuple) or not ort_outs:
                continue
            logits = np.asarray(ort_outs[0], dtype=np.float32)
            logits = np.squeeze(logits)
            if logits.ndim == 0:
                logits = logits.reshape(1)
            if logits.ndim >= 1 and logits.shape[-1] == 2 and logits.size % 2 == 0:
                logits2d = logits.reshape(-1, 2)
                exps = np.exp(logits2d - np.max(logits2d, axis=1, keepdims=True))
                denom = np.sum(exps, axis=1, keepdims=True) + 1e-9
                probs = (exps[:, 1:2] / denom).reshape(-1)
            else:
                logits1d = logits.reshape(-1)
                probs = 1.0 / (1.0 + np.exp(-logits1d))
            probs = np.clip(probs, 0.0, 1.0)
            chunk_probs.extend(probs.tolist())
            if self._onnx_state_output_index is not None and self._onnx_state_output_index < len(
                ort_outs
            ):
                try:
                    state_out = np.asarray(
                        ort_outs[self._onnx_state_output_index], dtype=np.float32
                    )
                    state = state_out.reshape(tuple(state_shape))
                except Exception:
                    state = np.zeros(tuple(state_shape), dtype=np.float32)
            context = window[:, -context_size:]
            self._onnx_state_cache = state
            self._onnx_context_cache = context
        if not chunk_probs:
            return []
        chunk_probs = np.asarray(chunk_probs, dtype=np.float32)
        if chunk_probs.size > num_chunks:
            chunk_probs = chunk_probs[:num_chunks]
        chunk_duration = chunk_size / float(sr)
        total_duration = orig_samples / float(sr)
        speech_mask = chunk_probs > self.threshold
        segments: list[tuple[float, float]] = []
        start_idx: int | None = None
        for idx, flag in enumerate(speech_mask):
            if flag and start_idx is None:
                start_idx = idx
            elif not flag and start_idx is not None:
                end_idx = idx
                start = start_idx * chunk_duration
                end = min(end_idx * chunk_duration, total_duration)
                if end - start >= min_speech_sec:
                    segments.append((start, end))
                start_idx = None
        if start_idx is not None:
            end_idx = len(speech_mask)
            start = start_idx * chunk_duration
            end = min(end_idx * chunk_duration, total_duration)
            if end - start >= min_speech_sec:
                segments.append((start, end))
        if not segments:
            return []
        pad = float(self.speech_pad_sec)
        padded = []
        for start, end in segments:
            s = max(0.0, start - pad)
            e = min(total_duration, end + pad)
            if e - s >= min_speech_sec:
                padded.append((s, e))
        if not padded:
            return []
        return merge_regions(padded, gap=min_silence_sec)

    def detect(
        self, wav: np.ndarray, sr: int, min_speech_sec: float, min_silence_sec: float
    ) -> list[tuple[float, float]]:
        if self.session is not None:
            try:
                return self._detect_with_onnx(
                    wav,
                    sr,
                    min_speech_sec=min_speech_sec,
                    min_silence_sec=min_silence_sec,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Silero ONNX VAD failed: %s", exc)
                return []
        if self.model is None or self.get_speech_timestamps is None:
            return []
        try:
            import torch

            wav_t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
            timestamps = self.get_speech_timestamps(
                wav_t,
                self.model,
                sampling_rate=sr,
                threshold=self.threshold,
                min_speech_duration_ms=int(float(min_speech_sec) * 1000),
                min_silence_duration_ms=int(float(min_silence_sec) * 1000),
                speech_pad_ms=int(self.speech_pad_sec * 1000),
            )
            spans = [(t["start"] / sr, t["end"] / sr) for t in timestamps]
            return merge_regions(spans, gap=min_silence_sec)
        except Exception as exc:  # pragma: no cover
            logger.warning("Silero VAD failed: %s", exc)
            return []


__all__ = ["SileroVAD"]
