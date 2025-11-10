from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .backends import ModelManager, backends
from .models import (
    BatchingConfig,
    TranscriptionSegment,
    estimate_snr_db,
    resample_audio_fast,
)
from .postprocess import distribute_batch_results

__all__ = [
    "AsyncTranscriber",
    "AudioTranscriber",
    "create_transcriber",
    "benchmark_transcription",
    "create_batch_groups",
]


class AsyncTranscriber:
    """High-performance async transcription engine."""

    def __init__(
        self,
        model_size: str = "large-v3",
        language: str | None = None,
        beam_size: int = 1,
        temperature: float | list[float] = 0.0,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.20,
        condition_on_previous_text: bool = False,
        word_timestamps: bool = True,
        max_asr_window_sec: int = 480,
        vad_min_silence_ms: int = 1200,
        language_mode: str = "auto",
        batching_config: BatchingConfig | None = None,
        max_workers: int = 2,
        segment_timeout_sec: float = 120.0,
        batch_timeout_sec: float = 600.0,
        compute_type: str | None = None,
        cpu_threads: int | None = None,
        asr_backend: str = "auto",
        local_first: bool = True,
        max_concurrent_segments: int = 2,
        **kwargs: Any,
    ) -> None:
        self.logger = logging.getLogger(__name__ + ".AsyncTranscriber")
        self.model_size = model_size
        self.language_mode = language_mode
        self.model_manager = ModelManager()
        self.max_workers = max(1, int(max_workers))
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="tx-worker",
        )
        self._model_concurrency = max(1, int(max_workers))
        self._max_concurrent_segments = max(1, int(max_concurrent_segments))
        self.segment_timeout_sec = float(segment_timeout_sec)
        self.batch_timeout_sec = float(batch_timeout_sec)

        self.config: dict[str, Any] = {
            "model_size": model_size,
            "language": language if language_mode != "auto" else None,
            "beam_size": beam_size,
            "temperature": temperature,
            "compression_ratio_threshold": compression_ratio_threshold,
            "log_prob_threshold": log_prob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text,
            "word_timestamps": word_timestamps,
            "max_asr_window_sec": max_asr_window_sec,
            "vad_min_silence_ms": vad_min_silence_ms,
            "language_mode": language_mode,
            "compute_type": compute_type or "float32",
            "cpu_threads": cpu_threads or os.cpu_count() or 2,
            "asr_backend": asr_backend,
            "local_first": local_first,
            "segment_timeout_sec": segment_timeout_sec,
            "batch_timeout_sec": batch_timeout_sec,
            "max_concurrent_segments": max_concurrent_segments,
        }
        if kwargs:
            self.config.update(kwargs)

        self.batching = batching_config or BatchingConfig(enabled=True)
        self._fallback_triggered = False
        self._fallback_reason: str | None = None
        self._fallback_to: str | None = None
        self._last_failures: dict[str, str] = {}
        self._stats = {"segments_processed": 0, "batches_processed": 0}

        self._loop = asyncio.get_event_loop()
        self._loop.set_default_executor(self.executor)
        self._reset_lock = threading.Lock()

    async def __aenter__(self) -> AsyncTranscriber:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.executor.shutdown(wait=True)

    async def transcribe_segments(
        self,
        audio_16k_mono: np.ndarray,
        sr: int,
        diar_segments: list[dict[str, Any]],
    ) -> list[TranscriptionSegment]:
        if audio_16k_mono.size == 0 or not diar_segments:
            return []

        audio = audio_16k_mono.astype(np.float32)
        segments = self._preprocess_segments(diar_segments, len(audio) / sr)

        if not segments:
            return []

        if self.language_mode == "auto":
            await self._pin_language(audio, sr, segments)

        groups = create_batch_groups(segments, self.batching)
        tasks = []
        for group_name, group_segments in groups.items():
            if group_name.startswith("batch_"):
                tasks.append(
                    self._process_batch_concurrent(audio, sr, group_segments)
                )
            elif group_name == "individual":
                tasks.append(
                    self._process_individual_concurrent(audio, sr, group_segments)
                )

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[TranscriptionSegment] = []
        for result in batch_results:
            if isinstance(result, Exception):
                self.logger.error("Batch processing failed: %s", result)
                continue
            all_results.extend(result)

        all_results.sort(key=lambda seg: seg.start_time)
        self._stats["segments_processed"] += len(all_results)
        return all_results

    def _preprocess_segments(
        self, segments: list[dict[str, Any]], audio_duration: float
    ) -> list[dict[str, Any]]:
        if not segments:
            return []

        normalized: list[dict[str, Any]] = []
        for seg in segments:
            start = float(seg.get("start_time", seg.get("start", 0.0)))
            end = float(seg.get("end_time", seg.get("end", start + 1.0)))
            start = max(0.0, start)
            end = min(audio_duration, max(start + self.batching.min_segment_sec, end))
            if end - start < self.batching.min_segment_sec:
                continue

            normalized.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "speaker_id": seg.get(
                        "speaker_id",
                        seg.get("speaker", f"Speaker_{len(normalized) + 1}"),
                    ),
                    "speaker_name": seg.get(
                        "speaker_name",
                        seg.get("speaker", f"Speaker_{len(normalized) + 1}"),
                    ),
                }
            )

        normalized.sort(key=lambda item: item["start_time"])
        resolved: list[dict[str, Any]] = []
        for seg in normalized:
            if resolved:
                prev = resolved[-1]
                if seg["start_time"] < prev["end_time"]:
                    midpoint = (prev["end_time"] + seg["start_time"]) / 2
                    prev["end_time"] = midpoint
                    seg["start_time"] = midpoint

            if seg["end_time"] - seg["start_time"] >= self.batching.min_segment_sec:
                resolved.append(seg)

        return resolved

    async def _pin_language(
        self, audio: np.ndarray, sr: int, segments: list[dict[str, Any]]
    ) -> None:
        if not segments:
            return

        longest = max(segments, key=lambda s: s["end_time"] - s["start_time"])
        pinning_result = await self._transcribe_single_segment(
            audio,
            sr,
            longest,
            force_detection=True,
        )
        if (
            pinning_result
            and pinning_result.language
            and (pinning_result.language_probability or 0.0) >= 0.75
        ):
            self.config["language"] = pinning_result.language
            self.logger.info("Language pinned to: %s", pinning_result.language)

    async def _process_batch_concurrent(
        self, audio: np.ndarray, sr: int, segments: list[dict[str, Any]]
    ) -> list[TranscriptionSegment]:
        if not segments:
            return []

        total_seg_duration = sum(seg["end_time"] - seg["start_time"] for seg in segments)
        self.logger.info(
            "[asr] batch of %d segments (%.1f sec audio, silence %.2fs)",
            len(segments),
            total_seg_duration,
            self.batching.batch_silence_sec,
        )

        silence_samples = int(self.batching.batch_silence_sec * sr)
        spacer = np.zeros(silence_samples, dtype=np.float32)

        concat_parts = []
        segment_boundaries = []
        current_time = 0.0

        for seg in segments:
            start_idx = int(seg["start_time"] * sr)
            end_idx = int(seg["end_time"] * sr)
            audio_segment = audio[start_idx:end_idx]
            if len(audio_segment) == 0:
                continue

            segment_boundaries.append(
                {
                    "original": seg,
                    "concat_start": current_time,
                    "concat_end": current_time + len(audio_segment) / sr,
                }
            )
            concat_parts.extend([audio_segment, spacer])
            current_time += (len(audio_segment) + len(spacer)) / sr

        if not concat_parts:
            return []

        concat_audio = np.concatenate(concat_parts)

        async with self.model_manager.get_model("primary", self.config) as model:
            try:
                batch_result = await asyncio.wait_for(
                    self._run_transcription(model, concat_audio, sr),
                    timeout=self.batch_timeout_sec,
                )
            except TimeoutError:
                self._last_failures["batch_timeout"] = (
                    f"Batch transcription exceeded {self.batch_timeout_sec}s"
                )
                self.logger.error(self._last_failures["batch_timeout"])
                return await self._process_individual_concurrent(audio, sr, segments)

        if not batch_result:
            return await self._process_individual_concurrent(audio, sr, segments)

        results = distribute_batch_results(batch_result, segment_boundaries)
        self._stats["batches_processed"] += 1
        self.logger.info(
            "[asr] batch completed (%d segments, total batches=%d)",
            len(results),
            self._stats["batches_processed"],
        )
        return results

    async def _process_individual_concurrent(
        self, audio: np.ndarray, sr: int, segments: list[dict[str, Any]]
    ) -> list[TranscriptionSegment]:
        requested = max(1, int(self.config.get("max_concurrent_segments", 2)))
        concurrency = min(requested, self._model_concurrency)
        semaphore = asyncio.Semaphore(concurrency)
        total = len(segments)
        if total:
            self.logger.info(
                "[asr] processing %d segments individually (max_concurrency=%d)",
                total,
                concurrency,
            )
        progress_lock = asyncio.Lock()
        completed = 0
        progress_step = max(1, total // 5) if total else 1

        async def process_one(seg: dict[str, Any]) -> TranscriptionSegment | None:
            nonlocal completed
            async with semaphore:
                try:
                    result = await self._transcribe_single_segment(audio, sr, seg)
                except TimeoutError:
                    self.logger.warning(
                        "Segment %s-%ss timed out after %ss",
                        seg.get("start_time", "?"),
                        seg.get("end_time", "?"),
                        self.segment_timeout_sec,
                    )
                    result = None
            async with progress_lock:
                completed += 1
                if total and (completed % progress_step == 0 or completed == total or total <= 10):
                    self.logger.info(
                        "[asr] individual progress %d/%d (%.0f%%)",
                        completed,
                        total,
                        (completed / total) * 100 if total else 100,
                    )
            return result

        tasks = [process_one(seg) for seg in segments]
        results = await asyncio.gather(*tasks)
        return [res for res in results if res is not None]

    async def _run_transcription(
        self,
        model: Any,
        audio: np.ndarray,
        sr: int,
        force_detection: bool = False,
    ) -> TranscriptionSegment | None:
        backend = "faster-whisper" if backends.has_faster_whisper else "openai-whisper"
        if backend == "faster-whisper":
            return self._faster_whisper_transcribe(model, audio, sr, force_detection)
        if backend == "openai-whisper":
            return self._openai_whisper_transcribe(model, audio, sr)
        raise RuntimeError("No transcription backend available")

    async def _transcribe_single_segment(
        self,
        audio: np.ndarray,
        sr: int,
        seg: dict[str, Any],
        force_detection: bool = False,
    ) -> TranscriptionSegment | None:
        start_idx = int(seg["start_time"] * sr)
        end_idx = int(seg["end_time"] * sr)
        segment_audio = audio[start_idx:end_idx]
        if len(segment_audio) == 0:
            return None

        segment_audio = resample_audio_fast(segment_audio, sr, sr)

        async with self.model_manager.get_model("primary", self.config) as model:
            result = await self._run_transcription(
                model,
                segment_audio,
                sr,
                force_detection=force_detection,
            )
        if result is None:
            return None

        try:
            result.start_time = float(seg["start_time"])
            result.end_time = float(seg["end_time"])
            result.speaker_id = seg.get("speaker_id")
            result.speaker_name = seg.get("speaker_name")
            result.snr_db = estimate_snr_db(segment_audio)
        except Exception:
            pass
        return result

    def _faster_whisper_transcribe(
        self,
        model: Any,
        audio: np.ndarray,
        sr: int,
        force_detection: bool = False,
    ) -> TranscriptionSegment | None:
        language = None if force_detection else self.config["language"]
        duration = float(len(audio) / sr) if sr else 0.0
        vad_min_sil_ms = int(self.config.get("vad_min_silence_ms", 1800))
        is_short = duration < max(3.0, vad_min_sil_ms / 1000.0)
        try:
            is_quiet = float(np.max(np.abs(audio))) < 1e-4
        except Exception:
            is_quiet = False

        def _call_transcribe(use_vad: bool, threshold_value: float):
            kw = dict(
                language=language,
                beam_size=self.config["beam_size"],
                temperature=self.config["temperature"],
                compression_ratio_threshold=self.config["compression_ratio_threshold"],
                log_prob_threshold=self.config["log_prob_threshold"],
                no_speech_threshold=threshold_value,
                condition_on_previous_text=self.config["condition_on_previous_text"],
                word_timestamps=self.config["word_timestamps"],
            )
            if use_vad:
                kw.update(
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": self.config["vad_min_silence_ms"],
                        "min_speech_duration_ms": 250,
                        "speech_pad_ms": 300,
                    },
                )
            else:
                kw.update(vad_filter=False)
            return model.transcribe(audio, **kw)

        prefer_vad = not (is_short or is_quiet)
        base_threshold = float(self.config["no_speech_threshold"])
        attempts: list[tuple[bool, float]] = []
        seen: set[tuple[bool, float]] = set()

        def _add_attempt(use_vad: bool, threshold_value: float) -> None:
            key = (use_vad, round(threshold_value, 4))
            if key not in seen:
                attempts.append((use_vad, threshold_value))
                seen.add(key)

        _add_attempt(prefer_vad, base_threshold)
        if prefer_vad:
            _add_attempt(False, base_threshold)

        if base_threshold > 0.05:
            lowered = max(0.0, min(0.15, base_threshold - 0.1))
            _add_attempt(False, lowered)
            if prefer_vad:
                _add_attempt(True, lowered)

        if base_threshold > 0.0:
            _add_attempt(False, 0.0)
            if prefer_vad:
                _add_attempt(True, 0.0)

        text_parts: list[str] = []
        words: list[dict[str, Any]] = []
        logprobs: list[float] = []
        last_error: Exception | None = None

        def _collect(_segments):
            tparts: list[str] = []
            wlist: list[dict[str, Any]] = []
            lps: list[float] = []
            for seg in _segments:
                text = getattr(seg, "text", "").strip()
                if text:
                    tparts.append(text)
                    if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                        lps.append(seg.avg_logprob)
                    if (
                        self.config["word_timestamps"]
                        and hasattr(seg, "words")
                        and seg.words
                    ):
                        for word in seg.words:
                            wlist.append(
                                {
                                    "word": getattr(word, "word", ""),
                                    "start": getattr(word, "start", 0.0),
                                    "end": getattr(word, "end", 0.0),
                                    "probability": getattr(word, "probability", 1.0),
                                }
                            )
            return tparts, wlist, lps

        for idx, (use_vad, threshold_value) in enumerate(attempts, start=1):
            self.logger.debug(
                "ASR attempt %s/%s (vad_filter=%s, no_speech_threshold=%.2f, duration=%.2fs)",
                idx,
                len(attempts),
                use_vad,
                threshold_value,
                duration,
            )
            try:
                segments, info = _call_transcribe(use_vad, threshold_value)
            except Exception as exc:
                last_error = exc
                msg = str(exc).lower()
                if use_vad and ("empty sequence" in msg or "vad" in msg or "max()" in msg):
                    self.logger.info(
                        "Retrying faster-whisper without VAD due to VAD error (threshold=%.2f)",
                        threshold_value,
                    )
                else:
                    self.logger.warning(
                        "faster-whisper attempt failed (vad=%s threshold=%.2f): %s",
                        use_vad,
                        threshold_value,
                        exc,
                    )
                continue

            text_parts, words, logprobs = _collect(segments)
            if text_parts:
                if (use_vad != prefer_vad) or (
                    abs(threshold_value - base_threshold) > 1e-6
                ):
                    self.logger.debug(
                        "faster-whisper succeeded with vad_filter=%s no_speech_threshold=%.2f",
                        use_vad,
                        threshold_value,
                    )
                last_error = None
                break
            else:
                self.logger.debug(
                    "faster-whisper returned empty transcript (vad_filter=%s, threshold=%.2f)",
                    use_vad,
                    threshold_value,
                )
                continue
        else:
            if last_error is not None:
                self.logger.error("faster-whisper transcription failed: %s", last_error)
            return None

        if not text_parts:
            self.logger.debug(
                "faster-whisper produced no text after %s attempts (duration=%.2fs)",
                len(attempts),
                duration,
            )
            return None

        return TranscriptionSegment(
            start_time=0.0,
            end_time=len(audio) / sr,
            text=" ".join(text_parts),
            confidence=float(np.exp(np.mean(logprobs))) if logprobs else 1.0,
            words=words if words else None,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            asr_logprob_avg=float(np.mean(logprobs)) if logprobs else None,
            model_used="faster-whisper",
        )

    def _openai_whisper_transcribe(
        self,
        model: Any,
        audio: np.ndarray,
        sr: int,
    ) -> TranscriptionSegment | None:
        try:
            result = model.transcribe(
                audio.astype(np.float32),
                language=self.config["language"],
                word_timestamps=self.config["word_timestamps"],
                temperature=(
                    self.config["temperature"]
                    if isinstance(self.config["temperature"], (int, float))
                    else 0.0
                ),
                fp16=False,
                verbose=False,
            )
        except Exception as exc:
            self.logger.error("OpenAI whisper transcription failed: %s", exc)
            return None

        text = result.get("text", "").strip()
        if not text:
            return None

        words = []
        if self.config["word_timestamps"] and "segments" in result:
            for segment in result["segments"]:
                if "words" in segment:
                    for word in segment["words"]:
                        words.append(
                            {
                                "word": word.get("word", ""),
                                "start": float(word.get("start", 0.0)),
                                "end": float(word.get("end", 0.0)),
                                "probability": float(word.get("probability", 1.0)),
                            }
                        )

        return TranscriptionSegment(
            start_time=0.0,
            end_time=len(audio) / sr,
            text=text,
            confidence=1.0,
            words=words if words else None,
            language=result.get("language"),
            language_probability=None,
            model_used="openai-whisper",
        )

    async def validate_backend(self) -> dict[str, Any]:
        validation = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "faster_whisper_available": backends.has_faster_whisper,
            "openai_whisper_available": backends.has_openai_whisper,
            "librosa_available": backends.has_librosa,
            "active_backend": None,
            "backend_functional": False,
            "error": None,
            "model_loadable": False,
            "test_transcription": False,
        }

        try:
            async with self.model_manager.get_model("validation", self.config) as model:
                validation["model_loadable"] = True
                validation["active_backend"] = (
                    "faster-whisper" if backends.has_faster_whisper else "openai-whisper"
                )

                test_audio = np.random.normal(0, 0.01, 1600).astype(np.float32)
                result = await self._run_transcription(model, test_audio, 16000)

                validation["backend_functional"] = result is not None
                validation["test_transcription"] = result is not None

        except Exception as exc:
            validation["error"] = str(exc)

        return validation

    def get_model_info(self) -> dict[str, Any]:
        backend = "none"
        if backends.has_faster_whisper:
            backend = "faster-whisper"
        elif backends.has_openai_whisper:
            backend = "openai-whisper"

        info = {
            "model_size": self.config["model_size"],
            "backend": backend,
            "device": "cpu",
            "compute_type": self.config.get("compute_type", "float32"),
            "language": self.config["language"] or "auto",
            "language_mode": self.config["language_mode"],
            "batching_enabled": self.batching.enabled,
            "max_workers": self.executor._max_workers,
            "word_timestamps": self.config["word_timestamps"],
            "vad_enabled": True,
            "stats": self._stats,
        }
        info.update(
            {
                "fallback_triggered": self._fallback_triggered,
                "fallback_reason": self._fallback_reason,
                "fallback_to": self._fallback_to,
                "load_errors": getattr(self.model_manager, "last_errors", {}),
                "runtime_failures": getattr(self, "_last_failures", {}),
            }
        )
        return info

    def get_performance_stats(self) -> dict[str, Any]:
        return {
            "segments_processed": self._stats["segments_processed"],
            "batches_processed": self._stats["batches_processed"],
            "batching_enabled": self.batching.enabled,
            "average_batch_size": (
                self._stats["segments_processed"]
                / max(1, self._stats["batches_processed"])
                if self._stats["batches_processed"] > 0
                else 0
            ),
            "executor_threads": self.executor._max_workers,
            "models_loaded": len(self.model_manager._models),
        }

    def _reset_executor(self) -> None:
        with self._reset_lock:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="tx-worker"
            )
            self._loop.set_default_executor(self.executor)


class AudioTranscriber:
    def __init__(self, **kwargs: Any) -> None:
        self._async_transcriber = AsyncTranscriber(**kwargs)
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def transcribe_segments(
        self,
        audio_16k_mono: np.ndarray,
        sr: int,
        diar_segments: list[dict[str, Any]],
    ) -> list[TranscriptionSegment]:
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_transcriber.transcribe_segments(
                audio_16k_mono, sr, diar_segments
            )
        )

    def get_model_info(self) -> dict[str, Any]:
        return self._async_transcriber.get_model_info()

    def validate_backend(self) -> dict[str, Any]:
        loop = self._get_loop()
        return loop.run_until_complete(self._async_transcriber.validate_backend())

    def get_performance_stats(self) -> dict[str, Any]:
        return self._async_transcriber.get_performance_stats()


def create_transcriber(
    model_size: str = "large-v3",
    enable_async: bool = False,
    enable_batching: bool = True,
    max_workers: int = 2,
    **kwargs: Any,
) -> AudioTranscriber | AsyncTranscriber:
    batching_config = BatchingConfig(enabled=enable_batching)
    kwargs.pop("device", None)
    transcriber_kwargs = {
        "model_size": model_size,
        "batching_config": batching_config,
        "max_workers": max_workers,
        **kwargs,
    }
    async_transcriber = AsyncTranscriber(**transcriber_kwargs)
    if enable_async:
        return async_transcriber
    return AudioTranscriber(**transcriber_kwargs)


async def benchmark_transcription(
    duration_sec: float = 60.0,
    num_segments: int = 12,
    model_size: str = "tiny",
) -> dict[str, Any]:
    sr = 16000
    total_samples = int(duration_sec * sr)
    audio = np.random.normal(0, 0.1, total_samples).astype(np.float32)
    segment_length = duration_sec / num_segments
    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = min(duration_sec, start + segment_length)
        segments.append(
            {
                "start_time": start,
                "end_time": end,
                "speaker_id": f"Speaker_{i % 2}",
                "speaker_name": f"Speaker {i % 2}",
            }
        )

    batching = BatchingConfig(enabled=True)
    transcriber = AsyncTranscriber(
        model_size=model_size,
        batching_config=batching,
    )

    start_time = time.time()
    results_batched = await transcriber.transcribe_segments(audio, sr, segments)
    batched_time = time.time() - start_time

    transcriber.batching.enabled = False
    start_time = time.time()
    results_individual = await transcriber.transcribe_segments(audio, sr, segments)
    individual_time = time.time() - start_time

    return {
        "test_config": {
            "duration_sec": duration_sec,
            "num_segments": num_segments,
            "model_size": model_size,
        },
        "batched": {
            "processing_time": batched_time,
            "segments_returned": len(results_batched),
            "rtf": batched_time / duration_sec,
        },
        "individual": {
            "processing_time": individual_time,
            "segments_returned": len(results_individual),
            "rtf": individual_time / duration_sec,
        },
        "speedup": individual_time / batched_time if batched_time > 0 else 0.0,
    }


def create_batch_groups(
    segments: list[dict[str, Any]],
    batching: BatchingConfig,
) -> dict[str, list[dict[str, Any]]]:
    if not batching.enabled or len(segments) < batching.min_segments_threshold:
        return {"individual": segments}

    short_segments: list[dict[str, Any]] = []
    long_segments: list[dict[str, Any]] = []
    for seg in segments:
        duration = seg["end_time"] - seg["start_time"]
        if duration <= batching.short_segment_max_sec:
            short_segments.append(seg)
        else:
            long_segments.append(seg)

    batched_groups: list[list[dict[str, Any]]] = []
    if short_segments:
        current_batch: list[dict[str, Any]] = []
        current_duration = 0.0
        for seg in short_segments:
            seg_duration = seg["end_time"] - seg["start_time"]
            if current_batch and (
                current_duration + seg_duration > batching.max_batch_duration_sec
                or len(current_batch) >= batching.max_segments_per_batch
            ):
                batched_groups.append(current_batch)
                current_batch = []
                current_duration = 0.0
            current_batch.append(seg)
            current_duration += seg_duration
        if current_batch:
            batched_groups.append(current_batch)

    groups: dict[str, list[dict[str, Any]]] = {"individual": long_segments}
    for idx, batch in enumerate(batched_groups):
        groups[f"batch_{idx}"] = batch
    return groups
