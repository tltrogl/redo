"""Robust audio loading and probing utilities."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

PCM_FORMATS = {"WAV", "WAVEX", "AIFF", "AIFFC"}
PCM_FALLBACK_SUBTYPES = {"PCM", "FLOAT", "DOUBLE"}

__all__ = [
    "PCM_FORMATS",
    "safe_load_audio",
    "decode_audio_segment",
    "probe_audio_metadata",
    "get_audio_duration",
]


def is_uncompressed_pcm(info: sf.Info) -> bool:
    """Return True when the file is a WAV/AIFF PCM variant we can read directly."""

    try:
        fmt = (info.format or "").upper()
        subtype = (info.subtype or "").upper()
    except AttributeError:
        return False

    if fmt not in PCM_FORMATS:
        return False

    return any(token in subtype for token in PCM_FALLBACK_SUBTYPES)


def load_uncompressed_with_soundfile(
    source: Path, target_sr: int, mono: bool
) -> tuple[np.ndarray, int]:
    """Read PCM WAV/AIFF directly via libsndfile."""

    y, sr = sf.read(source, always_2d=False, dtype="float32")
    if mono and y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        import soxr  # Local import to avoid dependency at import time

        y = soxr.resample(y, sr, target_sr)
        sr = target_sr
    return y.astype(np.float32), sr


def decode_with_ffmpeg(source: Path, target_sr: int, mono: bool) -> tuple[np.ndarray, int]:
    """Decode arbitrary containers via ffmpeg → temp wav → soundfile."""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1" if mono else "2",
        "-ar",
        str(target_sr),
        "-f",
        "wav",
        "-loglevel",
        "quiet",
        tmp_wav,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        y, sr = sf.read(tmp_wav, always_2d=False, dtype="float32")
        if mono and y.ndim > 1:
            y = np.mean(y, axis=1)
        logger.debug("Decoded %s via primary ffmpeg path", source)
        return y.astype(np.float32), sr
    except subprocess.TimeoutExpired as exc:
        logger.error("FFmpeg timeout for %s", source)
        raise RuntimeError(f"Audio decoding timeout for {source}") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - debug aid
        stderr_output = exc.stderr.decode() if exc.stderr else "No error details"
        logger.debug("FFmpeg returned non-zero exit for %s: %s", source, stderr_output)
        raise RuntimeError(
            f"Cannot decode audio file {source}. FFmpeg error: {stderr_output}"
        ) from exc
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass


def safe_load_audio(path: str, target_sr: int, mono: bool = True) -> tuple[np.ndarray, int]:
    """Robust loader that prefers libsndfile for PCM and ffmpeg otherwise."""

    p = Path(path)

    info: sf.Info | None = None
    try:
        info = sf.info(p)
    except Exception as exc_info:  # pragma: no cover - informational only
        logger.debug("soundfile could not inspect %s: %s", p, exc_info)

    if info and is_uncompressed_pcm(info):
        logger.debug("Loading PCM audio directly via soundfile: %s", p)
        return load_uncompressed_with_soundfile(p, target_sr=target_sr, mono=mono)

    try:
        logger.debug("Decoding %s via ffmpeg → WAV", p)
        return decode_with_ffmpeg(p, target_sr=target_sr, mono=mono)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to decode compressed audio containers; install ffmpeg"
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot decode audio file {p} via ffmpeg: {exc}") from exc


def decode_audio_segment(
    path: str,
    target_sr: int,
    *,
    mono: bool = True,
    start: float = 0.0,
    duration: float | None = None,
    info: sf.Info | None = None,
) -> np.ndarray:
    """Decode a bounded segment from an audio file into float32 samples."""

    source = Path(path)
    if info is None:
        try:
            info = sf.info(source)
        except Exception:
            info = None

    start = max(0.0, float(start))
    seg_duration = None if duration is None else max(0.0, float(duration))

    if info and is_uncompressed_pcm(info):
        with sf.SoundFile(source) as snd:
            sr_in = int(snd.samplerate)
            frames_start = int(round(start * sr_in))
            snd.seek(max(0, frames_start))
            frames = None
            if seg_duration is not None:
                frames = int(round(seg_duration * sr_in))
            data = snd.read(frames, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr_in != target_sr:
            import soxr  # Local import to avoid dependency at import time

            data = soxr.resample(data, sr_in, target_sr)
        return np.asarray(data, dtype=np.float32)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(source),
        "-ac",
        "1" if mono else "2",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "-loglevel",
        "quiet",
    ]
    if seg_duration is not None and seg_duration > 0:
        cmd.extend(["-t", str(seg_duration)])
    cmd.append("pipe:1")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to decode compressed audio containers") from exc

    assert proc.stdout is not None
    chunk = bytearray()
    while True:
        buf = proc.stdout.read(65536)
        if not buf:
            break
        chunk.extend(buf)

    stderr_data = b""
    if proc.stderr is not None:
        stderr_data = proc.stderr.read()

    ret = proc.wait()
    if ret != 0:
        stderr_text = stderr_data.decode(errors="ignore")
        raise RuntimeError(f"ffmpeg segment decode failed: {stderr_text}")

    if len(chunk) % 4 != 0:
        chunk = chunk[: len(chunk) - (len(chunk) % 4)]

    audio = np.frombuffer(chunk, dtype="<f4")
    return audio.astype(np.float32, copy=False)


def get_audio_duration(path: str, info: sf.Info | None = None) -> float:
    """Get audio duration with a robust fallback chain."""

    if info is not None:
        try:
            duration = getattr(info, "duration", None)
        except Exception:  # pragma: no cover - defensive
            duration = None
        if duration and duration > 0:
            logger.debug("Got duration via soundfile metadata: %ss", duration)
            return float(duration)

    try:
        info = sf.info(path)
        if info and info.duration and info.duration > 0:
            logger.debug("Got duration via soundfile metadata: %ss", info.duration)
            return float(info.duration)
    except Exception as exc:  # pragma: no cover - informational only
        logger.debug("soundfile failed for duration: %s", exc)

    try:
        import av  # type: ignore

        with av.open(path) as container:  # type: ignore[attr-defined]
            dur = None
            for stream in container.streams:
                if stream.type == "audio" and stream.duration and stream.time_base:
                    dur = float(stream.duration * stream.time_base)
                    break
            if dur is None and container.duration is not None:
                dur = float(container.duration) / 1e6
            if dur is not None and dur > 0:
                logger.debug("Got duration via PyAV: %ss", dur)
                return dur
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("PyAV failed for duration: %s", exc)

    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            logger.debug("Got duration via ffprobe: %ss", duration)
            return duration
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        ValueError,
    ) as exc:  # pragma: no cover - debug aid
        logger.debug("ffprobe failed for duration: %s", exc)

    logger.warning("Could not determine duration for %s, using 0.0", path)
    return 0.0


def probe_audio_metadata(path: str) -> tuple[float, sf.Info | None]:
    """Return the duration and cached :class:`soundfile.Info` for ``path``."""

    info: sf.Info | None = None
    try:
        info = sf.info(path)
    except Exception as exc_info:  # pragma: no cover - informational only
        logger.debug("soundfile could not inspect %s: %s", path, exc_info)

    duration = get_audio_duration(path, info=info)
    return duration, info
