"""Preprocessing stages (audio + background SED tagging)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..logging_utils import StageGuard, _fmt_hms
from ..pipeline_checkpoint_system import ProcessingStage
from .base import PipelineState
from .utils import (
    atomic_write_json,
    compute_audio_sha16,
    compute_audio_sha16_from_file,
    compute_pp_signature,
    compute_sed_signature,
    read_json_safe,
)

if TYPE_CHECKING:
    from ..orchestrator import AudioAnalysisPipelineV2

__all__ = ["run_preprocess", "run_background_sed"]


def run_preprocess(
    pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard
) -> None:
    if not hasattr(pipeline, "pre") or pipeline.pre is None:
        raise RuntimeError("Preprocessor component unavailable; initialization failed")

    # Compute preprocessing config signature early
    state.pp_sig = compute_pp_signature(pipeline.pp_conf)

    # Try to load cached preprocessed audio first
    state.audio_sha16 = compute_audio_sha16_from_file(state.input_audio_path)
    if state.audio_sha16:
        pipeline.checkpoints.seed_file_hash(state.input_audio_path, state.audio_sha16)

        cache_root = pipeline.cache_root
        cache_dir = cache_root / state.audio_sha16
        cache_dir.mkdir(parents=True, exist_ok=True)
        state.cache_dir = cache_dir

        if _load_preprocessed_cache(pipeline, state, cache_dir, guard):
            return

    # No cache or cache invalid - do actual preprocessing
    result = pipeline.pre.process_file(state.input_audio_path)
    state.y = np.asarray(result.audio, dtype=np.float32)
    state.sr = result.sample_rate
    state.health = result.health
    state.duration_s = float(result.duration_s)
    guard.progress(f"file duration {_fmt_hms(state.duration_s)}")
    pipeline.corelog.event(
        "preprocess",
        "metrics",
        duration_s=state.duration_s,
        snr_db=float(getattr(result.health, "snr_db", 0.0)) if result.health else None,
    )
    guard.done(duration_s=state.duration_s)

    # Save preprocessed audio to cache
    if not state.audio_sha16:
        state.audio_sha16 = compute_audio_sha16(state.y)
        if state.audio_sha16:
            pipeline.checkpoints.seed_file_hash(state.input_audio_path, state.audio_sha16)

    pipeline.checkpoints.create_checkpoint(
        state.input_audio_path,
        ProcessingStage.AUDIO_PREPROCESSING,
        {"sr": state.sr},
        progress=5.0,
        file_hash=state.audio_sha16,
    )

    cache_root = pipeline.cache_root
    if not state.audio_sha16:
        cache_dir = cache_root / "nohash"
    else:
        cache_dir = cache_root / state.audio_sha16
    cache_dir.mkdir(parents=True, exist_ok=True)
    state.cache_dir = cache_dir

    # Save preprocessed audio to cache
    try:
        _write_preprocessed_cache(pipeline, state, cache_dir)
        guard.progress(
            f"saved cache to {(cache_dir / 'preprocessed_audio.npy').as_posix()}"
        )
    except Exception as exc:
        pipeline.corelog.stage(
            "preprocess",
            "warn",
            message=f"failed to save cache: {exc}",
        )

    # Load diar/tx caches
    _load_diar_tx_caches(pipeline, state, cache_dir, guard)


def _load_diar_tx_caches(
    pipeline: AudioAnalysisPipelineV2,
    state: PipelineState,
    cache_dir: Path,
    guard: StageGuard | None = None,
) -> None:
    """Load diarization and transcription caches."""

    # Use corelog when no StageGuard is available
    def _progress(message: str) -> None:
        try:
            if guard is not None:
                guard.progress(message)
            else:
                pipeline.corelog.stage("resume", "progress", message=message)
        except Exception:
            # Logging must never fail the pipeline
            pass

    diar_path = cache_dir / "diar.json"
    tx_path = cache_dir / "tx.json"

    state.diar_cache = read_json_safe(diar_path)
    state.tx_cache = read_json_safe(tx_path)
    diar_cache_src = str(diar_path) if state.diar_cache else None
    tx_cache_src = str(tx_path) if state.tx_cache else None

    if not state.diar_cache or not state.tx_cache:
        for root in pipeline.cache_roots[1:]:
            alt_dir = Path(root) / state.audio_sha16
            alt_diar = read_json_safe(alt_dir / "diar.json")
            alt_tx = read_json_safe(alt_dir / "tx.json")
            if not state.diar_cache and alt_diar:
                state.diar_cache = alt_diar
                diar_cache_src = str(alt_dir / "diar.json")
            if not state.tx_cache and alt_tx:
                state.tx_cache = alt_tx
                tx_cache_src = str(alt_dir / "tx.json")
            if state.diar_cache and state.tx_cache:
                break

    def _cache_matches(obj: dict[str, object] | None) -> bool:
        return (
            bool(obj)
            and obj.get("version") == pipeline.cache_version
            and obj.get("audio_sha16") == state.audio_sha16
            and obj.get("pp_signature") == state.pp_sig
        )

    if _cache_matches(state.tx_cache):
        state.resume_tx = True
        state.resume_diar = bool(_cache_matches(state.diar_cache))
        if state.resume_diar:
            _progress("resume: using tx.json+diar.json caches; skipping diarize+ASR")
        else:
            _progress(
                "resume: using tx.json cache; skipping ASR and reconstructing turns from tx cache"
            )
        pipeline.corelog.event(
            "resume",
            "tx_cache_hit",
            audio_sha16=state.audio_sha16,
            src=tx_cache_src,
        )
    elif _cache_matches(state.diar_cache):
        state.resume_diar = True
        _progress("resume: using diar.json cache; skipping diarize")
        pipeline.corelog.event(
            "resume",
            "diar_cache_hit",
            audio_sha16=state.audio_sha16,
            src=diar_cache_src,
        )

    if pipeline.cfg.get("ignore_tx_cache"):
        state.diar_cache = None
        state.tx_cache = None
        state.resume_diar = False
        state.resume_tx = False


def _load_preprocessed_cache(
    pipeline: "AudioAnalysisPipelineV2",
    state: PipelineState,
    cache_dir: Path,
    guard: StageGuard,
) -> bool:
    """Load cached preprocessing artefacts if available and matching."""

    meta_path = cache_dir / "preprocessed.meta.json"
    audio_path = cache_dir / "preprocessed_audio.npy"
    legacy_path = cache_dir / "preprocessed.npz"

    def _matches(meta: dict[str, object] | None) -> bool:
        return bool(meta) and meta.get("pp_signature") == state.pp_sig

    if meta_path.exists() and audio_path.exists():
        meta = read_json_safe(meta_path)
        if not _matches(meta):
            return False

        try:
            audio = np.load(audio_path, mmap_mode="r")
        except ValueError:
            audio = np.load(audio_path)

        state.y = (
            audio
            if audio.dtype == np.float32
            else audio.astype(np.float32, copy=False)
        )
        state.sr = int(meta.get("sample_rate", 0) or 0)
        state.duration_s = float(meta.get("duration_s", 0.0) or 0.0)

        health_dict = meta.get("health") if isinstance(meta, dict) else None
        if isinstance(health_dict, dict):
            from ...pipeline.preprocess import AudioHealth

            state.health = AudioHealth(**health_dict)
        else:
            state.health = None

        guard.progress(
            f"loaded cached preprocessed audio {_fmt_hms(state.duration_s)}"
        )
        pipeline.corelog.event(
            "preprocess",
            "cache_hit",
            duration_s=state.duration_s,
            cache_path=str(audio_path),
        )
        guard.done(duration_s=state.duration_s)
        _load_diar_tx_caches(pipeline, state, cache_dir, guard)
        return True

    if legacy_path.exists():
        try:
            with np.load(legacy_path, allow_pickle=True) as cached:
                cached_sig = str(cached.get("pp_signature", ""))
                if cached_sig != state.pp_sig:
                    return False

                audio = cached["audio"].astype(np.float32)
                state.y = audio
                state.sr = int(cached["sample_rate"])
                state.duration_s = float(cached["duration_s"])

                if "health" in cached:
                    health_dict = cached["health"].item()
                    from ...pipeline.preprocess import AudioHealth

                    state.health = AudioHealth(**health_dict)
                else:
                    state.health = None

            guard.progress(
                f"loaded cached preprocessed audio {_fmt_hms(state.duration_s)}"
            )
            pipeline.corelog.event(
                "preprocess",
                "cache_hit",
                duration_s=state.duration_s,
                cache_path=str(legacy_path),
            )
            guard.done(duration_s=state.duration_s)

            try:
                _write_preprocessed_cache(pipeline, state, cache_dir)
                legacy_path.unlink(missing_ok=True)
            except Exception as exc:  # pragma: no cover - best effort
                pipeline.corelog.stage(
                    "preprocess",
                    "warn",
                    message=f"legacy cache upgrade failed: {exc}",
                )

            _load_diar_tx_caches(pipeline, state, cache_dir, guard)
            return True
        except Exception as exc:
            pipeline.corelog.stage(
                "preprocess",
                "warn",
                message=f"failed to load legacy cache: {exc}",
            )
            try:
                legacy_path.unlink(missing_ok=True)
            except TypeError:
                try:
                    legacy_path.unlink()
                except OSError:
                    pass
    return False


def _write_preprocessed_cache(
    pipeline: "AudioAnalysisPipelineV2",
    state: PipelineState,
    cache_dir: Path,
) -> None:
    """Persist preprocessing artefacts using a memory-mappable layout."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / "preprocessed_audio.npy"
    meta_path = cache_dir / "preprocessed.meta.json"

    y = np.asarray(state.y, dtype=np.float32)
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npy", delete=False) as tmp:
        np.save(tmp, y)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, audio_path)

    health_dict = None
    if state.health:
        health_dict = {
            "snr_db": state.health.snr_db,
            "clipping_detected": state.health.clipping_detected,
            "silence_ratio": state.health.silence_ratio,
            "rms_db": state.health.rms_db,
            "est_lufs": state.health.est_lufs,
            "dynamic_range_db": state.health.dynamic_range_db,
            "floor_clipping_ratio": state.health.floor_clipping_ratio,
            "is_chunked": getattr(state.health, "is_chunked", False),
            "chunk_info": getattr(state.health, "chunk_info", None),
        }

    meta_payload = {
        "version": pipeline.cache_version,
        "audio_sha16": state.audio_sha16,
        "pp_signature": state.pp_sig,
        "sample_rate": state.sr,
        "duration_s": state.duration_s,
        "dtype": "float32",
        "shape": list(y.shape),
        "health": health_dict,
    }
    atomic_write_json(meta_path, meta_payload)


def run_background_sed(
    pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard
) -> None:
    if not state.audio_sha16:
        if state.y.size:
            state.audio_sha16 = compute_audio_sha16(state.y)
    cache_dir = state.cache_dir
    if cache_dir is None:
        base = pipeline.cache_root
        key = state.audio_sha16 or "nohash"
        cache_dir = base / key
        cache_dir.mkdir(parents=True, exist_ok=True)
        state.cache_dir = cache_dir
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)

    sed_sig = compute_sed_signature(pipeline.cfg)
    sed_cache_path = cache_dir / "sed.json"
    cached = read_json_safe(sed_cache_path) if sed_cache_path.exists() else None
    if cached:
        matches = (
            cached.get("version") == pipeline.cache_version
            and cached.get("audio_sha16") == state.audio_sha16
            and cached.get("pp_signature") == state.pp_sig
            and cached.get("sed_signature") == sed_sig
            and cached.get("out_dir") == str(state.out_dir)
        )
        if matches:
            sed_info = cached.get("sed_info") or {}
            snapshot = dict(sed_info)
            events = snapshot.pop("timeline_events", None)
            if events is not None:
                snapshot["timeline_event_count"] = len(events)
            pipeline.stats.config_snapshot["background_sed"] = snapshot
            state.sed_info = sed_info
            guard.done()
            guard.progress(f"background SED reused cache from {sed_cache_path}")
            pipeline.corelog.event(
                "background_sed",
                "cache_hit",
                audio_sha16=state.audio_sha16,
                cache_path=str(sed_cache_path),
            )
            return

    empty_result = {"top": [], "dominant_label": None, "noise_score": 0.0}
    if not bool(pipeline.cfg.get("enable_sed", True)):
        disabled_result = dict(empty_result)
        disabled_result["enabled"] = False
        guard.progress("background sound event detection disabled via configuration")
        pipeline.stats.config_snapshot["background_sed"] = disabled_result
        state.sed_info = disabled_result
        guard.done()
        return

    sed_info = dict(empty_result)
    try:
        tagger = getattr(pipeline, "sed_tagger", None)
        if tagger is not None and state.y.size > 0 and state.sr:
            sed_info = dict(tagger.tag(state.y, state.sr) or empty_result)
            sed_info["enabled"] = True
            pipeline.corelog.event(
                "background_sed",
                "tags",
                dominant_label=sed_info.get("dominant_label"),
                noise_score=sed_info.get("noise_score"),
            )
        else:
            pipeline.corelog.stage(
                "background_sed",
                "warn",
                message="tagger unavailable; emitting empty background tag summary",
            )
            sed_info["enabled"] = True
    except (
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        pipeline.corelog.stage(
            "background_sed",
            "warn",
            message=f"tagging skipped: {exc}. Emitting empty background tag summary.",
        )
        sed_info["enabled"] = True
    finally:
        tl_cfg = {
            "mode": str(pipeline.cfg.get("sed_mode", "auto")).lower(),
            "window_sec": float(pipeline.cfg.get("sed_window_sec", 1.0)),
            "hop_sec": float(pipeline.cfg.get("sed_hop_sec", 0.5)),
            "enter": float(pipeline.cfg.get("sed_enter", 0.5)),
            "exit": float(pipeline.cfg.get("sed_exit", 0.35)),
            "min_dur": pipeline.cfg.get("sed_min_dur", {}),
            "default_min_dur": float(pipeline.cfg.get("sed_default_min_dur", 0.30)),
            "merge_gap": float(pipeline.cfg.get("sed_merge_gap", 0.20)),
            "classmap_csv": pipeline.cfg.get("sed_classmap_csv"),
            "write_jsonl": bool(pipeline.cfg.get("sed_timeline_jsonl", False)),
            "median_k": int(pipeline.cfg.get("sed_median_k", 5)),
            "batch_size": int(pipeline.cfg.get("sed_batch_size", 256)),
            # Cap total windows processed in timeline to avoid long hangs on multi-hour inputs
            "max_windows": int(pipeline.cfg.get("sed_max_windows", 6000)),
        }

        run_timeline = False
        noise_score = float(sed_info.get("noise_score", 0.0) or 0.0)
        if tl_cfg["mode"] == "timeline":
            run_timeline = True
        elif tl_cfg["mode"] == "auto":
            run_timeline = noise_score >= 0.30

        if run_timeline and state.out_dir is not None:
            try:
                from ...affect.sed_timeline import run_sed_timeline

                model_paths = getattr(tagger, "model_paths", None)
                labels = getattr(tagger, "labels", None)
                file_id = pipeline.stats.file_id or Path(state.input_audio_path).name
                artifacts = run_sed_timeline(
                    state.y,
                    sr=state.sr,
                    cfg=tl_cfg,
                    out_dir=state.out_dir,
                    file_id=file_id,
                    model_paths=model_paths,
                    labels=labels,
                )
                if artifacts is not None:
                    sed_info["timeline_csv"] = str(artifacts.csv)
                    sed_info["timeline_jsonl"] = str(artifacts.jsonl) if artifacts.jsonl else None
                    sed_info["timeline_events"] = artifacts.events
                    sed_info["timeline_mode"] = tl_cfg["mode"]
                    if getattr(artifacts, "mode", None):
                        sed_info["timeline_inference_mode"] = artifacts.mode
            except Exception as exc:  # pragma: no cover - runtime dependent
                pipeline.corelog.stage(
                    "background_sed",
                    "warn",
                    message=f"timeline generation failed: {exc}. Falling back to global tags only.",
                )

        sed_info.setdefault("enabled", True)
        snapshot = dict(sed_info)
        events = snapshot.pop("timeline_events", None)
        if events is not None:
            snapshot["timeline_event_count"] = len(events)
        pipeline.stats.config_snapshot["background_sed"] = snapshot
        state.sed_info = sed_info
        try:
            cache_payload = {
                "version": pipeline.cache_version,
                "audio_sha16": state.audio_sha16,
                "pp_signature": state.pp_sig,
                "sed_signature": sed_sig,
                "out_dir": str(state.out_dir),
                "sed_info": sed_info,
            }
            atomic_write_json(sed_cache_path, cache_payload)
            guard.progress(f"background SED cached results to {sed_cache_path}")
        except Exception as exc:  # pragma: no cover - best-effort cache
            pipeline.corelog.stage(
                "background_sed",
                "warn",
                message=f"failed to cache results: {exc}",
            )
        guard.done()
