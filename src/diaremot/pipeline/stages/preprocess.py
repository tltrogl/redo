"""Preprocessing stages (audio + background SED tagging)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..logging_utils import StageGuard, _fmt_hms
from ..pipeline_checkpoint_system import ProcessingStage
from .base import PipelineState
from .utils import (
    atomic_write_json,
    build_cache_payload,
    compute_audio_sha16,
    compute_audio_sha16_from_file,
    compute_pp_signature,
    compute_sed_signature,
    matches_pipeline_cache,
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
    state.sr = result.sample_rate
    state.health = result.health
    state.duration_s = float(result.duration_s)
    state.preprocessed_num_samples = result.num_samples
    if result.audio is not None:
        audio = result.audio
        if isinstance(audio, np.ndarray) and audio.dtype == np.float32:
            state.y = audio
        else:
            state.y = np.asarray(audio, dtype=np.float32)
    else:
        state.y = np.array([], dtype=np.float32)
    state.preprocessed_audio_path = Path(result.audio_path) if result.audio_path else None
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
        _write_preprocessed_cache(
            pipeline,
            state,
            cache_dir,
            source_path=state.preprocessed_audio_path,
            num_samples=result.num_samples,
        )
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

    if matches_pipeline_cache(
        state.tx_cache,
        version=pipeline.cache_version,
        audio_sha16=state.audio_sha16 or None,
        pp_signature=state.pp_sig,
        require_audio_sha=bool(state.audio_sha16),
    ):
        state.resume_tx = True
        state.resume_diar = bool(
            matches_pipeline_cache(
                state.diar_cache,
                version=pipeline.cache_version,
                audio_sha16=state.audio_sha16 or None,
                pp_signature=state.pp_sig,
                require_audio_sha=bool(state.audio_sha16),
            )
        )
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
    elif matches_pipeline_cache(
        state.diar_cache,
        version=pipeline.cache_version,
        audio_sha16=state.audio_sha16 or None,
        pp_signature=state.pp_sig,
        require_audio_sha=bool(state.audio_sha16),
    ):
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
        return matches_pipeline_cache(
            meta or {},
            version=pipeline.cache_version,
            audio_sha16=state.audio_sha16 or None,
            pp_signature=state.pp_sig,
            require_version=False,
            require_audio_sha=bool(state.audio_sha16),
        )

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
        shape = meta.get("shape")
        if isinstance(shape, list) and shape:
            state.preprocessed_num_samples = int(shape[0])
        else:
            state.preprocessed_num_samples = int(state.y.shape[0])
        state.preprocessed_audio_path = audio_path

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
                state.preprocessed_num_samples = int(audio.shape[0])

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
                _write_preprocessed_cache(
                    pipeline,
                    state,
                    cache_dir,
                    source_path=None,
                    num_samples=int(state.y.shape[0]),
                )
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
    *,
    source_path: Path | None = None,
    num_samples: int | None = None,
) -> None:
    """Persist preprocessing artefacts using a memory-mappable layout."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / "preprocessed_audio.npy"
    meta_path = cache_dir / "preprocessed.meta.json"

    target_path = audio_path
    if source_path and source_path.exists():
        if source_path.resolve() != target_path.resolve():
            shutil.move(str(source_path), target_path)
    else:
        y = np.asarray(state.y, dtype=np.float32)
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npy", delete=False) as tmp:
            np.save(tmp, y)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)

        os.replace(tmp_path, target_path)
        if num_samples is None:
            num_samples = int(y.shape[0])

    state.preprocessed_audio_path = target_path
    if num_samples is None:
        if state.preprocessed_num_samples is not None:
            num_samples = int(state.preprocessed_num_samples)
        else:
            num_samples = int(state.y.shape[0]) if state.y.size else 0
    state.preprocessed_num_samples = num_samples

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

    meta_payload = build_cache_payload(
        version=pipeline.cache_version,
        audio_sha16=state.audio_sha16,
        pp_signature=state.pp_sig,
        extra={
            "sample_rate": state.sr,
            "duration_s": state.duration_s,
            "dtype": "float32",
            "shape": [int(num_samples)],
            "health": health_dict,
        },
    )
    atomic_write_json(meta_path, meta_payload)


def _persist_timeline_events(
    cache_dir: Path,
    events: Sequence[Any],
    existing_path: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Persist timeline events to a dedicated JSON file.

    Returns the concrete path if any events were written. When ``existing_path``
    is provided the payload is rewritten in place to upgrade legacy caches.
    """

    if not events:
        return None

    if existing_path:
        events_path = Path(existing_path)
        try:
            events_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    else:
        events_path = cache_dir / "sed.timeline_events.json"

    payload = {"events": list(events)}
    atomic_write_json(events_path, payload)
    return events_path


def _hydrate_timeline_events(cache_dir: Path, sed_info: dict[str, Any]) -> None:
    """Populate in-memory timeline events from cached artefacts."""

    if not isinstance(sed_info, dict):
        return

    direct_events = sed_info.get("timeline_events")
    if isinstance(direct_events, list):
        if not sed_info.get("timeline_events_path"):
            try:
                events_path = _persist_timeline_events(cache_dir, direct_events)
            except Exception:  # pragma: no cover - cache hydration best effort
                events_path = None
            if events_path is not None:
                sed_info["timeline_events_path"] = str(events_path)
        sed_info["timeline_event_count"] = len(direct_events)
        return

    events_path = sed_info.get("timeline_events_path")
    if not events_path:
        return

    try:
        path_obj = Path(events_path)
        if not path_obj.is_absolute():
            candidate = cache_dir / path_obj
            if candidate.exists():
                path_obj = candidate
        if not path_obj.exists():
            return
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return

    events: Any
    if isinstance(payload, dict):
        events = payload.get("events")
    else:
        events = payload

    if isinstance(events, list):
        sed_info["timeline_events"] = events
        sed_info["timeline_event_count"] = len(events)


def _summarize_timeline_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate duration and score statistics for timeline events."""

    total_duration = 0.0
    total_weight = 0.0
    label_durations: dict[str, float] = {}
    label_weights: dict[str, float] = {}

    for raw_event in events:
        if not isinstance(raw_event, Mapping):
            continue
        label = str(raw_event.get("label") or "").strip()
        try:
            start = float(raw_event.get("start", 0.0))
            end = float(raw_event.get("end", 0.0))
        except Exception:
            continue
        duration = max(0.0, float(raw_event.get("duration", end - start)))
        try:
            score = float(raw_event.get("score", 0.0))
        except Exception:
            score = 0.0
        weight = float(raw_event.get("weight", score * duration))

        total_duration += duration
        total_weight += max(0.0, weight)

        if label:
            label_durations[label] = label_durations.get(label, 0.0) + duration
            label_weights[label] = label_weights.get(label, 0.0) + max(0.0, weight)

    label_mean_scores = {
        key: (label_weights[key] / duration if duration > 0 else 0.0)
        for key, duration in label_durations.items()
    }

    return {
        "timeline_total_duration": total_duration,
        "timeline_total_weight": total_weight,
        "timeline_label_durations": label_durations,
        "timeline_label_mean_scores": label_mean_scores,
    }


def _ensure_timeline_summaries(cache_dir: Path, sed_info: dict[str, Any]) -> None:
    """Populate aggregate timeline metrics when cached data lacks them."""

    if not sed_info or not sed_info.get("timeline_event_count"):
        return

    if all(key in sed_info for key in ("timeline_total_duration", "timeline_label_durations")):
        return

    events: Sequence[Mapping[str, Any]] | None = None
    events_path_raw = sed_info.get("timeline_events_path")
    if events_path_raw:
        payload = read_json_safe(Path(events_path_raw))
        if isinstance(payload, Mapping):
            raw_events = payload.get("events")
            if isinstance(raw_events, Sequence):
                collected: list[Mapping[str, Any]] = []
                for item in raw_events:
                    if isinstance(item, Mapping):
                        collected.append(item)
                events = collected

    if events is None:
        fallback_payload = read_json_safe(cache_dir / "sed.timeline_events.json")
        if isinstance(fallback_payload, Mapping):
            raw_events = fallback_payload.get("events")
            if isinstance(raw_events, Sequence):
                collected: list[Mapping[str, Any]] = []
                for item in raw_events:
                    if isinstance(item, Mapping):
                        collected.append(item)
                events = collected

    if events:
        sed_info.update(_summarize_timeline_events(events))


def _ensure_list(value: Any) -> list[Any]:
    """
    Ensure value is a proper list (not str/bytes), returning [] if not.
    
    Helper to normalize values that should be lists but might be missing,
    wrong type, or accidentally a string/bytes.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return list(value)



def run_background_sed(
    pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard
) -> None:
    state.ensure_audio()
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

    cfg_obj = pipeline.cfg
    if isinstance(cfg_obj, Mapping):
        rank_limit_raw = cfg_obj.get("sed_rank_export_limit")
    else:
        rank_limit_raw = getattr(cfg_obj, "sed_rank_export_limit", None)
    rank_limit_value: int | None
    if rank_limit_raw is None:
        rank_limit_value = None
    else:
        try:
            rank_limit_value = int(rank_limit_raw)
        except (TypeError, ValueError):
            pipeline.corelog.stage(
                "background_sed",
                "warn",
                message=f"invalid sed_rank_export_limit value {rank_limit_raw!r}; ignoring",
            )
            rank_limit_value = None

    sed_sig = compute_sed_signature(pipeline.cfg)
    sed_cache_path = cache_dir / "sed.json"
    cached = read_json_safe(sed_cache_path) if sed_cache_path.exists() else None
    if cached:
        extra_requirements: dict[str, Any] = {"sed_signature": sed_sig}
        if state.out_dir is not None:
            extra_requirements["out_dir"] = str(state.out_dir)
        if matches_pipeline_cache(
            cached,
            version=pipeline.cache_version,
            audio_sha16=state.audio_sha16 or None,
            pp_signature=state.pp_sig,
            extra=extra_requirements,
            require_audio_sha=bool(state.audio_sha16),
        ):
            sed_info = cached.get("sed_info") or {}
            _hydrate_timeline_events(cache_dir, sed_info)
            sed_info.setdefault("timeline_event_count", 0)
            
            # Normalize tagger metadata
            top_entries = _ensure_list(sed_info.get("top"))
            sed_info["top"] = top_entries
            
            try:
                cached_top_k = int(sed_info.get("tagger_top_k"))
            except (TypeError, ValueError):
                cached_top_k = len(top_entries)
            sed_info["tagger_top_k"] = cached_top_k
            
            cached_rank_limit = sed_info.get("tagger_rank_limit")
            if cached_rank_limit is None:
                sed_info["tagger_rank_limit"] = rank_limit_value
            else:
                try:
                    sed_info["tagger_rank_limit"] = int(cached_rank_limit)
                except (TypeError, ValueError):
                    sed_info["tagger_rank_limit"] = rank_limit_value
            
            ranking_entries = sed_info.get("tagger_ranking")
            if ranking_entries is not None:
                ranking_list = _ensure_list(ranking_entries)
                sed_info["tagger_ranking"] = ranking_list
            sed_info["tagger_ranking_size"] = int(
                sed_info.get("tagger_ranking_size") or len(_ensure_list(ranking_entries))
            )
            
            _ensure_timeline_summaries(cache_dir, sed_info)
            
            sed_info.setdefault("tagger_available", bool(sed_info.get("tagger_available")))
            sed_info.setdefault("tagger_backend", sed_info.get("tagger_backend"))
            sed_info.setdefault("tagger_error", sed_info.get("tagger_error"))
            
            snapshot = dict(sed_info)
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

    empty_result = {
        "top": [],
        "dominant_label": None,
        "noise_score": 0.0,
        "tagger_top_k": 0,
        "tagger_rank_limit": rank_limit_value,
        "tagger_ranking": [],
        "tagger_ranking_size": 0,
    }
    if not bool(pipeline.cfg.get("enable_sed", True)):
        disabled_result = dict(empty_result)
        disabled_result["enabled"] = False
        disabled_result["tagger_ranking"] = []
        guard.progress("background sound event detection disabled via configuration")
        pipeline.stats.config_snapshot["background_sed"] = disabled_result
        state.sed_info = disabled_result
        guard.done()
        return

    sed_info = dict(empty_result)
    sed_info["tagger_ranking"] = []
    
    tagger_backend = getattr(tagger, "backend", None) if (tagger := getattr(pipeline, "sed_tagger", None)) is not None else None
    tagger_available = bool(getattr(tagger, "available", tagger is not None)) if tagger is not None else False
    sed_info["tagger_backend"] = tagger_backend
    sed_info["tagger_available"] = tagger_available
    sed_info["tagger_error"] = None
    
    try:
        if tagger is not None and state.y.size > 0 and state.sr:
            result = tagger.tag(state.y, state.sr, rank_limit=rank_limit_value) or empty_result
            sed_info = dict(result)
            sed_info.setdefault("tagger_ranking", [])
            sed_info["enabled"] = True
            sed_info["tagger_backend"] = tagger_backend
            sed_info["tagger_available"] = tagger_available
            sed_info["tagger_error"] = None
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
            sed_info["tagger_error"] = "tagger unavailable"
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
        sed_info["tagger_error"] = str(exc)
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
                    sed_info["timeline_mode"] = tl_cfg["mode"]
                    if getattr(artifacts, "mode", None):
                        sed_info["timeline_inference_mode"] = artifacts.mode

                    events = list(getattr(artifacts, "events", []) or [])
                    if events:
                        events_path = None
                        try:
                            events_path = _persist_timeline_events(cache_dir, events)
                        except Exception as exc:  # pragma: no cover - best-effort cache
                            pipeline.corelog.stage(
                                "background_sed",
                                "warn",
                                message=f"failed to persist timeline events: {exc}",
                            )
                        sed_info["timeline_event_count"] = len(events)
                        sed_info["timeline_events"] = events
                        if events_path is not None:
                            sed_info["timeline_events_path"] = str(events_path)
                        else:
                            sed_info.pop("timeline_events_path", None)
                    else:
                        sed_info["timeline_event_count"] = 0
                        sed_info.pop("timeline_events_path", None)
                        sed_info.pop("timeline_events", None)
            except Exception as exc:  # pragma: no cover - runtime dependent
                pipeline.corelog.stage(
                    "background_sed",
                    "warn",
                    message=f"timeline generation failed: {exc}. Falling back to global tags only.",
                )

        sed_info.setdefault("enabled", True)
        sed_info.setdefault("timeline_event_count", 0)
        snapshot = dict(sed_info)
        pipeline.stats.config_snapshot["background_sed"] = snapshot
        state.sed_info = sed_info
        try:
            cache_sed_info = dict(sed_info)
            cache_sed_info.pop("timeline_events", None)
            extra_payload: dict[str, Any] = {
                "sed_signature": sed_sig,
                "sed_info": cache_sed_info,
            }
            if state.out_dir is not None:
                extra_payload["out_dir"] = str(state.out_dir)
            cache_payload = build_cache_payload(
                version=pipeline.cache_version,
                audio_sha16=state.audio_sha16,
                pp_signature=state.pp_sig,
                extra=extra_payload,
            )
            atomic_write_json(sed_cache_path, cache_payload)
            guard.progress(f"background SED cached results to {sed_cache_path}")
        except Exception as exc:  # pragma: no cover - best-effort cache
            pipeline.corelog.stage(
                "background_sed",
                "warn",
                message=f"failed to cache results: {exc}",
            )
        guard.done()
