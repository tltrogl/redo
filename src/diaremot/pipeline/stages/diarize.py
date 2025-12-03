"""Speaker diarization stage."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..logging_utils import StageGuard
from ..pipeline_checkpoint_system import ProcessingStage
from .base import PipelineState
from .utils import atomic_write_json, build_cache_payload

if TYPE_CHECKING:
    from ..orchestrator import AudioAnalysisPipelineV2

__all__ = ["run"]


def _fallback_turn(duration_s: float) -> dict[str, Any]:
    return {
        "start": 0.0,
        "end": duration_s,
        "speaker": "Speaker_1",
        "speaker_name": "Speaker_1",
    }


def _prepare_turn_cache(
    turns: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    json_turns: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []

    for turn in turns:
        try:
            tcopy = dict(turn)
        except (TypeError, ValueError, AttributeError):
            continue

        emb = tcopy.pop("embedding", None)
        if emb is not None:
            try:
                vec = np.asarray(emb, dtype=np.float32)
                if vec.ndim == 1:
                    embeddings.append(vec)
                    tcopy["embedding_idx"] = len(embeddings) - 1
            except Exception:
                # Ignore malformed embeddings
                pass

        json_turns.append(tcopy)

    if not embeddings:
        return json_turns, None

    try:
        return json_turns, np.stack(embeddings, axis=0)
    except ValueError:
        # Embeddings with mismatched shapes - fall back to no embedding cache
        return json_turns, None


def _rehydrate_embeddings(
    cache_dir: Path | None, turns: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    if not cache_dir or not turns:
        return turns or []

    emb_path = cache_dir / "diar_embeddings.npz"
    if not emb_path.exists():
        for turn in turns:
            turn.pop("embedding_idx", None)
        return turns

    try:
        with np.load(emb_path) as data:
            embeddings = data.get("embeddings")
            if embeddings is None:
                embeddings = data.get("arr_0")
    except Exception:
        for turn in turns:
            turn.pop("embedding_idx", None)
        return turns

    for turn in turns:
        idx = turn.pop("embedding_idx", None)
        if idx is None:
            continue
        try:
            turn["embedding"] = embeddings[int(idx)].astype(np.float32, copy=False)
        except Exception:
            turn["embedding"] = None

    return turns


def _speaker_metrics(turns: list[dict[str, Any]]) -> tuple[int, int]:
    speakers = {
        str(turn.get("speaker", "")).strip()
        for turn in turns
        if isinstance(turn, dict) and turn.get("speaker")
    }
    return len(speakers), len(turns)


def _extract_vad_flips(stats: Any) -> float:
    if not isinstance(stats, dict):
        return 0.0
    for key in ("vad_boundary_flips", "vad_flip_count", "vad_flips"):
        if key in stats:
            try:
                return float(stats[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _coerce_vad_stats(stats: Any) -> dict[str, float]:
    if not isinstance(stats, dict):
        return {}
    result: dict[str, float] = {}
    for key in ("vad_boundary_flips", "speech_regions", "analyzed_duration_sec"):
        if key in stats:
            try:
                result[key] = float(stats[key])
            except (TypeError, ValueError):
                continue
    return result


def _is_vad_unstable(vad_flips: float, duration_s: float) -> bool:
    try:
        flips = float(vad_flips)
        minutes = max(1, int(duration_s / 60) or 1)
    except (TypeError, ValueError):
        return False
    return (flips / minutes) > 60.0


def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    turns: list[dict[str, Any]] = []
    duration_s = state.duration_s

    state.ensure_audio()

    # Emit diagnostics about the diarization configuration and state
    try:
        cfg = getattr(pipeline, "diar", None).config if getattr(pipeline, "diar", None) else None
        diag = {
            "resume_tx": bool(state.resume_tx),
            "resume_diar": bool(state.resume_diar),
            "duration_s": float(duration_s),
        }
        if cfg is not None:
            diag.update(
                {
                    "vad_threshold": getattr(cfg, "vad_threshold", None),
                    "vad_min_speech_sec": getattr(cfg, "vad_min_speech_sec", None),
                    "vad_min_silence_sec": getattr(cfg, "vad_min_silence_sec", None),
                    "speaker_limit": getattr(cfg, "speaker_limit", None),
                    "clustering_backend": getattr(cfg, "clustering_backend", None),
                    "allow_energy_vad_fallback": getattr(cfg, "allow_energy_vad_fallback", None),
                    "single_speaker_collapse": getattr(cfg, "single_speaker_collapse", None),
                }
            )
        pipeline.corelog.stage(
            "diarize", "progress", message="diarize config/diagnostics", diagnostics=diag
        )
    except Exception:
        # Don't fail the stage for logging errors
        try:
            pipeline.corelog.stage(
                "diarize", "debug", message="diarize diagnostics generation failed"
            )
        except Exception:
            pass

    if (
        state.resume_tx
        and not state.diar_cache
        and state.tx_cache
        and state.tx_cache.get("segments")
    ):
        tx_segments = state.tx_cache.get("segments", []) or []
        for entry in tx_segments:
            try:
                start = float(entry.get("start", entry.get("start_time", 0.0)) or 0.0)
                end = float(entry.get("end", entry.get("end_time", 0.0)) or 0.0)
            except Exception:
                start, end = 0.0, 0.0
            if end < start:
                start, end = end, start
            speaker_id = str(entry.get("speaker_id", entry.get("speaker", "Speaker_1")))
            speaker_name = entry.get("speaker_name") or speaker_id
            turns.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker_id,
                    "speaker_name": speaker_name,
                }
            )
        if not turns:
            turns.append(_fallback_turn(duration_s))
        state.vad_unstable = False
        speakers_est, turn_count = _speaker_metrics(turns)
        guard.progress(
            f"resume (tx cache) estimated {speakers_est} speakers across {turn_count} turns",
        )
        try:
            pipeline.corelog.stage(
                "diarize",
                "progress",
                message="resume tx cache details",
                diagnostics={
                    "tx_cache_present": bool(state.tx_cache),
                    "tx_segments_count": len(tx_segments),
                },
            )
        except Exception:
            pass
        guard.done(turns=turn_count, speakers_est=speakers_est)
    elif state.resume_diar and state.diar_cache:
        turns = state.diar_cache.get("turns", []) or []
        turns = _rehydrate_embeddings(state.cache_dir, turns)
        if not turns:
            pipeline.corelog.stage(
                "diarize",
                "warn",
                message="Cached diar.json has 0 turns; proceeding with fallback",
            )
            turns = [_fallback_turn(duration_s)]
        cached_stats = _coerce_vad_stats(state.diar_cache.get("diagnostics"))
        cached_flips = _extract_vad_flips(cached_stats)
        state.vad_unstable = _is_vad_unstable(cached_flips, duration_s)
        speakers_est, turn_count = _speaker_metrics(turns)
        guard.progress(
            f"resume (diar cache) estimated {speakers_est} speakers across {turn_count} turns",
        )
        guard.done(turns=turn_count, speakers_est=speakers_est)
        try:
            pipeline.corelog.stage(
                "diarize",
                "progress",
                message="resume diar cache details",
                diagnostics={
                    "diar_cache_present": bool(state.diar_cache),
                    "diar_cache_turns": len(turns),
                    "cached_vad_stats": cached_stats,
                },
            )
        except Exception:
            pass
    else:
        vad_stats: dict[str, float] = {}
        try:
            # If configured, prefer SED timeline for diarization speech regions.
            use_sed_for_split = False
            try:
                cfg_val = (
                    pipeline.cfg.get("diar_use_sed_timeline") if hasattr(pipeline, "cfg") else None
                )
            except Exception:
                cfg_val = getattr(pipeline.cfg, "diar_use_sed_timeline", False)
            use_sed_for_split = bool(cfg_val)

            sed_regions = None
            if use_sed_for_split and state.sed_info:
                events_path = state.sed_info.get("timeline_events_path") or state.sed_info.get(
                    "timeline_events"
                )
                event_list = None
                if events_path and isinstance(events_path, str):
                    try:
                        with open(events_path, encoding="utf-8") as fh:
                            payload = json.load(fh)
                            event_list = (
                                payload.get("events") if isinstance(payload, dict) else None
                            )
                    except Exception:
                        event_list = None
                elif events_path and isinstance(events_path, list):
                    event_list = events_path
                if event_list:
                    # Build speech-like regions from SED timeline using simple heuristics
                    speech_like = {"speech", "conversation", "crowd", "talk", "voice", "dialogue"}
                    min_map = (
                        getattr(pipeline.cfg, "sed_min_dur", {}) if hasattr(pipeline, "cfg") else {}
                    )
                    default_min = float(getattr(pipeline.cfg, "sed_default_min_dur", 0.3) or 0.3)
                    pad = float(getattr(pipeline.cfg, "vad_speech_pad_sec", 0.0) or 0.0)
                    regions: list[tuple[float, float]] = []
                    for ev in event_list:
                        try:
                            label = str(ev.get("label", "")).strip().lower()
                            start = float(ev.get("start", 0.0) or 0.0)
                            end = float(ev.get("end", 0.0) or 0.0)
                        except Exception:
                            continue
                        duration = max(0.0, end - start)
                        min_dur = float(min_map.get(label, default_min) or default_min)
                        if duration < min_dur:
                            continue
                        if (
                            any(k in label for k in speech_like)
                            or str(state.sed_info.get("dominant_label", "")).strip().lower()
                            in label
                        ):
                            s = max(0.0, start - pad)
                            e = end + pad
                            regions.append((s, e))
                    # Merge overlapping / adjacent regions if necessary
                    regions.sort()
                    merged: list[tuple[float, float]] = []
                    for s, e in regions:
                        if not merged:
                            merged.append((s, e))
                        else:
                            last_s, last_e = merged[-1]
                            if s <= last_e + 0.1:
                                merged[-1] = (last_s, max(last_e, e))
                            else:
                                merged.append((s, e))
                    if merged:
                        sed_regions = merged
                        try:
                            # compute simple region stats
                            durations = [round(e - s, 3) for s, e in sed_regions]
                            total_dur = round(sum(durations), 3)
                            min_dur = round(min(durations or [0.0]), 3)
                            max_dur = round(max(durations or [0.0]), 3)
                            median_dur = round(np.median(durations).item() if durations else 0.0, 3)
                            pipeline.corelog.stage(
                                "diarize",
                                "progress",
                                message="using SED timeline for diarization speech regions",
                                diagnostics={
                                    "sed_regions": len(sed_regions),
                                    "sed_regions_total_s": total_dur,
                                    "sed_regions_min_s": min_dur,
                                    "sed_regions_median_s": median_dur,
                                    "sed_regions_max_s": max_dur,
                                    "diar_use_sed_timeline": bool(use_sed_for_split),
                                },
                            )
                        except Exception:
                            pass
            region_hint = "sed_timeline" if sed_regions else None
            turns = (
                pipeline.diar.diarize_audio(
                    state.y, state.sr, speech_regions=sed_regions, region_source=region_hint
                )
                or []
            )
            # If SED-driven splitting was requested but no timeline was available, log it
            if use_sed_for_split and not state.sed_info:
                try:
                    pipeline.corelog.stage(
                        "diarize",
                        "debug",
                        message="diar_use_sed_timeline requested but no state.sed_info present",
                    )
                except Exception:
                    pass
            elif use_sed_for_split and state.sed_info and not event_list:
                try:
                    pipeline.corelog.stage(
                        "diarize",
                        "debug",
                        message="diar_use_sed_timeline requested but parsed SED timeline event list is empty",
                        diagnostics={"sed_info_present": True},
                    )
                except Exception:
                    pass
        except (
            RuntimeError,
            ValueError,
            OSError,
            subprocess.CalledProcessError,
        ) as exc:
            pipeline.corelog.stage(
                "diarize",
                "warn",
                message=(
                    "Diarization failed: "
                    f"{exc}; reverting to single-speaker assumption. Verify ECAPA/pyannote assets."
                ),
            )
            try:
                pipeline.corelog.stage(
                    "diarize",
                    "warn",
                    message="Diarization exception diagnostic",
                    diagnostics={"exception_type": type(exc).__name__, "exception_str": str(exc)},
                )
            except Exception:
                pass
            turns = []
        if not turns:
            try:
                pipeline.corelog.stage(
                    "diarize",
                    "warn",
                    message="Diarizer returned 0 turns; using fallback",
                    diagnostics={
                        "vad_stats": vad_stats or {},
                        "resume_tx": bool(state.resume_tx),
                        "resume_diar": bool(state.resume_diar),
                    },
                )
            except Exception:
                pipeline.corelog.stage(
                    "diarize", "warn", message="Diarizer returned 0 turns; using fallback"
                )
            turns = [_fallback_turn(duration_s)]
        stats_getter = getattr(pipeline.diar, "get_vad_statistics", None)
        if callable(stats_getter):
            try:
                vad_stats = _coerce_vad_stats(stats_getter())
            except Exception:
                vad_stats = {}
        flip_count = _extract_vad_flips(vad_stats)
        state.vad_unstable = _is_vad_unstable(flip_count, duration_s)
        speakers_est, turn_count = _speaker_metrics(turns)
        guard.progress(f"estimated {speakers_est} speakers across {turn_count} turns")
        guard.done(turns=turn_count, speakers_est=speakers_est)

        # Emit additional diagnostics when diarization collapsed to single speaker
        try:
            if speakers_est <= 1:
                embeddings = []
                seg_embs = []
                try:
                    seg_embs = getattr(pipeline.diar, "get_segment_embeddings", lambda: [])() or []
                except Exception:
                    seg_embs = []
                embedding_count = len([e for e in seg_embs if e.get("embedding") is not None])
                diag = {
                    "speakers_est": int(speakers_est),
                    "turn_count": int(turn_count),
                    "embedding_count": int(embedding_count),
                    "vad_stats": vad_stats or {},
                    "resume_tx": bool(state.resume_tx),
                    "resume_diar": bool(state.resume_diar),
                }
                pipeline.corelog.stage(
                    "diarize",
                    "warn",
                    message="Diarization collapsed to single speaker; diagnostic payload",
                    diagnostics=diag,
                )
        except Exception:
            try:
                pipeline.corelog.stage(
                    "diarize", "debug", message="diarize single-speaker diagnostics failed"
                )
            except Exception:
                pass

        debug_payload: dict[str, Any] | None = None
        debug_getter = getattr(pipeline.diar, "get_debug_payload", None)
        if callable(debug_getter):
            try:
                debug_payload = debug_getter() or {}
            except Exception as exc:
                debug_payload = None
                try:
                    pipeline.corelog.stage(
                        "diarize",
                        "debug",
                        message="diarize debug payload unavailable",
                        diagnostics={"error": str(exc)},
                    )
                except Exception:
                    pass
        if debug_payload:
            try:
                diag_dir = state.out_dir / "diagnostics"
                diag_dir.mkdir(parents=True, exist_ok=True)
                diag_path = diag_dir / "diarization_debug.json"
                atomic_write_json(diag_path, debug_payload)
                summary = {
                    "debug_artifact": str(diag_path),
                    "speaker_count": debug_payload.get("turns", {}).get("speaker_count"),
                }
                pipeline.corelog.stage(
                    "diarize",
                    "progress",
                    message="Diarization debug artifact written",
                    diagnostics=summary,
                )
            except Exception as exc:  # pragma: no cover - diagnostics best-effort
                try:
                    pipeline.corelog.stage(
                        "diarize",
                        "debug",
                        message="Failed to persist diarization debug artifact",
                        diagnostics={"error": str(exc)},
                    )
                except Exception:
                    pass

        if state.cache_dir:
            try:
                turns_json, embeddings = _prepare_turn_cache(turns)
                payload = build_cache_payload(
                    version=pipeline.cache_version,
                    audio_sha16=state.audio_sha16,
                    pp_signature=state.pp_sig,
                    extra={
                        "turns": turns_json,
                        "saved_at": time.time(),
                    },
                )
                if vad_stats:
                    payload["diagnostics"] = vad_stats
                atomic_write_json(state.cache_dir / "diar.json", payload)
                if embeddings is not None:
                    np.savez_compressed(
                        state.cache_dir / "diar_embeddings.npz",
                        embeddings=embeddings,
                    )
            except OSError as exc:
                pipeline.corelog.stage(
                    "diarize",
                    "warn",
                    message=f"[cache] diar.json write failed: {exc}. Ensure cache directory is writable.",
                )
            except Exception as exc:  # pragma: no cover - cache best effort
                pipeline.corelog.stage(
                    "diarize",
                    "warn",
                    message=f"[cache] diar embedding write failed: {exc}",
                )

    if not turns:
        turns = [_fallback_turn(duration_s)]

    state.turns = turns
    pipeline.checkpoints.create_checkpoint(
        state.input_audio_path,
        ProcessingStage.DIARIZATION,
        turns,
        progress=30.0,
    )
