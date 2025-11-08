"""Speaker diarization stage."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..logging_utils import StageGuard
from ..pipeline_checkpoint_system import ProcessingStage
from .base import PipelineState
from .utils import atomic_write_json

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
    turns: list[dict[str, Any]]
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


def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    turns: list[dict[str, Any]] = []
    duration_s = state.duration_s

    state.ensure_audio()

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
        vad_toggles = 0
        try:
            vad_toggles = sum(1 for t in turns if t.get("is_boundary_flip"))
        except (TypeError, AttributeError):
            vad_toggles = 0
        state.vad_unstable = (vad_toggles / max(1, int(duration_s / 60) or 1)) > 60
        speakers_est, turn_count = _speaker_metrics(turns)
        guard.progress(
            f"resume (diar cache) estimated {speakers_est} speakers across {turn_count} turns",
        )
        guard.done(turns=turn_count, speakers_est=speakers_est)
    else:
        try:
            turns = pipeline.diar.diarize_audio(state.y, state.sr) or []
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
            turns = []
        if not turns:
            pipeline.corelog.stage("diarize", "warn", message="Diarizer returned 0 turns; using fallback")
            turns = [_fallback_turn(duration_s)]

        vad_toggles = 0
        try:
            vad_toggles = sum(1 for t in turns if t.get("is_boundary_flip"))
        except (TypeError, AttributeError):
            vad_toggles = 0
        state.vad_unstable = (vad_toggles / max(1, int(duration_s / 60) or 1)) > 60
        speakers_est, turn_count = _speaker_metrics(turns)
        guard.progress(f"estimated {speakers_est} speakers across {turn_count} turns")
        guard.done(turns=turn_count, speakers_est=speakers_est)

        if state.cache_dir:
            try:
                turns_json, embeddings = _prepare_turn_cache(turns)
                payload = {
                    "version": pipeline.cache_version,
                    "audio_sha16": state.audio_sha16,
                    "pp_signature": state.pp_sig,
                    "turns": turns_json,
                    "saved_at": time.time(),
                }
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
