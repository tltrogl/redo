from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import scipy.signal

from .clustering import build_agglo
from .config import DiarizationConfig, DiarizedTurn
from .embeddings import ECAPAEncoder
from .logger import logger
from .registry import SpeakerRegistry
from .segments import collapse_single_speaker_turns
from .utils import energy_vad_fallback
from .vad import SileroVAD

_MIN_PROGRESS_INTERVAL = 5.0


@contextmanager
def _periodic_progress_logger(
    name: str,
    *,
    start: float,
    interval: float,
    total: int | None = None,
) -> None:
    """Emit heartbeat logs for long-running clustering steps."""
    safe_interval = max(_MIN_PROGRESS_INTERVAL, float(interval))
    stop_event = threading.Event()

    def _tick() -> None:
        while not stop_event.wait(safe_interval):
            elapsed = time.perf_counter() - start
            if total is not None:
                logger.info(
                    "%s still running (elapsed %.1fs; embeddings=%d)",
                    name,
                    elapsed,
                    total,
                )
            else:
                logger.info("%s still running (elapsed %.1fs)", name, elapsed)

    worker = threading.Thread(target=_tick, name=f"{name}-progress", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop_event.set()
        worker.join(timeout=safe_interval)


def _run_with_progress(
    name: str,
    func: Callable[[], Any],
    *,
    total: int | None,
    interval: float | None,
) -> tuple[Any, float]:
    """Execute a callable while emitting periodic progress logs."""
    start = time.perf_counter()
    should_log = interval is not None and float(interval) > 0.0
    if should_log:
        with _periodic_progress_logger(
            name,
            start=start,
            interval=float(interval),
            total=total,
        ):
            result = func()
    else:
        result = func()
    elapsed = time.perf_counter() - start
    return result, elapsed


class SpeakerDiarizer:
    def __init__(self, config: DiarizationConfig):
        self.config = config
        self.vad = SileroVAD(
            config.vad_threshold,
            config.speech_pad_sec,
            backend=getattr(config, "vad_backend", "auto"),
        )
        self.ecapa = ECAPAEncoder(config.ecapa_model_path)
        self.registry = None
        if config.registry_path:
            try:
                self.registry = SpeakerRegistry(config.registry_path)
                logger.info("Registry loaded: %s", config.registry_path)
            except Exception as exc:
                logger.warning("Registry unavailable: %s", exc)
        self._last_turns: list[DiarizedTurn] = []
        self._last_vad_stats: dict[str, float] = {
            "vad_boundary_flips": 0.0,
            "speech_regions": 0.0,
            "analyzed_duration_sec": 0.0,
        }
        self._debug_payload: dict[str, Any] = {}

    def get_segment_embeddings(self) -> list[dict[str, Any]]:
        return [
            {"speaker": t.speaker, "embedding": t.embedding}
            for t in self._last_turns
            if t.embedding is not None
        ]

    def get_vad_statistics(self) -> dict[str, float]:
        """Return diagnostics collected during the most recent VAD pass."""

        return dict(self._last_vad_stats)

    def get_debug_payload(self) -> dict[str, Any]:
        """Return structured debugging information about the most recent diarization run."""

        payload = dict(self._debug_payload)
        payload["vad_stats"] = self.get_vad_statistics()
        return payload

    @staticmethod
    def _summarize_turns(turns: list[DiarizedTurn]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "speaker_count": 0,
            "turn_count": len(turns),
            "total_duration_sec": 0.0,
            "per_speaker": {},
        }
        if not turns:
            return summary
        per_speaker: dict[str, dict[str, float | int]] = {}
        total_duration = 0.0
        for turn in turns:
            speaker_id = turn.speaker or "Speaker_1"
            duration = max(float(turn.end) - float(turn.start), 0.0)
            entry = per_speaker.setdefault(
                speaker_id,
                {
                    "duration_sec": 0.0,
                    "turns": 0,
                },
            )
            entry["duration_sec"] = float(entry["duration_sec"]) + duration
            entry["turns"] = int(entry["turns"]) + 1
            total_duration += duration
        summary["speaker_count"] = len(per_speaker)
        summary["total_duration_sec"] = round(total_duration, 3)
        summary["per_speaker"] = {
            speaker: {
                "duration_sec": round(float(stats["duration_sec"]), 3),
                "turns": int(stats["turns"]),
            }
            for speaker, stats in per_speaker.items()
        }
        return summary

    def _build_affinity_matrix(self, X: np.ndarray) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity

        affinity = cosine_similarity(X)
        np.fill_diagonal(affinity, 1.0)
        percentile = float(getattr(self.config, "spectral_p_percentile", 0.0) or 0.0)
        if 0.0 < percentile < 1.0:
            thresholds = np.quantile(affinity, percentile, axis=1)
            affinity = np.where(affinity >= thresholds[:, None], affinity, 0.0)
        affinity = np.maximum(affinity, affinity.T)
        affinity = np.clip((affinity + 1.0) / 2.0, 0.0, 1.0)
        return affinity

    def _spectral_cluster_embeddings(
        self, X: np.ndarray
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        stats: dict[str, Any] = {
            "percentile": float(getattr(self.config, "spectral_p_percentile", 0.0) or 0.0)
        }
        try:
            from sklearn.cluster import SpectralClustering
            from sklearn.metrics import silhouette_score
        except Exception as exc:  # pragma: no cover - defensive import guard
            stats["error"] = f"spectral_unavailable: {exc}"
            return None, stats
        n_embeddings = int(X.shape[0])
        if n_embeddings < 2:
            stats["error"] = "insufficient_embeddings"
            return None, stats
        affinity = self._build_affinity_matrix(X)
        if self.config.speaker_limit:
            candidates = [int(self.config.speaker_limit)]
        else:
            min_c = self.config.min_speakers or getattr(self.config, "spectral_min_speakers", 2)
            max_c = self.config.max_speakers or getattr(
                self.config, "spectral_max_speakers", min(n_embeddings, 6)
            )
            min_c = max(2, int(min_c or 2))
            max_c = min(int(max_c or n_embeddings), n_embeddings)
            candidates = list(range(min_c, max_c + 1))
        candidates = [c for c in candidates if c > 1 and c <= n_embeddings]
        stats["candidates"] = candidates
        best_labels: np.ndarray | None = None
        best_score = -1.0
        best_k = None
        errors: list[str] = []
        for k in candidates:
            try:
                clusterer = SpectralClustering(
                    n_clusters=int(k),
                    affinity="precomputed",
                    assign_labels="kmeans",
                    random_state=0,
                    n_init=10,
                )
                labels = clusterer.fit_predict(affinity)
            except Exception as exc:  # pragma: no cover - defensive guard
                errors.append(f"k={k}: {exc}")
                continue
            unique_labels = len(set(labels))
            score = -1.0
            if unique_labels > 1:
                try:
                    score = float(silhouette_score(X, labels, metric="cosine"))
                except Exception as exc:  # pragma: no cover - diagnostic guardrail
                    errors.append(f"silhouette k={k}: {exc}")
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k
        stats.update({"best_k": best_k, "silhouette": best_score, "errors": errors})
        floor = float(getattr(self.config, "spectral_silhouette_floor", 0.0) or 0.0)
        if best_labels is None:
            return None, stats
        if floor > 0.0 and best_score < floor:
            stats["rejected_below_floor"] = True
            return None, stats
        return best_labels, stats

    def _run_agglomerative_clustering(
        self,
        X: np.ndarray,
        *,
        n_clusters: int | None,
        distance_threshold: float | None,
        progress_interval: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        n_embeddings = int(X.shape[0])
        clusterer = build_agglo(
            distance_threshold=distance_threshold,
            n_clusters=n_clusters,
            linkage=self.config.ahc_linkage,
            metric="cosine",
        )

        def _agglo_fit() -> np.ndarray:
            return clusterer.fit_predict(X)

        labels, aggl_elapsed = _run_with_progress(
            "[diarize] agglomerative clustering",
            _agglo_fit,
            total=n_embeddings,
            interval=progress_interval,
        )
        cluster_count = int(len(set(labels)))
        merge_count = None
        if hasattr(clusterer, "children_"):
            try:
                merge_count = int(getattr(clusterer, "children_").shape[0])
            except Exception:
                merge_count = None
        distance_desc: str
        if n_clusters:
            distance_desc = "n/a"
        elif distance_threshold is None:
            distance_desc = "None"
        else:
            distance_desc = f"{float(distance_threshold):.3f}"
        stats = {
            "cluster_count": cluster_count,
            "elapsed_sec": aggl_elapsed,
            "n_embeddings": n_embeddings,
            "distance_threshold": distance_desc,
            "speaker_limit": self.config.speaker_limit,
            "n_clusters_requested": n_clusters,
        }
        if merge_count is not None:
            stats["merges"] = merge_count
        return labels, stats

    def diarize_audio(
        self,
        wav: np.ndarray,
        sr: int,
        speech_regions: list[tuple[float, float]] | None = None,
        *,
        region_source: str | None = None,
    ) -> list[dict[str, Any]]:
        self._last_turns = []
        self._last_vad_stats = {
            "vad_boundary_flips": 0.0,
            "speech_regions": 0.0,
            "analyzed_duration_sec": 0.0,
        }
        self._debug_payload = {
            "config": {
                "speaker_limit": self.config.speaker_limit,
                "ahc_distance_threshold": self.config.ahc_distance_threshold,
                "ahc_linkage": self.config.ahc_linkage,
                "vad_threshold": self.config.vad_threshold,
                "vad_min_speech_sec": self.config.vad_min_speech_sec,
                "vad_min_silence_sec": self.config.vad_min_silence_sec,
                "single_speaker_collapse": self.config.single_speaker_collapse,
            }
        }
        if wav is None or wav.size == 0:
            return []
        if wav.ndim > 1:
            wav = np.mean(wav, axis=0)
        if sr != self.config.target_sr:
            wav = scipy.signal.resample_poly(wav, self.config.target_sr, sr).astype(np.float32)
            sr = self.config.target_sr
        else:
            wav = wav.astype(np.float32)
        duration_sec = float(len(wav)) / float(sr or 1)
        self._last_vad_stats["analyzed_duration_sec"] = duration_sec
        speech_region_source = region_source or (
            "external" if speech_regions is not None else "silero_vad"
        )
        self._debug_payload["input"] = {
            "duration_sec": duration_sec,
            "sample_rate": sr,
            "speech_region_source": speech_region_source,
        }
        try:
            logger.info(
                "[diarize] processing %.1f minutes of audio (sr=%d)", duration_sec / 60.0, sr
            )
        except Exception:
            pass
        # If caller provided explicit speech_regions (e.g. from SED timeline), use that
        if speech_regions is None:
            speech_regions = self.vad.detect(
                wav, sr, self.config.vad_min_speech_sec, self.config.vad_min_silence_sec
            )
        else:
            # Provide VAD stats so diagnostics still meaningful
            self._last_vad_stats["vad_boundary_flips"] = float(len(speech_regions) or 0)
            self._last_vad_stats["speech_regions"] = float(len(speech_regions) or 0)
        if not speech_regions and self.config.allow_energy_vad_fallback:
            logger.info("Using energy VAD fallback")
            speech_regions = energy_vad_fallback(
                wav, sr, self.config.energy_gate_db, self.config.energy_hop_sec
            )
            speech_region_source = "energy_vad"
        if not speech_regions:
            logger.warning("No speech detected by VAD")
            self._last_vad_stats["vad_boundary_flips"] = 0.0
            self._last_vad_stats["speech_regions"] = 0.0
            self._debug_payload["vad"] = {
                "region_source": speech_region_source,
                "region_count": 0,
                "speech_total_sec": 0.0,
                "coverage_pct": 0.0,
            }
            return []

        # Split overlong speech regions into smaller chunks to avoid single gigantic turns.
        speech_regions = self._split_long_regions(
            speech_regions,
            max_len=self.config.max_vad_region_sec,
            overlap=self.config.vad_region_overlap_sec,
        )

        region_count = float(len(speech_regions))
        self._last_vad_stats["speech_regions"] = region_count
        self._last_vad_stats["vad_boundary_flips"] = region_count
        speech_total = sum(max(0.0, end - start) for start, end in speech_regions)
        try:
            coverage = (speech_total / duration_sec * 100.0) if duration_sec else 0.0
            logger.info(
                "[diarize] VAD detected %d regions totalling %.1f minutes (%.1f%% of audio)",
                len(speech_regions),
                speech_total / 60.0,
                coverage,
            )
        except Exception:
            coverage = 0.0
        self._debug_payload["vad"] = {
            "region_source": speech_region_source,
            "region_count": len(speech_regions),
            "speech_total_sec": round(speech_total, 3),
            "coverage_pct": round(coverage, 2) if duration_sec else 0.0,
        }
        windows = self._extract_embedding_windows(wav, sr, speech_regions)
        window_lengths = [max(0.0, w["end"] - w["start"]) for w in windows]
        self._debug_payload["embeddings"] = {
            "window_count": len(windows),
            "speech_region_count": len(speech_regions),
            "min_window_sec": round(min(window_lengths), 3) if window_lengths else 0.0,
            "max_window_sec": round(max(window_lengths), 3) if window_lengths else 0.0,
            "avg_window_sec": round(
                sum(window_lengths) / len(window_lengths), 3
            )
            if window_lengths
            else 0.0,
        }
        if len(windows) < 2:
            turn = {
                "start": speech_regions[0][0],
                "end": speech_regions[-1][1],
                "speaker": "Speaker_1",
                "speaker_name": "Speaker_1",
                "embedding": windows[0]["embedding"] if windows else None,
            }
            duration = max(turn["end"] - turn["start"], 0.0)
            self._debug_payload["turns"] = {
                "speaker_count": 1,
                "turn_count": 1,
                "total_duration_sec": round(duration, 3),
                "per_speaker": {
                    "Speaker_1": {
                        "duration_sec": round(duration, 3),
                        "turns": 1,
                    }
                },
            }
            return [turn]
        embeddings = [w["embedding"] for w in windows if w["embedding"] is not None]
        if not embeddings:
            logger.warning("No valid embeddings extracted")
            return []
        try:
            norms = [float(np.linalg.norm(vec)) for vec in embeddings if vec is not None]
            if norms:
                self._debug_payload.setdefault("embeddings", {}).update(
                    {
                        "norm_min": round(min(norms), 6),
                        "norm_max": round(max(norms), 6),
                        "norm_mean": round(sum(norms) / len(norms), 6),
                    }
                )
        except Exception:
            pass
        try:
            X = np.vstack(embeddings)
            n_embeddings = int(X.shape[0])
            progress_interval = float(
                getattr(self.config, "clustering_progress_interval_sec", 0.0) or 0.0
            )
            backend = (self.config.clustering_backend or "auto").strip().lower()
            labels = None
            clustering_debug: dict[str, Any] = {
                "requested_backend": backend,
                "embeddings": n_embeddings,
            }
            spectral_info: dict[str, Any] = {}
            label_hist: dict[str, int] = {}
            dominant_ratio = 1.0
            if backend in {"auto", "spectral"}:
                spectral_labels, spectral_info = self._spectral_cluster_embeddings(X)
                clustering_debug["spectral"] = spectral_info
                if spectral_labels is not None:
                    used_k = int(spectral_info.get("best_k") or len(set(spectral_labels)) or 0)
                    try:
                        logger.info(
                            "Spectral clustering selected %d clusters (silhouette=%.3f)",
                            used_k,
                            float(spectral_info.get("silhouette", -1.0) or -1.0),
                        )
                    except Exception:
                        pass
                    if backend == "spectral" or not bool(self.config.spectral_refine_with_ahc):
                        labels = spectral_labels
                        clustering_debug["final_backend"] = "spectral"
                    else:
                        labels, aggl_info = self._run_agglomerative_clustering(
                            X,
                            n_clusters=used_k if used_k > 0 else None,
                            distance_threshold=None,
                            progress_interval=progress_interval,
                        )
                        clustering_debug["agglomerative"] = aggl_info
                        clustering_debug["final_backend"] = "auto_refine_ahc"
                        clustering_debug["refined_from_spectral"] = used_k or None
                elif backend == "spectral":
                    clustering_debug["fallback_reason"] = "spectral_failed"
            if labels is None:
                n_clusters = int(self.config.speaker_limit) if self.config.speaker_limit else None
                distance_threshold = None if n_clusters else self.config.ahc_distance_threshold
                labels, aggl_info = self._run_agglomerative_clustering(
                    X,
                    n_clusters=n_clusters,
                    distance_threshold=distance_threshold,
                    progress_interval=progress_interval,
                )
                clustering_debug["agglomerative"] = aggl_info
                clustering_debug.setdefault("final_backend", "ahc")
            if labels is not None and len(labels) > 0:
                unique, counts = np.unique(labels, return_counts=True)
                for label_id, count in zip(unique, counts):
                    label_hist[f"Speaker_{int(label_id) + 1}"] = int(count)
            total_labels = float(sum(label_hist.values()))
            if total_labels > 0:
                dominant_ratio = max(label_hist.values()) / total_labels
            clustering_debug["clusters_detected"] = len(label_hist)
            clustering_debug["label_histogram"] = label_hist
            clustering_debug["dominant_ratio"] = round(dominant_ratio, 4)
            clustering_debug["spectral"] = spectral_info or clustering_debug.get("spectral")
            self._debug_payload["clustering"] = clustering_debug
        except Exception as exc:
            logger.error("Clustering failed: %s", exc)
            labels = np.zeros(len(embeddings), dtype=int)
        for window, label in zip(windows, labels, strict=False):
            window["speaker"] = f"Speaker_{label + 1}"
        turns = self._build_continuous_segments(windows, speech_regions)
        self._debug_payload["turns_initial"] = self._summarize_turns(turns)
        turns = self._merge_short_gaps(turns)
        turns = self._enforce_min_turn_duration(turns)
        turns = self._merge_similar_speakers(turns)
        self._debug_payload["turns_post_similarity_merge"] = self._summarize_turns(turns)
        should_collapse_single = bool(self.config.single_speaker_collapse)
        collapse_guard_payload: dict[str, Any] = {}
        # Heuristic: for very long files where VAD reports a single
        # continuous region covering almost the entire recording, avoid
        # collapsing down to one speaker even if a dominant cluster is
        # present. In practice this preserves useful segmentation for
        # court-style recordings instead of merging everything.
        vad_info = self._debug_payload.get("vad") or {}
        try:
            coverage_pct = float(vad_info.get("coverage_pct", 0.0) or 0.0)
            region_count_int = int(vad_info.get("region_count", len(speech_regions)))
        except Exception:
            coverage_pct = 0.0
            region_count_int = len(speech_regions)
        if (
            should_collapse_single
            and duration_sec >= 3600.0
            and region_count_int <= 1
            and coverage_pct >= 98.0
        ):
            should_collapse_single = False
            collapse_guard_payload.setdefault("reasons", []).append(
                {
                    "reason": "disabled_for_long_full_coverage_file",
                    "duration_sec": duration_sec,
                    "region_count": region_count_int,
                    "coverage_pct": coverage_pct,
                }
            )
        spectral_k = None
        spectral_silhouette = None
        try:
            spectral_k = int(spectral_info.get("best_k")) if spectral_info.get("best_k") else None
        except Exception:
            spectral_k = None
        try:
            raw_sil = spectral_info.get("silhouette")
            spectral_silhouette = float(raw_sil) if raw_sil is not None else None
        except Exception:
            spectral_silhouette = None
        min_guard_silhouette = max(
            0.05,
            float(getattr(self.config, "spectral_silhouette_floor", 0.0) or 0.0),
        )
        if should_collapse_single and len(label_hist) >= 2:
            if spectral_k and spectral_k >= 2 and (
                spectral_silhouette is not None and spectral_silhouette >= min_guard_silhouette
            ):
                should_collapse_single = False
                collapse_guard_payload.setdefault("reasons", []).append(
                    {
                        "reason": "guarded_by_spectral_clusters",
                        "spectral_k": spectral_k,
                        "spectral_silhouette": spectral_silhouette,
                        "min_guard_silhouette": min_guard_silhouette,
                    }
                )
            elif dominant_ratio <= max(0.92, float(self.config.single_speaker_dominance) + 0.02):
                should_collapse_single = False
                collapse_guard_payload.setdefault("reasons", []).append(
                    {
                        "reason": "guarded_by_cluster_balance",
                        "dominant_ratio": round(dominant_ratio, 4),
                        "clusters_detected": len(label_hist),
                    }
                )
        if collapse_guard_payload:
            self._debug_payload["single_speaker_collapse_guard"] = collapse_guard_payload
        if should_collapse_single:
            forced, canonical, stats = self._maybe_force_single_speaker(turns)
            if forced and canonical:
                details = []
                if stats:
                    if "silhouette" in stats:
                        details.append(f"silhouette={stats['silhouette']:.3f}")
                    if "dominance" in stats:
                        details.append(f"dominance={stats['dominance']:.2f}")
                    if "clusters" in stats:
                        details.append(f"clusters={int(stats['clusters'])}")
                suffix = f" ({', '.join(details)})" if details else ""
                logger.info(
                    "Forcing diarization clusters into single speaker '%s'%s",
                    canonical,
                    suffix,
                )
                for turn in turns:
                    turn.speaker = canonical
                    turn.speaker_name = canonical
                turns = self._merge_short_gaps(turns)
                payload = {"forced": True, "canonical": canonical}
                if stats:
                    payload.update(
                        {
                            "silhouette": stats.get("silhouette"),
                            "dominance": stats.get("dominance"),
                            "clusters": stats.get("clusters"),
                        }
                    )
                self._debug_payload["single_speaker_force"] = payload
        if should_collapse_single:
            self._debug_payload["turns_pre_collapse"] = self._summarize_turns(turns)
            collapsed, canonical, reason = collapse_single_speaker_turns(
                turns,
                dominance_threshold=self.config.single_speaker_dominance,
                centroid_threshold=self.config.single_speaker_centroid_threshold,
                min_turns=self.config.single_speaker_min_turns,
                secondary_max_ratio=self.config.single_speaker_secondary_max_ratio,
                secondary_min_duration_sec=self.config.single_speaker_secondary_min_duration_sec,
            )
            if collapsed:
                msg_reason = f" ({reason})" if reason else ""
                logger.info(
                    "Collapsing diarization clusters into single speaker '%s'%s",
                    canonical,
                    msg_reason,
                )
                turns = self._merge_short_gaps(turns)
                self._debug_payload["single_speaker_collapse"] = {
                    "collapsed": True,
                    "canonical": canonical,
                    "reason": reason,
                }
        turns = self._merge_short_gaps(turns)
        turns = self._assign_speaker_names(turns)
        self._last_turns = turns
        self._debug_payload["turns"] = self._summarize_turns(turns)
        return [self._turn_to_dict(t) for t in turns]

    def _extract_embedding_windows(
        self, wav: np.ndarray, sr: int, speech_regions: list[tuple[float, float]]
    ) -> list[dict[str, Any]]:
        clips: list[np.ndarray] = []
        meta: list[tuple[float, float]] = []
        for start_sec, end_sec in speech_regions:
            if end_sec - start_sec < self.config.min_embedtable_sec:
                continue
            cursor = start_sec
            while cursor < end_sec:
                win_end = min(cursor + self.config.embed_window_sec, end_sec)
                if win_end - cursor >= self.config.min_embedtable_sec:
                    start_idx = int(cursor * sr)
                    end_idx = int(win_end * sr)
                    clips.append(wav[start_idx:end_idx])
                    meta.append((cursor, win_end))
                cursor += self.config.embed_shift_sec
        if not clips:
            return []
        logger.info(
            "[diarize] preparing %d embedding windows across %d speech regions",
            len(clips),
            len(speech_regions),
        )
        try:
            max_batch = int(os.getenv("DIAREMOT_ECAPA_MAX_BATCH", "512"))
            if max_batch <= 0:
                max_batch = 512
        except Exception:
            max_batch = 512
        embeddings: list[np.ndarray | None] = []
        total_batches = max(1, math.ceil(len(clips) / max_batch))
        if len(clips) <= max_batch:
            embeddings = self.ecapa.embed_batch(clips, sr) or []
        else:
            for batch_idx, i in enumerate(range(0, len(clips), max_batch), start=1):
                batch = clips[i : i + max_batch]
                if total_batches > 1:
                    try:
                        logger.info(
                            "[diarize] ECAPA batch %d/%d (%d windows)",
                            batch_idx,
                            total_batches,
                            len(batch),
                        )
                    except Exception:
                        pass
                part = self.ecapa.embed_batch(batch, sr) or []
                embeddings.extend(part)
        windows: list[dict[str, Any]] = []
        for idx, (meta_item, emb) in enumerate(zip(meta, embeddings)):
            start_t, end_t = meta_item
            windows.append(
                {
                    "start": start_t,
                    "end": end_t,
                    "embedding": emb,
                    "speaker": None,
                    "region_idx": idx,
                }
            )
        try:
            logger.info(
                "ECAPA embeddings: %d windows batched (max_batch=%d)", len(windows), max_batch
            )
        except Exception:
            pass
        return windows

    def _build_continuous_segments(
        self,
        windows: list[dict[str, Any]],
        speech_regions: list[tuple[float, float]],
    ) -> list[DiarizedTurn]:
        if not windows:
            return []
        segments: list[DiarizedTurn] = []
        for region_start, region_end in speech_regions:
            region_windows = []
            for window in windows:
                if (
                    window.get("speaker") is not None
                    and window["start"] < region_end
                    and window["end"] > region_start
                    and window["embedding"] is not None
                ):
                    region_windows.append(
                        {
                            "start": max(window["start"], region_start),
                            "end": min(window["end"], region_end),
                            "speaker": window["speaker"],
                            "embedding": window["embedding"],
                            "duration": min(window["end"], region_end)
                            - max(window["start"], region_start),
                        }
                    )
            if not region_windows:
                continue
            region_windows.sort(key=lambda x: x["start"])
            events = []
            for entry in region_windows:
                events.append(
                    {
                        "time": entry["start"],
                        "type": "start",
                        "speaker": entry["speaker"],
                        "embedding": entry["embedding"],
                    }
                )
                events.append({"time": entry["end"], "type": "end", "speaker": entry["speaker"]})
            events.sort(key=lambda x: (x["time"], 0 if x["type"] == "end" else 1))
            active_speakers: dict[str, float] = {}
            current_time = region_start
            for event in events:
                event_time = event["time"]
                if event_time > current_time:
                    if active_speakers:
                        best_speaker = max(active_speakers, key=active_speakers.get)
                        emb_list = [
                            entry["embedding"]
                            for entry in region_windows
                            if entry["speaker"] == best_speaker
                            and entry["start"] < event_time
                            and entry["end"] > current_time
                            and entry["embedding"] is not None
                        ]
                        speaker_embedding = None
                        if emb_list:
                            pooled = np.mean(np.vstack(emb_list), axis=0)
                            norm = np.linalg.norm(pooled)
                            speaker_embedding = pooled / (norm + 1e-8) if norm > 0 else pooled
                        segments.append(
                            DiarizedTurn(
                                start=current_time,
                                end=event_time,
                                speaker=best_speaker,
                                speaker_name=best_speaker,
                                embedding=speaker_embedding,
                            )
                        )
                if event["type"] == "start":
                    active_speakers[event["speaker"]] = (
                        active_speakers.get(event["speaker"], 0.0) + 1.0
                    )
                else:
                    active_speakers.pop(event["speaker"], None)
                current_time = event_time
            if active_speakers:
                dominant_speaker = max(active_speakers, key=active_speakers.get)
                emb_list = [
                    entry["embedding"]
                    for entry in region_windows
                    if entry["speaker"] == dominant_speaker
                    and entry["start"] < region_end
                    and entry["end"] > current_time
                    and entry["embedding"] is not None
                ]
                speaker_embedding = None
                if emb_list:
                    pooled = np.mean(np.vstack(emb_list), axis=0)
                    norm = np.linalg.norm(pooled)
                    speaker_embedding = pooled / (norm + 1e-8) if norm > 0 else pooled
                segments.append(
                    DiarizedTurn(
                        start=current_time,
                        end=region_end,
                        speaker=dominant_speaker,
                        speaker_name=dominant_speaker,
                        embedding=speaker_embedding,
                    )
                )
        if not segments:
            return []
        merged = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            gap = seg.start - last.end
            if last.speaker == seg.speaker and gap <= self.config.max_gap_to_merge_sec:
                last_duration = last.end - last.start
                seg_duration = seg.end - seg.start
                if last.embedding is not None and seg.embedding is not None:
                    pooled = last.embedding * last_duration + seg.embedding * seg_duration
                    norm = np.linalg.norm(pooled)
                    last.embedding = pooled / (norm + 1e-8) if norm > 0 else pooled
                last.end = seg.end
            else:
                merged.append(seg)
        return merged

    def _maybe_force_single_speaker(
        self, turns: list[DiarizedTurn]
    ) -> tuple[bool, str | None, dict[str, float] | None]:
        if not turns:
            return False, None, None
        speakers = [t.speaker for t in turns if t.speaker]
        unique_speakers = {spk for spk in speakers if spk}
        if len(unique_speakers) <= 1:
            return False, None, None
        max_clusters = int(getattr(self.config, "single_speaker_force_max_clusters", 0) or 0)
        if max_clusters > 0 and len(unique_speakers) > max_clusters:
            return False, None, None
        durations: dict[str, float] = {}
        total_duration = 0.0
        for turn in turns:
            if not turn.speaker:
                continue
            duration = max(float(turn.end) - float(turn.start), 0.0)
            if duration <= 0:
                continue
            durations[turn.speaker] = durations.get(turn.speaker, 0.0) + duration
            total_duration += duration
        if total_duration <= 0.0 or not durations:
            return False, None, None
        dominant_speaker, dominant_duration = max(durations.items(), key=lambda item: item[1])
        dominance_ratio = dominant_duration / total_duration
        dominance_floor = float(getattr(self.config, "single_speaker_force_dominance", 0.0) or 0.0)
        if dominance_floor > 0.0 and dominance_ratio < dominance_floor:
            return False, None, None
        embeddings: list[np.ndarray] = []
        labels: list[str] = []
        for turn in turns:
            if turn.embedding is None or not turn.speaker:
                continue
            vec = np.asarray(turn.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm <= 0.0:
                continue
            embeddings.append(vec / norm)
            labels.append(turn.speaker)
        if len(embeddings) < 2 or len(set(labels)) <= 1:
            return False, None, None
        try:
            from sklearn.metrics import silhouette_score

            arr = np.stack(embeddings, axis=0)
            label_map = {spk: idx for idx, spk in enumerate(sorted(set(labels)))}
            label_indices = np.asarray([label_map[label] for label in labels], dtype=np.int32)
            score = float(silhouette_score(arr, label_indices, metric="cosine"))
        except Exception as exc:  # pragma: no cover - diagnostic guardrail
            try:
                logger.debug(
                    "Silhouette score unavailable for single-speaker heuristic: %s",
                    exc,
                )
            except Exception:
                pass
            return False, None, None
        threshold = float(getattr(self.config, "single_speaker_silhouette_threshold", 0.0) or 0.0)
        if threshold <= 0.0:
            return False, None, None
        if score <= threshold:
            stats = {
                "silhouette": score,
                "dominance": dominance_ratio,
                "clusters": float(len(unique_speakers)),
            }
            return True, dominant_speaker, stats
        return False, None, None

    def _merge_similar_speakers(self, turns: list[DiarizedTurn]) -> list[DiarizedTurn]:
        if not turns:
            return turns
        by_spk: dict[str, list[np.ndarray]] = {}
        for turn in turns:
            if turn.embedding is None:
                continue
            by_spk.setdefault(turn.speaker, []).append(np.asarray(turn.embedding, dtype=np.float32))
        if len(by_spk) <= 1:
            return turns

        def centroid(vecs: list[np.ndarray]) -> np.ndarray:
            arr = np.vstack(vecs)
            c = arr.mean(axis=0)
            return c / (np.linalg.norm(c) + 1e-9)

        centroids: dict[str, np.ndarray] = {k: centroid(v) for k, v in by_spk.items() if v}
        if len(centroids) <= 1:
            return turns

        def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
            return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        min_speakers = int(self.config.post_merge_min_speakers or 0)
        base_thresh = float(self.config.post_merge_distance_threshold or 0.0)
        if base_thresh <= 0.0:
            return turns
        dynamic_thresh = base_thresh
        if len(centroids) >= max(6, min_speakers + 5):
            # Use a conservative boost to encourage merging when we have many speakers,
            # but avoid using single_speaker_centroid_threshold which is for a different purpose.
            dynamic_thresh = max(dynamic_thresh, 0.35)
        thresh = dynamic_thresh
        changed = True
        while changed and len(centroids) > max(1, min_speakers):
            changed = False
            keys = list(centroids.keys())
            best_pair = None
            best_dist = 1e9
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    dist = cosine_distance(centroids[keys[i]], centroids[keys[j]])
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (keys[i], keys[j])
            if best_pair is None or best_dist >= thresh:
                break
            a, b = best_pair
            for turn in turns:
                if turn.speaker == b:
                    turn.speaker = a
                    turn.speaker_name = a
            if b in by_spk:
                by_spk.setdefault(a, []).extend(by_spk[b])
                by_spk.pop(b, None)
            if a in by_spk:
                centroids[a] = centroid(by_spk[a])
            centroids.pop(b, None)
            changed = True
            if len(centroids) <= max(1, min_speakers):
                break
        return turns

    def _merge_short_gaps(self, turns: list[DiarizedTurn]) -> list[DiarizedTurn]:
        if not turns:
            return []
        merged = [turns[0]]
        for turn in turns[1:]:
            last = merged[-1]
            gap = turn.start - last.end
            if last.speaker == turn.speaker and 0 <= gap <= self.config.max_gap_to_merge_sec:
                last.end = turn.end
            else:
                merged.append(turn)
        return merged

    def _enforce_min_turn_duration(self, turns: list[DiarizedTurn]) -> list[DiarizedTurn]:
        if not turns:
            return []
        out: list[DiarizedTurn] = []
        min_len = float(getattr(self.config, "min_turn_sec", 0.0) or 0.0)
        for idx, cur in enumerate(turns):
            duration = float(cur.end - cur.start)
            if duration >= min_len or min_len <= 0.0:
                out.append(cur)
                continue
            merged = False
            if out:
                prev = out[-1]
                gap = max(0.0, cur.start - prev.end)
                if prev.speaker == cur.speaker and gap <= self.config.max_gap_to_merge_sec:
                    prev.end = max(prev.end, cur.end)
                    merged = True
            if not merged and idx + 1 < len(turns):
                nxt = turns[idx + 1]
                gap = max(0.0, nxt.start - cur.end)
                if nxt.speaker == cur.speaker and gap <= self.config.max_gap_to_merge_sec:
                    nxt.start = min(nxt.start, cur.start)
                    merged = True
            if not merged:
                out.append(cur)
        return out

    def _assign_speaker_names(self, turns: list[DiarizedTurn]) -> list[DiarizedTurn]:
        if not self.registry:
            return turns
        for turn in turns:
            if turn.embedding is not None:
                name, similarity = self.registry.match(turn.embedding)
                if name and similarity >= self.config.auto_assign_cosine:
                    turn.speaker_name = name
                    turn.candidate_name = name
                    if self.config.flag_band_low <= similarity <= self.config.flag_band_high:
                        turn.needs_review = True
        return turns

    def reassign_with_registry(self, turns: list[dict[str, Any]]) -> None:
        if not self.registry:
            return
        for turn in turns:
            embedding = turn.get("embedding")
            if embedding is not None:
                name, similarity = self.registry.match(np.asarray(embedding))
                if name and similarity >= self.config.auto_assign_cosine:
                    turn["speaker_name"] = name
                    turn["candidate_name"] = name

    def _split_long_regions(
        self,
        regions: list[tuple[float, float]],
        *,
        max_len: float,
        overlap: float,
    ) -> list[tuple[float, float]]:
        """Split overlong VAD regions into smaller chunks with overlap."""
        if max_len <= 0.0:
            return regions
        out: list[tuple[float, float]] = []
        for start, end in regions:
            length = max(0.0, end - start)
            if length <= max_len:
                out.append((start, end))
                continue
            cursor = start
            while cursor < end:
                win_end = min(cursor + max_len, end)
                out.append((cursor, win_end))
                if win_end >= end:
                    break
                cursor = max(cursor + max_len - overlap, cursor + 0.1)  # ensure progress
        return out

    def _turn_to_dict(self, turn: DiarizedTurn) -> dict[str, Any]:
        return {
            "start": turn.start,
            "end": turn.end,
            "speaker": turn.speaker,
            "speaker_name": turn.speaker_name,
            "candidate_name": turn.candidate_name,
            "needs_review": turn.needs_review,
            "embedding": (turn.embedding.tolist() if turn.embedding is not None else None),
        }


__all__ = ["SpeakerDiarizer"]
