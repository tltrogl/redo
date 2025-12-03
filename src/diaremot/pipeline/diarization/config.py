from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiarizationConfig:
    target_sr: int = 16000
    mono: bool = True
    vad_threshold: float = 0.35
    vad_min_speech_sec: float = 0.80
    vad_min_silence_sec: float = 0.80
    speech_pad_sec: float = 0.10
    vad_backend: str = "auto"
    embed_window_sec: float = 1.5
    embed_shift_sec: float = 0.75
    min_embedtable_sec: float = 0.6
    topk_windows: int = 3
    ahc_linkage: str = "average"
    ahc_distance_threshold: float = 0.45
    speaker_limit: int | None = None
    clustering_backend: str = "ahc"
    clustering_progress_interval_sec: float = 60.0
    min_speakers: int | None = None
    max_speakers: int | None = None
    collar_sec: float = 0.25
    min_turn_sec: float = 1.50
    max_gap_to_merge_sec: float = 1.00
    post_merge_distance_threshold: float = 0.45
    post_merge_min_speakers: int | None = None
    registry_path: str = "registry/speaker_registry.json"
    auto_assign_cosine: float = 0.70
    flag_band_low: float = 0.60
    flag_band_high: float = 0.70
    ecapa_model_path: str | None = None
    allow_energy_vad_fallback: bool = True
    energy_gate_db: float = -33.0
    energy_hop_sec: float = 0.01
    # Split long VAD regions to avoid collapsing an entire recording into one turn.
    max_vad_region_sec: float = 45.0
    vad_region_overlap_sec: float = 0.50
    single_speaker_collapse: bool = True
    single_speaker_dominance: float = 0.88
    single_speaker_centroid_threshold: float = 0.20
    single_speaker_min_turns: int = 3
    single_speaker_silhouette_threshold: float = 0.25
    single_speaker_force_dominance: float = 0.70
    single_speaker_force_max_clusters: int = 4
    single_speaker_secondary_max_ratio: float = 0.05


@dataclass
class DiarizedTurn:
    start: float
    end: float
    speaker: str
    speaker_name: str | None = None
    candidate_name: str | None = None
    needs_review: bool = False
    embedding: np.ndarray | None = None


__all__ = ["DiarizationConfig", "DiarizedTurn"]
