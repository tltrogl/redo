"""Backward-compatible shim exposing the modular diarization package."""

from __future__ import annotations

from .diarization import (
    MODEL_ROOTS,
    ECAPAEncoder,
    SileroVAD,
    SpeakerDiarizer,
    SpeakerRegistry,
    SpectralClusterer,
    build_agglo,
    collapse_single_speaker_turns,
    iter_model_subpaths,
)
from .diarization.config import DiarizationConfig, DiarizedTurn


def _agglo(distance_threshold: float | None, **kwargs):
    return build_agglo(distance_threshold, **kwargs)


_SileroWrapper = SileroVAD
_ECAPAWrapper = ECAPAEncoder

__all__ = [
    "DiarizationConfig",
    "DiarizedTurn",
    "SpeakerDiarizer",
    "SpeakerRegistry",
    "collapse_single_speaker_turns",
    "iter_model_subpaths",
    "MODEL_ROOTS",
    "_agglo",
    "_SileroWrapper",
    "_ECAPAWrapper",
    "SpectralClusterer",
]
