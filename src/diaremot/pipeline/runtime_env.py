"""Compatibility shims for legacy imports targeting the pipeline runtime env."""

from __future__ import annotations

import os
from pathlib import Path

from .runtime import PipelineEnvironment, bootstrap_environment

__all__ = [
    "DEFAULT_MODELS_ROOT",
    "MODEL_ROOTS",
    "DEFAULT_WHISPER_MODEL",
    "WINDOWS_MODELS_ROOT",
    "configure_local_cache_env",
    "resolve_default_whisper_model",
    "iter_model_roots",
    "set_primary_model_root",
]


_ENV: PipelineEnvironment = bootstrap_environment()
MODEL_ROOTS: tuple[Path, ...] = _ENV.model_roots
DEFAULT_MODELS_ROOT: Path = MODEL_ROOTS[0]
DEFAULT_WHISPER_MODEL: Path = _ENV.default_whisper_model
WINDOWS_MODELS_ROOT: Path | None = MODEL_ROOTS[0] if os.name == "nt" else None


def _sync(env: PipelineEnvironment) -> None:
    global _ENV, MODEL_ROOTS, DEFAULT_MODELS_ROOT, DEFAULT_WHISPER_MODEL, WINDOWS_MODELS_ROOT
    _ENV = env
    MODEL_ROOTS = env.model_roots
    DEFAULT_MODELS_ROOT = env.model_roots[0]
    DEFAULT_WHISPER_MODEL = env.default_whisper_model
    WINDOWS_MODELS_ROOT = env.model_roots[0] if os.name == "nt" else None


def configure_local_cache_env() -> None:
    """Ensure cache-related environment variables are initialised."""

    _sync(bootstrap_environment(apply=True))


def iter_model_roots() -> tuple[Path, ...]:
    """Return the ordered tuple of candidate model roots."""

    return MODEL_ROOTS


def resolve_default_whisper_model() -> Path:
    """Return the resolved default Faster-Whisper model directory."""

    return DEFAULT_WHISPER_MODEL


def set_primary_model_root(path: Path | str) -> None:
    """Override the primary model root and refresh cached search paths."""

    env = bootstrap_environment(model_root=Path(path), apply=True)
    os.environ["DIAREMOT_MODEL_DIR"] = str(env.primary_model_root)
    _sync(env)
