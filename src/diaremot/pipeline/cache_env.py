"""Legacy import shim delegating to the unified runtime environment bootstrap."""

from __future__ import annotations

from .runtime import bootstrap_environment

__all__ = ["configure_local_cache_env"]


def configure_local_cache_env() -> None:
    """Ensure the shared cache environment is initialised."""

    bootstrap_environment(apply=True)


_configured = False


def _configure_once() -> None:
    global _configured
    if not _configured:
        configure_local_cache_env()
        _configured = True


_configure_once()
