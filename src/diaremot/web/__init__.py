"""DiaRemot Web Application Package."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():
    """Create and configure the FastAPI application."""
    from .api.app import app
    return app
