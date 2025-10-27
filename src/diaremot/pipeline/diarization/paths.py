from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..runtime_env import DEFAULT_MODELS_ROOT, iter_model_roots

MODEL_ROOTS = tuple(iter_model_roots()) or (DEFAULT_MODELS_ROOT,)


def iter_model_subpaths(*relative_paths: Path | str) -> Iterator[Path]:
    for root in MODEL_ROOTS:
        for rel in relative_paths:
            yield Path(root) / Path(rel)


__all__ = ["MODEL_ROOTS", "iter_model_subpaths"]
