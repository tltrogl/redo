from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ...utils.model_paths import collect_model_roots

_DISCOVERED_ROOTS = tuple(collect_model_roots())

if _DISCOVERED_ROOTS:
    MODEL_ROOTS = _DISCOVERED_ROOTS
else:
    MODEL_ROOTS = (Path.cwd() / "models",)


def iter_model_subpaths(*relative_paths: Path | str) -> Iterator[Path]:
    for root in MODEL_ROOTS:
        for rel in relative_paths:
            yield Path(root) / Path(rel)


__all__ = ["MODEL_ROOTS", "iter_model_subpaths"]
