"""Runtime environment scaffolding for the DiaRemot pipeline.

This module mirrors the structure used by the paralinguistics package: a
compact, dataclass-centric definition of the runtime environment with explicit
exports and no implicit global state beyond an optional cached singleton.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT_MARKERS: Final[tuple[str, ...]] = ("pyproject.toml", ".git")
LINUX_CANONICAL_MODEL_ROOT: Final[Path] = Path("/srv/models")
WINDOWS_CANONICAL_MODEL_ROOT: Final[Path] = Path("D:/models")
LINUX_CACHE_BASE: Final[Path] = Path("/srv/.cache")
WINDOWS_CACHE_BASE: Final[Path] = Path("D:/srv/.cache")
WHISPER_FALLBACK_DIRS: Final[tuple[tuple[str, ...], ...]] = (
    ("distil-large-v3",),
    ("faster-whisper", "distil-large-v3"),
    ("tiny.en",),
    ("faster-whisper", "tiny.en"),
    ("ct2", "tiny.en"),
)


def _find_project_root(start: Path) -> Path | None:
    current = start
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        for marker in PROJECT_ROOT_MARKERS:
            if (candidate / marker).exists():
                return candidate
    return None


def _ensure_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    probe = path / ".cache_write_test"
    try:
        probe.touch(exist_ok=True)
    except OSError:
        return False
    else:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            return False
    return os.access(path, os.W_OK | os.X_OK)


def _iter_cache_candidates(script_path: Path, preferred: Path | None) -> Iterator[Path]:
    if preferred is not None:
        yield Path(preferred)

    base = WINDOWS_CACHE_BASE if os.name == "nt" else LINUX_CACHE_BASE
    yield base

    project_root = _find_project_root(script_path)
    if project_root is not None:
        yield project_root / ".cache"

    yield Path.cwd() / ".cache"
    yield Path.home() / ".cache" / "diaremot"


def _default_model_root() -> Path:
    preferred = WINDOWS_CANONICAL_MODEL_ROOT if os.name == "nt" else LINUX_CANONICAL_MODEL_ROOT
    if _ensure_writable_directory(preferred):
        return preferred

    fallback = Path.cwd() / ".cache" / "models"
    _ensure_writable_directory(fallback)
    return fallback


def _coerce_root(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _dedupe_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for candidate in paths:
        resolved = candidate.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(resolved)
    return tuple(ordered)


def _discover_model_roots(primary: Path, script_path: Path) -> tuple[Path, ...]:
    roots: list[Path] = [primary]
    roots.append(WINDOWS_CANONICAL_MODEL_ROOT if os.name == "nt" else LINUX_CANONICAL_MODEL_ROOT)

    project_root = _find_project_root(script_path)
    if project_root is not None:
        roots.append(project_root / "models")

    roots.extend(
        [
            Path.cwd() / "models",
            Path.home() / "models",
            Path.cwd() / ".cache" / "models",
        ]
    )
    return _dedupe_paths(roots)


def _iter_whisper_candidates(model_roots: Iterable[Path]) -> Iterator[Path]:
    for root in model_roots:
        for parts in WHISPER_FALLBACK_DIRS:
            yield Path(root).joinpath(*parts)
    yield Path.home() / "whisper_models" / "tiny.en"


@dataclass(slots=True, frozen=True)
class PipelineEnvironment:
    """Resolved cache/model layout used by the pipeline runtime."""

    cache_root: Path
    model_roots: tuple[Path, ...]
    default_whisper_model: Path
    hf_home: Path
    huggingface_cache: Path
    transformers_cache: Path
    torch_home: Path
    xdg_cache: Path

    @property
    def primary_model_root(self) -> Path:
        return self.model_roots[0]

    def apply(self) -> None:
        """Set environment variables so runtime components share the cache layout."""

        targets = {
            "HF_HOME": self.hf_home,
            "HUGGINGFACE_HUB_CACHE": self.huggingface_cache,
            "TRANSFORMERS_CACHE": self.transformers_cache,
            "TORCH_HOME": self.torch_home,
            "XDG_CACHE_HOME": self.xdg_cache,
        }
        for env_name, target in targets.items():
            target.mkdir(parents=True, exist_ok=True)
            existing = os.environ.get(env_name)
            if existing:
                try:
                    existing_path = Path(existing).expanduser().resolve()
                except (OSError, RuntimeError, ValueError):
                    existing_path = None
                if existing_path is not None:
                    try:
                        if existing_path == target.resolve() or existing_path.is_relative_to(
                            self.cache_root
                        ):
                            continue
                    except AttributeError:
                        if str(existing_path).startswith(str(self.cache_root)):
                            continue
            os.environ[env_name] = str(target)

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("DIAREMOT_MODEL_DIR", str(self.primary_model_root))

    def with_primary_model_root(self, path: Path) -> PipelineEnvironment:
        """Return a new environment with ``path`` promoted to the primary root."""

        script_path = Path(__file__).resolve()
        resolved = Path(path).expanduser().resolve()
        model_roots = _discover_model_roots(resolved, script_path)
        whisper_model = _resolve_default_whisper_model(model_roots)
        cache_root = self.cache_root
        return PipelineEnvironment(
            cache_root=cache_root,
            model_roots=model_roots,
            default_whisper_model=whisper_model,
            hf_home=self.hf_home,
            huggingface_cache=self.huggingface_cache,
            transformers_cache=self.transformers_cache,
            torch_home=self.torch_home,
            xdg_cache=self.xdg_cache,
        )

    @classmethod
    def discover(
        cls,
        *,
        preferred_cache: Path | None = None,
        model_root: Path | None = None,
    ) -> PipelineEnvironment:
        script_path = Path(__file__).resolve()

        cache_root = None
        for candidate in _iter_cache_candidates(script_path, _coerce_root(preferred_cache)):
            resolved = Path(candidate).expanduser().resolve()
            if _ensure_writable_directory(resolved):
                cache_root = resolved
                break
        if cache_root is None:
            raise PermissionError("Unable to locate a writable cache directory for DiaRemot")

        env_map = {
            "hf_home": cache_root / "hf",
            "huggingface_cache": cache_root / "hf",
            "transformers_cache": cache_root / "transformers",
            "torch_home": cache_root / "torch",
            "xdg_cache": cache_root,
        }

        env_model_root = _coerce_root(model_root) or _coerce_root(
            os.environ.get("DIAREMOT_MODEL_DIR")
        )
        if env_model_root is None:
            env_model_root = _default_model_root()
        env_model_root = env_model_root.expanduser().resolve()

        model_roots = _discover_model_roots(env_model_root, script_path)
        whisper_model = _resolve_default_whisper_model(model_roots)

        return cls(
            cache_root=cache_root,
            model_roots=model_roots,
            default_whisper_model=whisper_model,
            **env_map,
        )


def _resolve_default_whisper_model(model_roots: Iterable[Path]) -> Path:
    override = _coerce_root(os.environ.get("WHISPER_MODEL_PATH"))
    if override and override.exists():
        return override
    for candidate in _iter_whisper_candidates(model_roots):
        if candidate.exists():
            return candidate
    return next(_iter_whisper_candidates(model_roots))


_BOOTSTRAPPED_ENV: PipelineEnvironment | None = None


def bootstrap_environment(
    *,
    preferred_cache: Path | None = None,
    model_root: Path | None = None,
    apply: bool = True,
) -> PipelineEnvironment:
    """Return a cached pipeline environment, optionally refreshing with overrides."""

    global _BOOTSTRAPPED_ENV

    refresh = _BOOTSTRAPPED_ENV is None or preferred_cache is not None or model_root is not None
    if refresh:
        _BOOTSTRAPPED_ENV = PipelineEnvironment.discover(
            preferred_cache=preferred_cache,
            model_root=model_root,
        )
    env = _BOOTSTRAPPED_ENV
    if env is None:
        raise RuntimeError("Pipeline environment bootstrap failed")
    if apply:
        env.apply()
    return env


__all__ = ["PipelineEnvironment", "bootstrap_environment"]
