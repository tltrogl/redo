from __future__ import annotations

import asyncio
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from ..runtime_env import iter_model_roots

__all__ = [
    "configure_environment",
    "BackendAvailability",
    "backends",
    "ModelManager",
    "get_system_capabilities",
]


def configure_environment() -> None:
    """Apply CPU-only environment safeguards and suppress noisy warnings."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "TORCH_DEVICE": "cpu",
            "FORCE_CPU": "1",
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("CT2_FORCE_CPU_ISA", "GENERIC")


class BackendAvailability:
    """Runtime discovery of optional transcription dependencies."""

    _instance: BackendAvailability | None = None

    def __new__(cls) -> BackendAvailability:  # pragma: no cover - singleton boilerplate
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._check_backends()
        return cls._instance

    def _check_backends(self) -> None:
        import importlib.util as _util

        try:
            import librosa

            self.librosa = librosa
            self.has_librosa = True
        except Exception:  # pragma: no cover - optional dependency
            self.has_librosa = False
            self.librosa = None

        self.has_faster_whisper = False
        self.WhisperModel = None  # type: ignore[attr-defined]
        if _util.find_spec("faster_whisper") is not None:
            try:
                from faster_whisper import WhisperModel as _WhisperModel  # type: ignore

                self.has_faster_whisper = True
                self.WhisperModel = _WhisperModel
            except Exception:  # pragma: no cover - import side-effects
                self.has_faster_whisper = False
                self.WhisperModel = None

        self.has_openai_whisper = _util.find_spec("whisper") is not None
        self.openai_whisper = None


backends = BackendAvailability()


class ModelManager:
    """Lazy-loading transcription models with backend fallbacks."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._model_configs: dict[str, dict[str, Any]] = {}
        self._load_locks: dict[str, asyncio.Lock] = {}
        self.logger = logging.getLogger(__name__ + ".ModelManager")
        self.last_errors: dict[str, str] = {}

    @asynccontextmanager
    async def get_model(self, model_key: str, config: dict[str, Any]):
        lock = self._load_locks.get(model_key)
        if lock is None:
            lock = asyncio.Lock()
            self._load_locks[model_key] = lock

        async with lock:
            if self._should_reload_model(model_key, config):
                await self._load_model(model_key, config)

        try:
            yield self._models[model_key]
        finally:
            pass

    def _should_reload_model(self, model_key: str, config: dict[str, Any]) -> bool:
        existing_model = self._models.get(model_key)
        if existing_model is None:
            return True

        previous_config = self._model_configs.get(model_key)
        if previous_config is None:
            return True

        critical_keys = ("model_size", "compute_type", "asr_backend", "cpu_threads")
        return any(previous_config.get(key) != config.get(key) for key in critical_keys)

    async def _load_model(self, model_key: str, config: dict[str, Any]) -> None:
        loop = asyncio.get_event_loop()

        def _load() -> Any:
            pref = str(config.get("asr_backend", "auto")).lower()
            last_error: Exception | None = None

            def _try_faster() -> Any | None:
                nonlocal last_error
                if not backends.has_faster_whisper:
                    return None
                try:
                    return self._load_faster_whisper(config)
                except Exception as exc:  # pragma: no cover - runtime fallback guard
                    last_error = exc
                    self.logger.warning("Faster-Whisper load failed: %s", exc)
                    return None

            def _try_openai() -> Any | None:
                nonlocal last_error
                if not backends.has_openai_whisper:
                    return None
                try:
                    return self._load_openai_whisper(config)
                except Exception as exc:  # pragma: no cover - runtime fallback guard
                    last_error = exc
                    self.logger.warning("OpenAI Whisper load failed: %s", exc)
                    return None

            if pref == "openai":
                model = _try_openai() or _try_faster()
            elif pref == "faster":
                model = _try_faster() or _try_openai()
            else:
                model = _try_faster() or _try_openai()

            if model is None:
                if last_error is not None:
                    raise RuntimeError("No transcription backend available") from last_error
                raise RuntimeError("No transcription backend available")
            return model

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-loader") as executor:
            model = await loop.run_in_executor(executor, _load)
            self._models[model_key] = model
            self._model_configs[model_key] = config.copy()

    def _load_faster_whisper(self, config: dict[str, Any]) -> Any:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:
            self.last_errors["faster_whisper_import"] = str(exc)
            raise

        compute_type = str(config.get("compute_type", "float32")).lower()
        if compute_type not in ("float32", "int8", "int8_float16", "float16"):
            compute_type = "float32"

        model_size = config["model_size"]
        failure_notes: list[str] = []
        prefer_local = True if config.get("local_first", True) else False

        # Allow explicit override for distil Whisper model locations.
        # If the user passes the Systran distil identifier but stores the
        # converted CTranslate2 weights under a custom directory, they can
        # set DIAREMOT_DISTIL_WHISPER_DIR to that path and we will prefer it.
        if isinstance(model_size, str) and "faster-distil-whisper-large-v3" in model_size:
            override = os.getenv("DIAREMOT_DISTIL_WHISPER_DIR")
            if override:
                model_size = override

        is_local_path = False
        try:
            if isinstance(model_size, str) and Path(str(model_size)).exists():
                is_local_path = True
        except Exception:
            is_local_path = False

        def _try_load(identifier: str | Path, local_only: bool):
            kwargs = {
                "device": "cpu",
                "compute_type": compute_type,
                "cpu_threads": config.get("cpu_threads", 1),
                "download_root": None,
                "local_files_only": bool(local_only),
            }
            return WhisperModel(str(identifier), **kwargs)

        if is_local_path:
            try:
                model = _try_load(model_size, local_only=True)
                self.logger.info("Loaded faster-whisper (local path): %s", model_size)
                return model
            except Exception as exc:
                self.last_errors["faster_whisper_load"] = str(exc)
                self.logger.warning(
                    "Local faster-whisper path failed; continuing to cached/remote fallbacks: %s",
                    exc,
                )

        candidate_dirs: list[Path] = []
        try:
            rel_candidates: list[Path] = []
            if isinstance(model_size, str):
                rel_candidates.extend(
                    [
                        Path(model_size),
                        Path("faster-whisper") / model_size,
                        Path("ct2") / model_size,
                    ]
                )
                # Special-case Systran distil: also probe the conventional
                # "<model_root>/faster-whisper/distil" directory so users
                # can keep weights under DIAREMOT_MODEL_DIR without using
                # the full repo id as a folder name.
                if "faster-distil-whisper-large-v3" in model_size:
                    rel_candidates.append(Path("faster-whisper") / "distil")
            for root in iter_model_roots():
                for rel in rel_candidates:
                    candidate = Path(root) / rel
                    if candidate.exists():
                        candidate_dirs.append(candidate)
        except Exception:
            candidate_dirs = []

        for candidate in candidate_dirs:
            try:
                model = _try_load(candidate, local_only=True)
                self.logger.info("Loaded faster-whisper (model dir): %s", candidate)
                return model
            except Exception as exc:
                note = f"dir={candidate} local_only=True: {exc}"
                failure_notes.append(note)
                self.last_errors[f"faster_whisper_load:{candidate}"] = str(exc)
                self.logger.warning(
                    "Candidate local faster-whisper directory %s failed: %s",
                    candidate,
                    exc,
                )

        order = (True, False) if prefer_local else (False, True)
        for local_only in order:
            try:
                model = _try_load(model_size, local_only=local_only)
                source = "local" if local_only else "remote"
                self.logger.info("Loaded faster-whisper (%s): %s", source, model_size)
                return model
            except Exception as exc:
                self.last_errors[f"faster_whisper_load:{local_only}"] = str(exc)
                failure_notes.append(f"model={model_size} local_only={local_only}: {exc}")
                if local_only and prefer_local:
                    self.logger.info(
                        "Model not found in local cache; will attempt download if permitted: %s",
                        exc,
                    )
                elif not local_only and not prefer_local:
                    self.logger.warning(
                        "Remote load for faster-whisper failed: %s; trying fallbacks",
                        exc,
                    )

        fallback_models = ["large-v3", "large-v2", "medium", "small", "base", "tiny"]
        # Allow disabling the fallback ladder entirely. When set, failures
        # to load the requested model are surfaced instead of silently
        # switching to a different Whisper size.
        if os.getenv("DIAREMOT_DISABLE_WHISPER_FALLBACK", "").strip() == "1":
            fallback_models = []
        for fallback in fallback_models:
            for local_only in order:
                try:
                    self.logger.info(
                        "Trying fallback model: %s (local_only=%s)",
                        fallback,
                        local_only,
                    )
                    model = self._load_fallback_model(
                        WhisperModel,
                        fallback,
                        compute_type,
                        config,
                        local_only,
                    )
                    self.logger.info(
                        "Successfully loaded fallback faster-whisper: %s",
                        fallback,
                    )
                    return model
                except Exception as exc:
                    self.last_errors[f"faster_whisper_load:{fallback}:{local_only}"] = str(exc)
                    failure_notes.append(
                        f"fallback={fallback} local_only={local_only}: {exc}"
                    )
                    self.logger.warning(
                        "Fallback %s (local_only=%s) failed: %s",
                        fallback,
                        local_only,
                        exc,
                    )

        msg = [
            "No faster-whisper model could be loaded using local cache or remote fallbacks.",
            "Ensure the desired CTranslate2 weights exist under DIAREMOT_MODEL_DIR",
            "or rerun the CLI with --remote-first to allow downloads.",
        ]
        if failure_notes:
            msg.append("Attempts:" + " | ".join(failure_notes))
        raise RuntimeError(" ".join(msg))

    def _load_fallback_model(
        self,
        whisper_cls: Any,
        model_name: str,
        compute_type: str,
        config: dict[str, Any],
        local_only: bool,
    ) -> Any:
        return whisper_cls(
            model_name,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=config.get("cpu_threads", 1),
            download_root=None,
            local_files_only=local_only,
        )

    def _load_openai_whisper(self, config: dict[str, Any]) -> Any:
        model_name = self._map_to_openai_model(config["model_size"])
        try:
            import whisper as openai_whisper  # type: ignore

            model = openai_whisper.load_model(model_name, device="cpu")
            self.logger.info("Loaded OpenAI whisper: %s", model_name)
            return model
        except Exception as exc:
            self.last_errors["openai_whisper_load"] = str(exc)
            self.logger.warning(
                "OpenAI whisper load failed for '%s': %s; trying 'tiny'",
                model_name,
                exc,
            )
            import whisper as openai_whisper  # type: ignore

            tiny = openai_whisper.load_model("tiny", device="cpu")
            self.logger.info("Loaded OpenAI whisper fallback: tiny")
            return tiny

    @staticmethod
    def _map_to_openai_model(model_size: str) -> str:
        if "/" in model_size and "faster-whisper" in model_size:
            lowered = model_size.lower()
            if "turbo" in lowered:
                return "large-v3"
            if "large-v3" in lowered:
                return "large-v3"
            if "large-v2" in lowered:
                return "large-v2"
            if "medium" in lowered:
                return "medium"
            if "small" in lowered:
                return "small"
            if "base" in lowered:
                return "base"
            if "tiny" in lowered:
                return "tiny"

        name_map = {
            "turbo": "large-v3",
            "large-v3": "large-v3",
            "large-v2": "large-v2",
            "large": "large",
            "medium": "medium",
            "small": "small",
            "base": "base",
            "tiny": "tiny",
        }
        for key, value in name_map.items():
            if key in model_size.lower():
                return value
        return "large-v3"


def get_system_capabilities() -> dict[str, Any]:
    import multiprocessing
    import platform

    capabilities = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": multiprocessing.cpu_count(),
        "backends": {
            "faster_whisper": backends.has_faster_whisper,
            "openai_whisper": backends.has_openai_whisper,
            "librosa": backends.has_librosa,
        },
        "environment": {
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "not set"),
            "torch_device": os.getenv("TORCH_DEVICE", "not set"),
            "omp_num_threads": os.getenv("OMP_NUM_THREADS", "not set"),
        },
    }

    try:
        if backends.has_librosa:
            capabilities.setdefault("versions", {})["librosa"] = backends.librosa.__version__
    except Exception:  # pragma: no cover - optional
        pass

    try:
        if backends.has_faster_whisper:
            import faster_whisper  # type: ignore

            capabilities.setdefault("versions", {})["faster_whisper"] = getattr(
                faster_whisper,
                "__version__",
                "unknown",
            )
    except Exception:  # pragma: no cover - optional
        pass

    return capabilities
