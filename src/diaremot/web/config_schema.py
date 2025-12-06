"""Configuration schema generator for the web UI.

Extracts all configurable parameters from PipelineConfig and generates
a JSON schema with metadata for UI rendering.
"""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

from diaremot.pipeline.config import PipelineConfig


# Parameter metadata for UI rendering
PARAMETER_METADATA: dict[str, dict[str, Any]] = {
    # Diarization
    "vad_threshold": {
        "label": "VAD Threshold",
        "description": "Silero VAD probability threshold. Lower = more sensitive to speech.",
        "group": "diarization",
        "ui_type": "slider",
        "step": 0.05,
    },
    "vad_min_speech_sec": {
        "label": "Minimum Speech Duration",
        "description": "Minimum detected speech segment duration in seconds.",
        "group": "diarization",
        "ui_type": "number",
        "step": 0.1,
        "unit": "seconds",
    },
    "vad_min_silence_sec": {
        "label": "Minimum Silence Duration",
        "description": "Minimum silence duration between speech segments.",
        "group": "diarization",
        "ui_type": "number",
        "step": 0.1,
        "unit": "seconds",
    },
    "vad_speech_pad_sec": {
        "label": "Speech Padding",
        "description": "Padding added around detected speech regions.",
        "group": "diarization",
        "ui_type": "number",
        "step": 0.05,
        "unit": "seconds",
    },
    "vad_backend": {
        "label": "VAD Backend",
        "description": "Voice Activity Detection backend (auto/onnx/torch).",
        "group": "diarization",
        "ui_type": "select",
        "options": ["auto", "onnx", "torch"],
    },
    "ahc_distance_threshold": {
        "label": "AHC Distance Threshold",
        "description": "Agglomerative clustering distance threshold for speaker grouping.",
        "group": "diarization",
        "ui_type": "slider",
        "step": 0.01,
    },
    "speaker_limit": {
        "label": "Speaker Limit",
        "description": "Maximum number of speakers to detect (null for unlimited).",
        "group": "diarization",
        "ui_type": "number",
        "nullable": True,
    },
    "clustering_backend": {
        "label": "Clustering Backend",
        "description": "Clustering algorithm (auto=spectral→AHC, ahc, spectral).",
        "group": "diarization",
        "ui_type": "select",
        "options": ["auto", "ahc", "spectral"],
    },
    "min_speakers": {
        "label": "Minimum Speakers",
        "description": "Minimum speakers for spectral clustering.",
        "group": "diarization",
        "ui_type": "number",
        "nullable": True,
        "advanced": True,
    },
    "max_speakers": {
        "label": "Maximum Speakers",
        "description": "Maximum speakers for spectral clustering.",
        "group": "diarization",
        "ui_type": "number",
        "nullable": True,
        "advanced": True,
    },
    "spectral_p_percentile": {
        "label": "Spectral Affinity Percentile",
        "description": "Row-wise percentile to keep strongest embedding affinities (0-1).",
        "group": "diarization",
        "ui_type": "slider",
        "step": 0.01,
        "advanced": True,
    },
    "spectral_silhouette_floor": {
        "label": "Spectral Silhouette Floor",
        "description": "Reject spectral labels if cosine silhouette falls below this value.",
        "group": "diarization",
        "ui_type": "slider",
        "step": 0.01,
        "advanced": True,
    },
    "spectral_refine_with_ahc": {
        "label": "Refine Spectral with AHC",
        "description": "Run AHC with the spectral speaker count to stabilise boundaries.",
        "group": "diarization",
        "ui_type": "checkbox",
        "advanced": True,
    },
    "disable_energy_vad_fallback": {
        "label": "Disable Energy VAD Fallback",
        "description": "Disable energy-based VAD fallback when Silero fails.",
        "group": "diarization",
        "ui_type": "toggle",
        "advanced": True,
    },
    "energy_gate_db": {
        "label": "Energy Gate Threshold",
        "description": "Energy VAD gating threshold in dB.",
        "group": "diarization",
        "ui_type": "number",
        "unit": "dB",
        "advanced": True,
    },
    "energy_hop_sec": {
        "label": "Energy VAD Hop Length",
        "description": "Energy VAD hop length in seconds.",
        "group": "diarization",
        "ui_type": "number",
        "step": 0.001,
        "unit": "seconds",
        "advanced": True,
    },
    # Transcription
    "whisper_model": {
        "label": "Whisper Model",
        "description": "Whisper/Faster-Whisper model identifier.",
        "group": "transcription",
        "ui_type": "select",
        "options": [
            "tiny.en",
            "tiny",
            "base.en",
            "base",
            "small.en",
            "small",
            "medium.en",
            "medium",
            "large-v2",
            "large-v3",
        ],
    },
    "asr_backend": {
        "label": "ASR Backend",
        "description": "Automatic Speech Recognition backend.",
        "group": "transcription",
        "ui_type": "select",
        "options": ["faster", "whisper"],
    },
    "compute_type": {
        "label": "Compute Type",
        "description": "CTranslate2 compute type (int8 is fastest, float32 is most accurate).",
        "group": "transcription",
        "ui_type": "select",
        "options": ["int8", "float16", "float32"],
    },
    "cpu_threads": {
        "label": "CPU Threads",
        "description": "Number of CPU threads for ASR backend.",
        "group": "transcription",
        "ui_type": "number",
        "min": 1,
        "max": 16,
    },
    "language": {
        "label": "Language",
        "description": "Override ASR language detection (e.g., 'en', 'es', 'fr').",
        "group": "transcription",
        "ui_type": "text",
        "nullable": True,
        "placeholder": "auto-detect",
    },
    "language_mode": {
        "label": "Language Mode",
        "description": "Language detection mode.",
        "group": "transcription",
        "ui_type": "select",
        "options": ["auto", "manual"],
    },
    "beam_size": {
        "label": "Beam Size",
        "description": "Beam search size (larger = more accurate but slower).",
        "group": "transcription",
        "ui_type": "number",
        "min": 1,
        "max": 10,
    },
    "temperature": {
        "label": "Sampling Temperature",
        "description": "Sampling temperature for ASR (0.0 = deterministic).",
        "group": "transcription",
        "ui_type": "slider",
        "step": 0.1,
    },
    "no_speech_threshold": {
        "label": "No-Speech Threshold",
        "description": "Threshold for detecting silence in Whisper.",
        "group": "transcription",
        "ui_type": "slider",
        "step": 0.05,
    },
    "ignore_tx_cache": {
        "label": "Ignore Transcription Cache",
        "description": "Ignore cached transcription results.",
        "group": "transcription",
        "ui_type": "toggle",
        "advanced": True,
    },
    "enable_async_transcription": {
        "label": "Async Transcription",
        "description": "Enable asynchronous transcription engine.",
        "group": "transcription",
        "ui_type": "toggle",
        "advanced": True,
    },
    "max_asr_window_sec": {
        "label": "Max ASR Window",
        "description": "Maximum audio length per ASR window in seconds.",
        "group": "transcription",
        "ui_type": "number",
        "unit": "seconds",
        "advanced": True,
    },
    "segment_timeout_sec": {
        "label": "Segment Timeout",
        "description": "Timeout per ASR segment in seconds.",
        "group": "transcription",
        "ui_type": "number",
        "unit": "seconds",
        "advanced": True,
    },
    "batch_timeout_sec": {
        "label": "Batch Timeout",
        "description": "Timeout for batch of ASR segments in seconds.",
        "group": "transcription",
        "ui_type": "number",
        "unit": "seconds",
        "advanced": True,
    },
    # Affect & Emotion
    "disable_affect": {
        "label": "Disable Affect Analysis",
        "description": "Skip emotion and affect analysis stages.",
        "group": "affect",
        "ui_type": "toggle",
    },
    "affect_backend": {
        "label": "Affect Backend",
        "description": "Backend for affect models (onnx is faster, torch is fallback).",
        "group": "affect",
        "ui_type": "select",
        "options": ["auto", "onnx", "torch"],
    },
    "affect_text_model_dir": {
        "label": "Text Emotion Model Directory",
        "description": "Path to GoEmotions text emotion model.",
        "group": "affect",
        "ui_type": "path",
        "nullable": True,
        "advanced": True,
    },
    "affect_intent_model_dir": {
        "label": "Intent Model Directory",
        "description": "Path to intent classification model.",
        "group": "affect",
        "ui_type": "path",
        "nullable": True,
        "advanced": True,
    },
    "affect_ser_model_dir": {
        "label": "Speech Emotion Model Directory",
        "description": "Path to speech emotion recognition model.",
        "group": "affect",
        "ui_type": "path",
        "nullable": True,
        "advanced": True,
    },
    "affect_vad_model_dir": {
        "label": "VAD Model Directory",
        "description": "Path to valence/arousal/dominance model.",
        "group": "affect",
        "ui_type": "path",
        "nullable": True,
        "advanced": True,
    },
    "affect_analyzer_threads": {
        "label": "Affect Analyzer Threads",
        "description": "Number of threads for affect analyzers.",
        "group": "affect",
        "ui_type": "number",
        "nullable": True,
        "advanced": True,
    },
    # Sound Event Detection
    "enable_sed": {
        "label": "Enable Sound Event Detection",
        "description": "Enable background sound event detection (music, keyboard, etc.).",
        "group": "sed",
        "ui_type": "toggle",
    },
    "sed_mode": {
        "label": "SED Mode",
        "description": "Sound event detection mode (auto/global/timeline).",
        "group": "sed",
        "ui_type": "select",
        "options": ["auto", "global", "timeline"],
    },
    "sed_window_sec": {
        "label": "SED Window Length",
        "description": "Timeline SED window length in seconds.",
        "group": "sed",
        "ui_type": "number",
        "step": 0.1,
        "unit": "seconds",
    },
    "sed_hop_sec": {
        "label": "SED Hop Length",
        "description": "Timeline SED hop length in seconds.",
        "group": "sed",
        "ui_type": "number",
        "step": 0.1,
        "unit": "seconds",
    },
    "sed_enter": {
        "label": "SED Enter Threshold",
        "description": "Hysteresis enter threshold for sound events.",
        "group": "sed",
        "ui_type": "slider",
        "step": 0.05,
    },
    "sed_exit": {
        "label": "SED Exit Threshold",
        "description": "Hysteresis exit threshold for sound events.",
        "group": "sed",
        "ui_type": "slider",
        "step": 0.05,
    },
    "sed_merge_gap": {
        "label": "SED Merge Gap",
        "description": "Merge sound events separated by this gap in seconds.",
        "group": "sed",
        "ui_type": "number",
        "step": 0.05,
        "unit": "seconds",
    },
    "sed_default_min_dur": {
        "label": "SED Default Min Duration",
        "description": "Default minimum event duration in seconds.",
        "group": "sed",
        "ui_type": "number",
        "step": 0.1,
        "unit": "seconds",
    },
    "sed_median_k": {
        "label": "SED Median Filter Size",
        "description": "Median filter kernel size (must be odd).",
        "group": "sed",
        "ui_type": "number",
        "advanced": True,
    },
    "sed_batch_size": {
        "label": "SED Batch Size",
        "description": "Batch size for SED inference.",
        "group": "sed",
        "ui_type": "number",
        "advanced": True,
    },
    "sed_max_windows": {
        "label": "SED Max Windows",
        "description": "Maximum number of windows to process.",
        "group": "sed",
        "ui_type": "number",
        "advanced": True,
    },
    "sed_timeline_jsonl": {
        "label": "Write SED Debug JSONL",
        "description": "Write per-frame SED debug output.",
        "group": "sed",
        "ui_type": "toggle",
        "advanced": True,
    },
    "sed_classmap_csv": {
        "label": "SED Class Map CSV",
        "description": "CSV mapping AudioSet labels to collapsed groups.",
        "group": "sed",
        "ui_type": "path",
        "nullable": True,
        "advanced": True,
    },
    # Preprocessing
    "noise_reduction": {
        "label": "Noise Reduction",
        "description": "Enable gentle noise reduction.",
        "group": "preprocessing",
        "ui_type": "toggle",
    },
    "auto_chunk_enabled": {
        "label": "Auto Chunking",
        "description": "Automatically chunk long audio files.",
        "group": "preprocessing",
        "ui_type": "toggle",
    },
    "chunk_threshold_minutes": {
        "label": "Chunk Threshold",
        "description": "Activate chunking for files longer than this.",
        "group": "preprocessing",
        "ui_type": "number",
        "unit": "minutes",
    },
    "chunk_size_minutes": {
        "label": "Chunk Size",
        "description": "Size of each chunk in minutes.",
        "group": "preprocessing",
        "ui_type": "number",
        "unit": "minutes",
    },
    "chunk_overlap_seconds": {
        "label": "Chunk Overlap",
        "description": "Overlap between chunks in seconds.",
        "group": "preprocessing",
        "ui_type": "number",
        "unit": "seconds",
    },
    "target_sr": {
        "label": "Target Sample Rate",
        "description": "Target sample rate for preprocessing.",
        "group": "preprocessing",
        "ui_type": "number",
        "unit": "Hz",
        "advanced": True,
    },
    "loudness_mode": {
        "label": "Loudness Mode",
        "description": "Loudness normalization mode.",
        "group": "preprocessing",
        "ui_type": "select",
        "options": ["asr", "broadcast"],
        "advanced": True,
    },
    # Advanced
    "registry_path": {
        "label": "Speaker Registry Path",
        "description": "Path to persistent speaker registry file.",
        "group": "advanced",
        "ui_type": "path",
    },
    "cache_root": {
        "label": "Cache Root Directory",
        "description": "Root directory for caching.",
        "group": "advanced",
        "ui_type": "path",
    },
    "log_dir": {
        "label": "Log Directory",
        "description": "Directory for log files.",
        "group": "advanced",
        "ui_type": "path",
    },
    "checkpoint_dir": {
        "label": "Checkpoint Directory",
        "description": "Directory for pipeline checkpoints.",
        "group": "advanced",
        "ui_type": "path",
    },
    "quiet": {
        "label": "Quiet Mode",
        "description": "Reduce console verbosity.",
        "group": "advanced",
        "ui_type": "toggle",
    },
    "cpu_diarizer": {
        "label": "CPU-Optimized Diarizer",
        "description": "Enable CPU-optimized diarization wrapper.",
        "group": "advanced",
        "ui_type": "toggle",
    },
    "local_first": {
        "label": "Local Models First",
        "description": "Prefer local models before remote downloads.",
        "group": "advanced",
        "ui_type": "toggle",
    },
}

# Group definitions with ordering
PARAMETER_GROUPS = {
    "diarization": {
        "label": "Diarization",
        "description": "Speaker segmentation and voice activity detection",
        "icon": "users",
        "order": 1,
    },
    "transcription": {
        "label": "Transcription",
        "description": "Automatic speech recognition settings",
        "icon": "text",
        "order": 2,
    },
    "affect": {
        "label": "Affect & Emotion",
        "description": "Emotion analysis and intent classification",
        "icon": "heart",
        "order": 3,
    },
    "sed": {
        "label": "Sound Events",
        "description": "Background sound event detection",
        "icon": "music",
        "order": 4,
    },
    "preprocessing": {
        "label": "Preprocessing",
        "description": "Audio preprocessing and chunking",
        "icon": "settings",
        "order": 5,
    },
    "advanced": {
        "label": "Advanced",
        "description": "Advanced configuration options",
        "icon": "code",
        "order": 6,
    },
}


def generate_config_schema() -> dict[str, Any]:
    """Generate comprehensive config schema from PipelineConfig.

    Returns:
        Dictionary with:
        - parameters: Parameter definitions with metadata
        - groups: Group definitions
        - defaults: Default values
        - presets: Available presets
    """
    schema: dict[str, Any] = {
        "parameters": {},
        "groups": PARAMETER_GROUPS,
        "defaults": {},
        "presets": {},
    }

    # Extract fields from PipelineConfig dataclass
    config_fields = fields(PipelineConfig)

    for field in config_fields:
        field_name = field.name
        field_type = field.type

        # Handle default values properly
        if field.default is not MISSING:
            default_value = field.default
        elif field.default_factory is not MISSING:
            default_value = field.default_factory()
        else:
            # Field has no default, use None as fallback
            default_value = None

        # Get metadata or create basic entry
        metadata = PARAMETER_METADATA.get(field_name, {})

        # Build parameter definition
        param_def: dict[str, Any] = {
            "name": field_name,
            "type": _get_json_type(field_type),
            "default": _serialize_value(default_value),
            "label": metadata.get("label", _camel_to_title(field_name)),
            "description": metadata.get("description", ""),
            "group": metadata.get("group", "advanced"),
            "ui_type": metadata.get("ui_type", _infer_ui_type(field_type)),
            "advanced": metadata.get("advanced", False),
        }

        # Add UI-specific metadata
        if "min" in metadata:
            param_def["min"] = metadata["min"]
        if "max" in metadata:
            param_def["max"] = metadata["max"]
        if "step" in metadata:
            param_def["step"] = metadata["step"]
        if "options" in metadata:
            param_def["options"] = metadata["options"]
        if "unit" in metadata:
            param_def["unit"] = metadata["unit"]
        if "nullable" in metadata:
            param_def["nullable"] = metadata["nullable"]
        if "placeholder" in metadata:
            param_def["placeholder"] = metadata["placeholder"]

        schema["parameters"][field_name] = param_def
        schema["defaults"][field_name] = _serialize_value(default_value)

    # Add built-in presets
    schema["presets"] = _get_builtin_presets()

    return schema


def _get_json_type(python_type: Any) -> str:
    """Convert Python type annotation to JSON schema type."""
    type_str = str(python_type)

    if "int" in type_str:
        return "integer"
    elif "float" in type_str:
        return "number"
    elif "bool" in type_str:
        return "boolean"
    elif "str" in type_str:
        return "string"
    elif "list" in type_str or "List" in type_str:
        return "array"
    elif "dict" in type_str or "Dict" in type_str:
        return "object"
    elif "Path" in type_str:
        return "string"
    else:
        return "string"


def _infer_ui_type(python_type: Any) -> str:
    """Infer UI control type from Python type."""
    type_str = str(python_type)

    if "bool" in type_str:
        return "toggle"
    elif "int" in type_str:
        return "number"
    elif "float" in type_str:
        return "slider"
    elif "Path" in type_str:
        return "path"
    else:
        return "text"


def _serialize_value(value: Any) -> Any:
    """Serialize value for JSON schema."""
    from pathlib import Path

    if isinstance(value, Path):
        return str(value)
    elif isinstance(value, (list, dict)):
        return value
    elif value is None:
        return None
    else:
        return value


def _camel_to_title(name: str) -> str:
    """Convert camel_case to Title Case."""
    return " ".join(word.capitalize() for word in name.split("_"))


def _get_builtin_presets() -> dict[str, dict[str, Any]]:
    """Get built-in configuration presets."""
    return {
        "fast": {
            "name": "Fast",
            "description": "Optimized for speed with int8 quantization and minimal analysis",
            "overrides": {
                "whisper_model": "tiny.en",
                "compute_type": "int8",
                "beam_size": 1,
                "temperature": 0.0,
                "disable_affect": True,
                "enable_sed": False,
                "noise_reduction": False,
            },
        },
        "accurate": {
            "name": "Accurate",
            "description": "Optimized for accuracy with larger models and beam search",
            "overrides": {
                "whisper_model": "medium.en",
                "compute_type": "float32",
                "beam_size": 4,
                "temperature": 0.0,
                "no_speech_threshold": 0.2,
                "disable_affect": False,
                "enable_sed": True,
            },
        },
        "offline": {
            "name": "Offline",
            "description": "ONNX-only models with no remote downloads",
            "overrides": {
                "affect_backend": "onnx",
                "vad_backend": "onnx",
                "local_first": True,
                "disable_affect": False,
            },
        },
        "balanced": {
            "name": "Balanced",
            "description": "Balanced speed and accuracy (default settings)",
            "overrides": {},
        },
    }
