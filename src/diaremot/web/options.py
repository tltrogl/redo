from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from diaremot.cli import BUILTIN_PROFILES

TRUTHY = {"1", "true", "on", "yes"}
FALSY = {"0", "false", "off", "no"}

GROUPS: tuple[str, ...] = (
    "General & Profiles",
    "ASR & Language",
    "Diarization & VAD",
    "Sound Events & Affect",
    "Chunking & Performance",
    "Advanced Paths & Debugging",
)


@dataclass(frozen=True)
class PipelineOption:
    key: str
    label: str
    description: str
    field_type: str
    group: str
    default: Any = None
    config_key: str | None = None
    choices: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    advanced: bool = False
    placeholder: str | None = None
    step: float | None = None
    include_in_cli: bool = True
    cli_flag: str | None = None
    cli_invert: bool = False

    def to_metadata(self) -> dict[str, Any]:
        cli_flag = None
        if self.include_in_cli:
            name = self.cli_flag or self.key
            cli_flag = name.replace("_", "-")
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "type": self.field_type,
            "group": self.group,
            "default": self.default,
            "choices": list(self.choices),
            "advanced": self.advanced,
            "placeholder": self.placeholder,
            "step": self.step,
            "cliFlag": cli_flag,
            "cliInvert": self.cli_invert,
        }


def _choice(label: str, value: Any) -> dict[str, Any]:
    return {"label": label, "value": value}


def _profile_choices() -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    for key in BUILTIN_PROFILES:
        entries.append(_choice(key.title(), key))
    return tuple(entries)


PIPELINE_OPTIONS: list[PipelineOption] = [
    PipelineOption(
        key="profile",
        label="Profile",
        description="Preset overrides applied before the rest of the form.",
        field_type="select",
        group=GROUPS[0],
        default="default",
        choices=_profile_choices(),
    ),
    PipelineOption(
        key="profile_overrides",
        label="Profile Overrides (JSON)",
        description="Optional JSON object merged on top of the selected profile before other options.",
        field_type="textarea",
        group=GROUPS[0],
        advanced=True,
        placeholder='{"compute_type": "float32"}',
        include_in_cli=False,
    ),
    PipelineOption(
        key="model_root",
        label="Model Root",
        description="Overrides DIAREMOT_MODEL_DIR for this run.",
        field_type="text",
        group=GROUPS[0],
        advanced=True,
    ),
    PipelineOption(
        key="registry_path",
        label="Speaker Registry Path",
        description="Location of the persistent speaker registry JSON file.",
        field_type="text",
        group=GROUPS[5],
        default="speaker_registry.json",
        config_key="registry_path",
        advanced=True,
    ),
    PipelineOption(
        key="clear_cache",
        label="Clear cache before run",
        description="Delete diarization/ASR caches before running.",
        field_type="boolean",
        group=GROUPS[0],
        default=False,
    ),
    PipelineOption(
        key="remote_first",
        label="Prefer remote model downloads",
        description="When enabled the runtime skips local-first model discovery.",
        field_type="boolean",
        group=GROUPS[5],
        default=False,
        advanced=True,
        config_key="local_first",
        cli_flag="remote-first",
    ),
    PipelineOption(
        key="ignore_tx_cache",
        label="Ignore transcription cache",
        description="Force a fresh ASR run even when cached transcripts exist.",
        field_type="boolean",
        group=GROUPS[5],
        default=False,
    ),
    PipelineOption(
        key="quiet",
        label="Quiet console output",
        description="Reduces CLI verbosity for long runs.",
        field_type="boolean",
        group=GROUPS[0],
        default=False,
        advanced=True,
    ),
    PipelineOption(
        key="async_asr",
        label="Async ASR scheduler",
        description="Run the Faster-Whisper async scheduler for overlapping ASR windows.",
        field_type="boolean",
        group=GROUPS[1],
        default=False,
        config_key="enable_async_transcription",
    ),
    PipelineOption(
        key="language",
        label="Language override",
        description="Optional ISO language code passed to the ASR backend.",
        field_type="text",
        group=GROUPS[1],
    ),
    PipelineOption(
        key="language_mode",
        label="Language mode",
        description="Controls how the ASR backend interprets language hints.",
        field_type="select",
        group=GROUPS[1],
        default="auto",
        choices=(
            _choice("Auto detect", "auto"),
            _choice("Manual override", "manual"),
        ),
    ),
    PipelineOption(
        key="cpu_diarizer",
        label="CPU-optimised diarizer",
        description="Enable the lighter-weight diarizer wrapper.",
        field_type="boolean",
        group=GROUPS[2],
        default=False,
        advanced=True,
    ),
    PipelineOption(
        key="whisper_model",
        label="Whisper model",
        description="Identifier passed to Faster-Whisper (e.g. faster-whisper-tiny.en).",
        field_type="text",
        group=GROUPS[1],
        default="faster-whisper-tiny.en",
    ),
    PipelineOption(
        key="asr_backend",
        label="ASR backend",
        description="Selects the ASR backend implementation.",
        field_type="select",
        group=GROUPS[1],
        default="faster",
        choices=(
            _choice("Faster-Whisper", "faster"),
            _choice("Whisper (reference)", "whisper"),
        ),
    ),
    PipelineOption(
        key="asr_compute_type",
        label="ASR compute type",
        description="CTranslate2 compute precision used by Faster-Whisper.",
        field_type="select",
        group=GROUPS[1],
        default="int8",
        config_key="compute_type",
        choices=(
            _choice("int8 (fastest)", "int8"),
            _choice("float32", "float32"),
            _choice("int8_float16", "int8_float16"),
        ),
    ),
    PipelineOption(
        key="asr_cpu_threads",
        label="ASR CPU threads",
        description="Number of CPU threads dedicated to the ASR backend.",
        field_type="integer",
        group=GROUPS[1],
        default=1,
        step=1,
    ),
    PipelineOption(
        key="beam_size",
        label="Beam size",
        description="Beam size passed to the decoder.",
        field_type="integer",
        group=GROUPS[1],
        default=1,
        step=1,
    ),
    PipelineOption(
        key="temperature",
        label="Sampling temperature",
        description="Sampling temperature for ASR decoding.",
        field_type="float",
        group=GROUPS[1],
        default=0.0,
        step=0.05,
    ),
    PipelineOption(
        key="no_speech_threshold",
        label="No-speech threshold",
        description="Probability threshold used to drop silent segments.",
        field_type="float",
        group=GROUPS[1],
        default=0.20,
        step=0.01,
    ),
    PipelineOption(
        key="asr_window_sec",
        label="ASR window (sec)",
        description="Maximum duration submitted per ASR window.",
        field_type="integer",
        group=GROUPS[1],
        default=480,
        step=1,
    ),
    PipelineOption(
        key="asr_segment_timeout",
        label="ASR segment timeout",
        description="Timeout per ASR segment (seconds).",
        field_type="float",
        group=GROUPS[1],
        default=300.0,
        step=1.0,
    ),
    PipelineOption(
        key="asr_batch_timeout",
        label="ASR batch timeout",
        description="Timeout per ASR batch (seconds).",
        field_type="float",
        group=GROUPS[1],
        default=1200.0,
        step=1.0,
    ),
    PipelineOption(
        key="clustering_backend",
        label="Clustering backend",
        description="Diarization clustering algorithm.",
        field_type="select",
        group=GROUPS[2],
        default="ahc",
        choices=(
            _choice("Agglomerative", "ahc"),
            _choice("Spectral", "spectral"),
        ),
    ),
    PipelineOption(
        key="speaker_limit",
        label="Speaker limit",
        description="Upper bound on discovered speakers (blank = auto).",
        field_type="integer",
        group=GROUPS[2],
        step=1,
    ),
    PipelineOption(
        key="min_speakers",
        label="Min speakers (spectral)",
        description="Lower bound on speakers when using spectral clustering.",
        field_type="integer",
        group=GROUPS[2],
        advanced=True,
        step=1,
    ),
    PipelineOption(
        key="max_speakers",
        label="Max speakers (spectral)",
        description="Upper bound on speakers when using spectral clustering.",
        field_type="integer",
        group=GROUPS[2],
        advanced=True,
        step=1,
    ),
    PipelineOption(
        key="ahc_distance_threshold",
        label="AHC distance threshold",
        description="Agglomerative clustering threshold (smaller merges more).",
        field_type="float",
        group=GROUPS[2],
        default=0.15,
        step=0.01,
    ),
    PipelineOption(
        key="vad_backend",
        label="VAD backend",
        description="Voice activity detector backend.",
        field_type="select",
        group=GROUPS[2],
        default="auto",
        choices=(
            _choice("Auto", "auto"),
            _choice("ONNX", "onnx"),
            _choice("Torch", "torch"),
        ),
    ),
    PipelineOption(
        key="vad_threshold",
        label="VAD threshold",
        description="Silero probability threshold.",
        field_type="float",
        group=GROUPS[2],
        default=0.35,
        step=0.01,
    ),
    PipelineOption(
        key="vad_min_speech_sec",
        label="Min speech (sec)",
        description="Minimum detected speech duration.",
        field_type="float",
        group=GROUPS[2],
        default=0.8,
        step=0.05,
    ),
    PipelineOption(
        key="vad_min_silence_sec",
        label="Min silence (sec)",
        description="Minimum detected silence duration.",
        field_type="float",
        group=GROUPS[2],
        default=0.8,
        step=0.05,
    ),
    PipelineOption(
        key="vad_speech_pad_sec",
        label="Speech padding (sec)",
        description="Padding applied around detected speech windows.",
        field_type="float",
        group=GROUPS[2],
        default=0.1,
        step=0.05,
    ),
    PipelineOption(
        key="disable_energy_vad_fallback",
        label="Disable energy VAD fallback",
        description="Prevents energy-based VAD rescue when Silero fails.",
        field_type="boolean",
        group=GROUPS[2],
        default=False,
        advanced=True,
    ),
    PipelineOption(
        key="energy_gate_db",
        label="Energy gate (dB)",
        description="Energy VAD gating threshold in dB.",
        field_type="float",
        group=GROUPS[2],
        default=-33.0,
        step=0.5,
        advanced=True,
    ),
    PipelineOption(
        key="energy_hop_sec",
        label="Energy hop (sec)",
        description="Energy VAD hop size in seconds.",
        field_type="float",
        group=GROUPS[2],
        default=0.01,
        step=0.01,
        advanced=True,
    ),
    PipelineOption(
        key="noise_reduction",
        label="Enable noise reduction",
        description="Apply gentle spectral subtraction before diarization.",
        field_type="boolean",
        group=GROUPS[4],
        default=False,
    ),
    PipelineOption(
        key="chunk_enabled",
        label="Chunking mode",
        description="Controls automatic chunking of long recordings.",
        field_type="select",
        group=GROUPS[4],
        default="auto",
        choices=(
            _choice("Auto", "auto"),
            _choice("Force enabled", "true"),
            _choice("Force disabled", "false"),
        ),
    ),
    PipelineOption(
        key="chunk_threshold_minutes",
        label="Chunk threshold (min)",
        description="Duration that triggers chunking when auto is enabled.",
        field_type="float",
        group=GROUPS[4],
        default=30.0,
        step=1.0,
    ),
    PipelineOption(
        key="chunk_size_minutes",
        label="Chunk size (min)",
        description="Chunk duration when splitting long recordings.",
        field_type="float",
        group=GROUPS[4],
        default=20.0,
        step=1.0,
    ),
    PipelineOption(
        key="chunk_overlap_seconds",
        label="Chunk overlap (sec)",
        description="Temporal overlap between consecutive chunks.",
        field_type="float",
        group=GROUPS[4],
        default=30.0,
        step=1.0,
    ),
    PipelineOption(
        key="disable_affect",
        label="Disable affect stage",
        description="Skips affect/intent analysis when enabled.",
        field_type="boolean",
        group=GROUPS[3],
        default=False,
    ),
    PipelineOption(
        key="affect_backend",
        label="Affect backend",
        description="Preferred backend for affect models.",
        field_type="select",
        group=GROUPS[3],
        default="onnx",
        choices=(
            _choice("ONNX", "onnx"),
            _choice("Torch", "torch"),
            _choice("Auto", "auto"),
        ),
    ),
    PipelineOption(
        key="affect_text_model_dir",
        label="Text emotion model dir",
        description="Path to the GoEmotions ONNX export.",
        field_type="text",
        group=GROUPS[5],
        advanced=True,
    ),
    PipelineOption(
        key="affect_intent_model_dir",
        label="Intent model dir",
        description="Path to the BART-MNLI intent model.",
        field_type="text",
        group=GROUPS[5],
        advanced=True,
    ),
    PipelineOption(
        key="affect_ser_model_dir",
        label="Speech emotion model dir",
        description="Path to the SER-8 ONNX model.",
        field_type="text",
        group=GROUPS[5],
        advanced=True,
    ),
    PipelineOption(
        key="affect_vad_model_dir",
        label="VAD (dimensional) model dir",
        description="Path to the valence/arousal/dominance ONNX model.",
        field_type="text",
        group=GROUPS[5],
        advanced=True,
    ),
    PipelineOption(
        key="enable_sed",
        label="Enable background SED",
        description="Runs sound event detection and timelines.",
        field_type="boolean",
        group=GROUPS[3],
        default=True,
        cli_flag="disable-sed",
        cli_invert=True,
    ),
    PipelineOption(
        key="sed_mode",
        label="SED mode",
        description="Controls whether the detailed timeline runs.",
        field_type="select",
        group=GROUPS[3],
        default="auto",
        choices=(
            _choice("Auto", "auto"),
            _choice("Global only", "global"),
            _choice("Timeline", "timeline"),
        ),
    ),
    PipelineOption(
        key="sed_window_sec",
        label="SED window (sec)",
        description="Window size used by the SED timeline.",
        field_type="float",
        group=GROUPS[3],
        default=1.0,
        step=0.1,
    ),
    PipelineOption(
        key="sed_hop_sec",
        label="SED hop (sec)",
        description="Hop length for the SED timeline.",
        field_type="float",
        group=GROUPS[3],
        default=0.5,
        step=0.1,
    ),
    PipelineOption(
        key="sed_enter",
        label="SED enter threshold",
        description="Timeline hysteresis enter threshold.",
        field_type="float",
        group=GROUPS[3],
        default=0.5,
        step=0.05,
        advanced=True,
    ),
    PipelineOption(
        key="sed_exit",
        label="SED exit threshold",
        description="Timeline hysteresis exit threshold.",
        field_type="float",
        group=GROUPS[3],
        default=0.35,
        step=0.05,
        advanced=True,
    ),
    PipelineOption(
        key="sed_merge_gap",
        label="SED merge gap",
        description="Merge SED events separated by <= gap seconds.",
        field_type="float",
        group=GROUPS[3],
        default=0.20,
        step=0.05,
        advanced=True,
    ),
    PipelineOption(
        key="sed_min_dur",
        label="SED min duration map",
        description="JSON or key=value pairs mapping collapsed labels to minimum duration seconds.",
        field_type="textarea",
        group=GROUPS[3],
        advanced=True,
        placeholder='{"noise": 1.5, "music": 0.5}'
    ),
    PipelineOption(
        key="sed_classmap_csv",
        label="SED classmap CSV",
        description="Optional CSV mapping AudioSet labels to collapsed groups.",
        field_type="text",
        group=GROUPS[5],
        advanced=True,
        cli_flag="sed-classmap",
    ),
    PipelineOption(
        key="sed_timeline_jsonl",
        label="Write SED debug JSONL",
        description="Persists the per-frame SED timeline alongside the events CSV.",
        field_type="boolean",
        group=GROUPS[5],
        default=False,
        advanced=True,
        config_key="sed_timeline_jsonl",
        cli_flag="sed-write-jsonl",
    ),
    PipelineOption(
        key="sed_batch_size",
        label="SED batch size",
        description="Batch size used when running the SED ONNX model.",
        field_type="integer",
        group=GROUPS[3],
        default=256,
        step=1,
        advanced=True,
        include_in_cli=False,
    ),
    PipelineOption(
        key="sed_median_k",
        label="SED median filter window",
        description="Odd-sized median window applied to SED logits.",
        field_type="integer",
        group=GROUPS[3],
        default=5,
        step=1,
        advanced=True,
        include_in_cli=False,
    ),
]


_OPTION_INDEX = {opt.key: opt for opt in PIPELINE_OPTIONS}


@dataclass
class ResolvedPipelineOptions:
    overrides: dict[str, Any]
    clear_cache: bool
    model_root: str | None


def list_pipeline_options() -> dict[str, Any]:
    return {
        "groups": list(GROUPS),
        "options": [opt.to_metadata() for opt in PIPELINE_OPTIONS],
    }


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    lowered = str(value).strip().lower()
    if lowered in TRUTHY:
        return True
    if lowered in FALSY:
        return False
    raise ValueError(f"Cannot interpret '{value}' as a boolean flag")


def _parse_json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("JSON overrides must be an object mapping keys to values")
    return parsed


def _parse_min_duration_map(value: str) -> dict[str, float]:
    text = value.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        entries: dict[str, float] = {}
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    "sed_min_dur must be JSON or comma-separated key=value pairs"
                )
            key, raw_val = part.split("=", 1)
            entries[key.strip()] = float(raw_val)
        return entries
    if not isinstance(parsed, dict):
        raise ValueError("sed_min_dur JSON must be an object")
    return {str(key): float(val) for key, val in parsed.items()}


def _coerce_numeric(value: Any, *, as_int: bool) -> Any:
    if value in {None, ""}:
        return None
    return int(value) if as_int else float(value)


def resolve_pipeline_options(payload: dict[str, Any]) -> ResolvedPipelineOptions:
    profile_name: str | None = None
    profile_overrides: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    clear_cache = False
    model_root: str | None = None

    for key, raw_value in payload.items():
        option = _OPTION_INDEX.get(key)
        if option is None:
            continue
        value: Any
        if option.field_type == "boolean":
            value = _parse_bool(raw_value)
        elif option.field_type == "integer":
            value = _coerce_numeric(raw_value, as_int=True)
        elif option.field_type == "float":
            value = _coerce_numeric(raw_value, as_int=False)
        elif option.field_type == "textarea":
            if option.key == "profile_overrides":
                text = str(raw_value or "").strip()
                value = _parse_json_object(text) if text else {}
            elif option.key == "sed_min_dur":
                text = str(raw_value or "").strip()
                value = _parse_min_duration_map(text) if text else None
            else:
                text = str(raw_value or "").strip()
                value = text or None
        elif option.field_type == "select":
            text = str(raw_value) if raw_value is not None else ""
            if option.key == "chunk_enabled":
                lowered = text.lower()
                if lowered in {"", "auto"}:
                    value = None
                elif lowered in {"true", "enabled", "1"}:
                    value = True
                elif lowered in {"false", "disabled", "0"}:
                    value = False
                else:
                    raise ValueError(f"Unknown chunk_enabled value '{raw_value}'")
            elif option.key == "profile":
                trimmed = text.strip()
                value = trimmed or None
            else:
                value = text.strip() or None
        else:  # default to text inputs
            text = str(raw_value or "").strip()
            value = text or None

        if option.key == "profile":
            profile_name = value
            continue
        if option.key == "profile_overrides":
            profile_overrides = value or {}
            continue
        if option.key == "clear_cache":
            clear_cache = bool(value)
            continue
        if option.key == "model_root":
            model_root = value
            continue
        if option.key == "remote_first":
            overrides[option.config_key or "local_first"] = not bool(value)
            continue
        if option.key == "chunk_enabled":
            overrides[option.key] = value
            continue
        if value is None and option.field_type != "boolean":
            continue
        config_key = option.config_key or option.key
        overrides[config_key] = value

    base_overrides: dict[str, Any] = {}
    if profile_name:
        profile_value = BUILTIN_PROFILES.get(profile_name)
        if profile_value is None:
            raise ValueError(f"Unknown profile '{profile_name}'")
        base_overrides.update(profile_value)
    if profile_overrides:
        base_overrides.update(profile_overrides)

    final_overrides = {**base_overrides, **overrides}
    return ResolvedPipelineOptions(overrides=final_overrides, clear_cache=clear_cache, model_root=model_root)
