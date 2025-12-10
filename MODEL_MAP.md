# DiaRemot Model Map - Complete Reference

**Date:** 2025-03-08  
**Version:** 2.2.0

> **See Also:** [DATAFLOW.md](DATAFLOW.md) – Stage-by-stage pipeline data flow documentation

---

## How DiaRemot locates models

- The pipeline promotes a primary model root from `DIAREMOT_MODEL_DIR`. When it is unset the bootstrapper prefers `/srv/models` on Linux or `D:/models` on Windows and gracefully falls back to `<repo>/.cache/models` if those locations are not writable.【F:src/diaremot/pipeline/runtime/environment.py†L34-L126】
- Every stage iterates across the ordered roots exposed by `iter_model_roots`: the promoted root, the canonical platform root, `<repo>/models`, the user's `~/models`, and the project cache directory.【F:src/diaremot/utils/model_paths.py†L18-L55】
- Component loaders probe a small list of case-insensitive relative paths under each root. The first on-disk match wins. Environment variables override specific assets when present.

---

## Complete Model Inventory

| # | Component | Default relative path | Key files | Quantisation | Primary source |
|---|-----------|----------------------|-----------|--------------|----------------|
| 1 | Silero VAD | `silero_vad.onnx` (root; alias `Diarization/silero_vad/`) | `silero_vad.onnx` | Float32 | TorchHub `snakers4/silero-vad` (ONNX export) |
| 2 | ECAPA Speaker Embeddings | `Diarization/ecapa-onnx/ecapa_tdnn.onnx` | `ecapa_tdnn.onnx` | Float32 | `speechbrain/spkrec-ecapa-voxceleb` |
| 3 | Speech Emotion (SER8) | `Affect/ser8/` | `model.int8.onnx` (+ tokenizer files) | INT8 | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` |
| 4 | Valence/Arousal/Dominance | `Affect/VAD_dim/` | `model.onnx` (+ tokenizer files) | Float32 | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` |
| 5 | Text Emotions (GoEmotions) | `text_emotions/` | `model.int8.onnx` (+ tokenizer files) | INT8 | `SamLowe/roberta-base-go_emotions` |
| 6 | Intent Classification | `intent/` or `bart/` | `model_int8.onnx` (optional) + `config.json` + tokenizer | INT8 / Torch | `facebook/bart-large-mnli` |
| 7 | Sound Event Detection (PANNs) | `Affect/sed_panns/` | `model.onnx`, `class_labels_indices.csv` | Float32 | `qiuqiangkong/panns_cnn14` |
| 8 | ASR (Faster-Whisper) | Auto-downloaded to cache | `model.bin`, `tokenizer.json`, `vocabulary.txt` | CTranslate2 | `Systran/faster-whisper-*` |

Each relative path above is resolved against every candidate model root until a match is found.

---

## Component Details

### 1. Silero VAD (Voice Activity Detection)

- **Purpose:** Segment audio into speech / non-speech regions.
- **Search order:** `silero_vad.onnx`, `silero/vad.onnx` under each model root (case-insensitive directories included).【F:src/diaremot/pipeline/diarization/vad.py†L41-L111】
- **Environment override:** `SILERO_VAD_ONNX_PATH` (file path).
- **Fallback:** If no ONNX file is found and networking is available, the loader falls back to the TorchHub package; otherwise an energy-based VAD is used.【F:src/diaremot/pipeline/diarization/vad.py†L18-L191】

### 2. ECAPA-TDNN Speaker Embeddings

- **Purpose:** Generate 192-D speaker embeddings for clustering.
- **Search order:** `ecapa_onnx/ecapa_tdnn.onnx`, `Diarization/ecapa-onnx/ecapa_tdnn.onnx`, then `ecapa_tdnn.onnx` at the root.【F:src/diaremot/pipeline/diarization/embeddings.py†L19-L41】
- **Environment override:** `ECAPA_ONNX_PATH` (file path).
- **Fallback:** None – the ONNX export is required.

### 3. Speech Emotion Recognition (SER8)

- **Purpose:** Predict eight categorical emotions from audio segments.
- **Search order:** `model.int8.onnx`, `ser8.int8.onnx`, `model.onnx`, `ser_8class.onnx` within the resolved directory.【F:src/diaremot/affect/emotion_analyzer.py†L184-L260】
- **Environment override:** `DIAREMOT_SER_ONNX` (file path).
- **Required artefacts:** `model.int8.onnx`, `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `vocab.json`, `special_tokens_map.json`.
- **Fallback:** None – quantised ONNX is expected.

### 4. Valence / Arousal / Dominance (Dimensional Affect)

- **Purpose:** Estimate continuous V/A/D scores for speech turns.
- **Search order:** `model.onnx`, then `vad_model.onnx` in the directory resolved for `Affect/VAD_dim/` (case-insensitive aliases allowed).【F:src/diaremot/affect/emotion_analyzer.py†L186-L213】
- **Environment override:** `AFFECT_VAD_DIM_MODEL_DIR` (directory).
- **Required artefacts:** `model.onnx`, `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `vocab.json`, `special_tokens_map.json`, `added_tokens.json`.
- **Fallback:** If missing, the pipeline emits neutral V/A/D values and records a warning.【F:src/diaremot/affect/emotion_analyzer.py†L1066-L1107】

### 5. Text Emotions (GoEmotions)

- **Purpose:** Classify transcript text into 28 GoEmotions categories.
- **Search order:** `model.int8.onnx`, `model.onnx`, `roberta-base-go_emotions.onnx` within the `text_emotions/` directory (case-insensitive aliases `goemotions-onnx/`, etc.).【F:src/diaremot/affect/emotion_analyzer.py†L178-L205】
- **Environment override:** `DIAREMOT_TEXT_EMO_MODEL_DIR` (directory).
- **Fallback:** When ONNX assets are missing and downloads are allowed, a Transformers pipeline is loaded from the Hugging Face Hub.【F:src/diaremot/affect/emotion_analyzer.py†L482-L564】

### 6. Intent Classification (Zero-shot NLI)

- **Purpose:** Infer utterance intent using zero-shot natural language inference.
- **Directory requirements:** `config.json` with a `model_type`, tokenizer assets (`tokenizer.json` or `vocab.json` + `merges.txt`), and optionally `model_int8.onnx` / `model.onnx`. Directories named `intent/`, `bart/`, or `bart-large-mnli/` under any model root are discovered automatically.【F:src/diaremot/affect/emotion_analyzer.py†L229-L405】
- **Environment override:** `DIAREMOT_INTENT_MODEL_DIR` (directory).
- **Fallback:** If no ONNX export is available the loader uses the Transformers pipeline (which requires allowing downloads).【F:src/diaremot/affect/emotion_analyzer.py†L1111-L1296】

### 7. Sound Event Detection (PANNs)

- **Purpose:** Tag global ambient sounds and optionally produce a per-frame timeline.
- **Default directory:** `Affect/sed_panns/` (aliases `panns/` or `panns_cnn14/`).【F:src/diaremot/affect/sed_panns.py†L18-L66】
- **Required artefacts:** `model.onnx` plus the class label CSV shipped with the ONNX export.
- **Environment override:** `DIAREMOT_PANNS_DIR` (directory).
- **Fallback:** When ONNX is missing but the optional `panns_inference` package is available, the pipeline falls back to the PyTorch implementation.【F:src/diaremot/affect/sed_panns.py†L86-L167】

### 8. Automatic Speech Recognition (Faster-Whisper)

- **Purpose:** Transcribe diarised speech segments.
- **Default model:** `faster-whisper-distil-large-v3` stored in the CTranslate2 format.
- **Discovery:** The runtime searches `distil-large-v3/`, `faster-whisper/distil-large-v3/`, `ct2/distil-large-v3/` under each model root, then falls back to `~/whisper_models/distil-large-v3`. If nothing is present, CTranslate2 assets are downloaded into `HF_HOME`/`huggingface_hub` on demand.【F:src/diaremot/pipeline/runtime/environment.py†L12-L118】
- **Override:** Set `WHISPER_MODEL_PATH` (directory) or pass `--whisper-model` / `--model-root` via the CLI.

---

## Environment Variable Summary

| Variable | Scope | Expected value |
|----------|-------|----------------|
| `DIAREMOT_MODEL_DIR` | Global | Primary model root directory |
| `SILERO_VAD_ONNX_PATH` | Silero VAD | Path to `silero_vad.onnx` |
| `ECAPA_ONNX_PATH` | ECAPA embeddings | Path to `ecapa_tdnn.onnx` |
| `DIAREMOT_SER_ONNX` | SER8 | Path to `model.int8.onnx` |
| `DIAREMOT_TEXT_EMO_MODEL_DIR` | Text emotions | Directory containing GoEmotions ONNX + tokenizer |
| `AFFECT_VAD_DIM_MODEL_DIR` | V/A/D | Directory containing `model.onnx` + tokenizer |
| `DIAREMOT_INTENT_MODEL_DIR` | Intent | Directory containing BART-MNLI assets |
| `DIAREMOT_PANNS_DIR` | PANNs SED | Directory containing PANNs ONNX + labels |

---

## Quick Verification Script

```python
from pathlib import Path
import os

models_to_check = {
    "Silero VAD": ("silero_vad.onnx", "Diarization/silero_vad/silero_vad.onnx"),
    "ECAPA Embeddings": "Diarization/ecapa-onnx/ecapa_tdnn.onnx",
    "SER8 (Speech Emotion)": "Affect/ser8/model.int8.onnx",
    "VAD Emotion": "Affect/VAD_dim/model.onnx",
    "Text Emotions (GoEmotions)": "text_emotions/model.int8.onnx",
    "Intent (BART-MNLI)": "intent/model_int8.onnx",
    "PANNs SED": "Affect/sed_panns/model.onnx",
}

model_root = Path(os.getenv("DIAREMOT_MODEL_DIR", Path.cwd() / "models")).expanduser()
print(f"Checking models in: {model_root}\n")

for name, rel_path in models_to_check.items():
    if isinstance(rel_path, (list, tuple)):
        candidates = [Path(p) for p in rel_path]
    else:
        candidates = [Path(rel_path)]
    found_path = None
    for candidate in candidates:
        full = model_root / candidate
        if full.exists():
            found_path = full
            break
    if found_path is None:
        found_path = model_root / candidates[0]
    status = "✓ FOUND" if found_path.exists() else "✗ MISSING"
    size = f"{found_path.stat().st_size / 1024 / 1024:.1f}MB" if found_path.exists() else "N/A"
    rel_display = " | ".join(str(p) for p in candidates)
    print(f"{status:10} {name:30} {size:>8}  {rel_display}")
```

---

## Key Takeaways

1. Use `DIAREMOT_MODEL_DIR` to switch between model bundles; the runtime searches sensible fallbacks automatically.【F:src/diaremot/pipeline/runtime/environment.py†L34-L118】【F:src/diaremot/utils/model_paths.py†L18-L55】
2. ONNX assets for SER8, GoEmotions, and BART-MNLI are quantised (`model.int8.onnx`). Keep the accompanying tokenizer metadata in the same directory for each component.【F:src/diaremot/affect/emotion_analyzer.py†L178-L260】
3. Faster-Whisper assets are managed by Hugging Face caches; only set `WHISPER_MODEL_PATH` when using a custom CTranslate2 export.【F:src/diaremot/pipeline/runtime/environment.py†L92-L118】
4. When a model is missing, stages either fall back to slower PyTorch implementations (Silero VAD, PANNs, intent/text emotions) or emit neutral defaults with warnings so the pipeline can still complete.【F:src/diaremot/pipeline/diarization/vad.py†L18-L191】【F:src/diaremot/affect/emotion_analyzer.py†L482-L1107】【F:src/diaremot/affect/sed_panns.py†L86-L212】

---

**Last Updated:** 2025-03-08  
**Author:** DiaRemot v2.2.0 code audit
