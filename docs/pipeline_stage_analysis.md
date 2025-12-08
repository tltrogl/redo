# DiaRemot Pipeline Stage Analysis

This document provides a stage-by-stage breakdown of the DiaRemot audio analytics pipeline. The registry in `src/diaremot/pipeline/stages/__init__.py` defines the canonical execution order; each entry is analysed below with emphasis on responsibilities, rationale, inputs/outputs, caching, and failure handling. The final section highlights redundancies and inconsistencies observed across stages.

## Stage Order Overview

| Order | Stage Key | Runner |
| --- | --- | --- |
| 1 | `dependency_check` | `dependency_check.run`
| 2 | `preprocess` | `preprocess.run_preprocess`
| 3 | `background_sed` | `preprocess.run_background_sed`
| 4 | `diarize` | `diarize.run`
| 5 | `affect_audio` | `affect.run_audio_affect`
| 6 | `transcribe` | `asr.run`
| 7 | `paralinguistics` | `paralinguistics.run`
| 8 | `affect_and_assemble` | `affect.run`
| 9 | `overlap_interruptions` | `summaries.run_overlap`
| 10 | `conversation_analysis` | `summaries.run_conversation`
| 11 | `speaker_rollups` | `summaries.run_speaker_rollups`
| 12 | `outputs` | `summaries.run_outputs`

Source: `PIPELINE_STAGES` registry.【F:src/diaremot/pipeline/stages/__init__.py†L1-L30】

## 1. Dependency Check

- **What it does:** Fetches the dependency health summary and records whether any optional components are unhealthy.【F:src/diaremot/pipeline/stages/dependency_check.py†L3-L48】
- **Why:** Gives early warning about missing optional packages without hard-failing long inference runs; honours `validate_dependencies` to avoid redundant checks during production sweeps.【F:src/diaremot/pipeline/stages/dependency_check.py†L17-L25】
- **How it is used:** Populates `pipeline.stats.config_snapshot` and `state.dependency_summary`, and logs warnings that surface in QC reports. Downstream stages read the snapshot to decide whether to skip work (e.g., paralinguistics stage checks for upstream failures).【F:src/diaremot/pipeline/stages/dependency_check.py†L27-L48】

## 2. Preprocess

- **What it does:** Loads or generates the preprocessed waveform, computes signatures, and primes resume caches for diarization and ASR.【F:src/diaremot/pipeline/stages/preprocess.py†L31-L120】
- **Why:** Normalising audio up front enables consistent downstream sampling and underpins deterministic caching keyed by audio hash and preprocessing configuration.【F:src/diaremot/pipeline/stages/preprocess.py†L37-L120】
- **How it is used:**
  - Seeds `state.y`, `state.sr`, `state.health`, and `state.duration_s` for later stages.【F:src/diaremot/pipeline/stages/preprocess.py†L53-L76】
  - Persists checkpoints and cache artefacts (`preprocessed_audio.npy`, `preprocessed.meta.json`) that allow future runs to short-circuit expensive work.【F:src/diaremot/pipeline/stages/preprocess.py†L83-L119】【F:src/diaremot/pipeline/stages/preprocess.py†L203-L326】
  - Hydrates `state.diar_cache` and `state.tx_cache` so diarization and transcription can resume or be skipped if digests match.【F:src/diaremot/pipeline/stages/preprocess.py†L122-L200】

## 3. Background Sound Event Detection (SED)

- **What it does:** Computes or reloads background sound tags, ambient noise scores, and optional SED timeline artefacts, caching the results per audio hash and configuration signature.【F:src/diaremot/pipeline/stages/preprocess.py†L422-L631】
- **Why:** Ambient context is reused across affect analysis and reporting; caching avoids re-running ONNX SED passes for identical inputs.【F:src/diaremot/pipeline/stages/preprocess.py†L439-L490】【F:src/diaremot/pipeline/stages/preprocess.py†L502-L631】
- **How it is used:** Stores the structured `sed_info` dictionary on the pipeline state and in the stats snapshot, enabling affect assembly to inject noise tags, timeline overlaps, and derived SNR estimates into per-segment records.【F:src/diaremot/pipeline/stages/preprocess.py†L480-L631】

## 4. Diarize

- **What it does:** Either reconstructs speaker turns from caches or runs the diarizer to produce turn intervals, embeddings, and VAD diagnostics; results are checkpointed for resumption.【F:src/diaremot/pipeline/stages/diarize.py†L144-L275】
- **Why:** Accurate turn segmentation underpins ASR batching and per-speaker analytics; cache-aware resumption prevents redundant diarization when transcripts already exist.【F:src/diaremot/pipeline/stages/diarize.py†L150-L233】
- **How it is used:**
  - Sets `state.turns` and `state.vad_unstable`, which later inform ASR scheduling and affect metadata (e.g., `vad_unstable` flag on segments).【F:src/diaremot/pipeline/stages/diarize.py†L220-L274】
  - Persists `diar.json` and optional embedding matrices so future runs can skip heavy diarization when hashes match.【F:src/diaremot/pipeline/stages/diarize.py†L235-L264】

## 5. Audio Affect (pre-ASR)

- **What it does:** Streams an audio-only affect pass right after diarization, computing VAD-derived valence/arousal/dominance, SER8 emotions, and SED-overlap/noise metadata for every turn while persisting provisional CSV/timeline outputs.【F:src/diaremot/pipeline/stages/affect.py†L240-L394】
- **Why:** Preserves affect context for long recordings even if ASR or later enrichment halts, and avoids recomputing audio affect during final assembly.【F:src/diaremot/pipeline/stages/affect.py†L240-L394】
- **How it is used:** Stores interim rows on `state.audio_affect` and stamps them onto diarization turns so transcription and final affect can reattach the pre-ASR payloads.【F:src/diaremot/pipeline/stages/affect.py†L267-L394】

## 6. Transcribe

- **What it does:** Transcribes each diarization turn (sync or async), preferring cached digests when available, and merges matching audio-affect rows into normalized segments for downstream reuse; provisional outputs include those audio hints.【F:src/diaremot/pipeline/stages/asr.py†L165-L457】
- **Why:** Aligning ASR segments with diarization ensures consistent speaker-labelled transcripts while cache comparisons prevent stale reuse and carry forward early affect enrichment.【F:src/diaremot/pipeline/stages/asr.py†L185-L350】
- **How it is used:**
  - Populates `state.norm_tx`, stores ASR checkpoints, writes `tx.json`, and streams early CSV/timeline/readable artifacts that already include audio-affect payloads.【F:src/diaremot/pipeline/stages/asr.py†L308-L457】
  - Records failure fallbacks by emitting placeholder segments if ASR errors occur, allowing the pipeline to continue while flagging degraded accuracy.【F:src/diaremot/pipeline/stages/asr.py†L360-L414】

## 7. Paralinguistics

- **What it does:** Invokes the configured paralinguistics module to extract prosody, pause, and voice-quality metrics per transcript segment, unless upstream failures were recorded.【F:src/diaremot/pipeline/stages/paralinguistics.py†L18-L47】
- **Why:** These metrics enrich downstream affect analysis (WPM, pauses, jitter, etc.) and guard against wasted work when transcription or preprocessing failed.【F:src/diaremot/pipeline/stages/paralinguistics.py†L22-L34】
- **How it is used:** Stores the resulting metrics map on `state.para_metrics` for the affect stage to merge into final segment rows.【F:src/diaremot/pipeline/stages/paralinguistics.py†L36-L47】

## 8. Affect and Assembly

- **What it does:** Reuses audio-only affect rows where available, runs text emotion and intent, intersects SED context, and produces the full 53-column output schema described in `SEGMENT_COLUMNS`.【F:src/diaremot/pipeline/stages/affect.py†L404-L760】【F:src/diaremot/pipeline/outputs.py†L14-L68】
- **Why:** Consolidates multimodal analysis so later stages can emit CSVs and summaries without duplicating expensive inference or data munging.【F:src/diaremot/pipeline/stages/affect.py†L636-L757】
- **How it is used:**
  - Writes enriched segment dictionaries to `state.segments_final`, tagging each with affect payloads, noise scores, SED overlaps, and paralinguistic statistics.【F:src/diaremot/pipeline/stages/affect.py†L636-L757】
  - Propagates SED overlaps back to `state.turns` (events, estimated SNR) so summary stages can reference environmental context.【F:src/diaremot/pipeline/stages/affect.py†L758-L771】

## 9. Overlap and Interruptions

- **What it does:** Queries the paralinguistics module for overlap/interrupt statistics, normalises per-speaker counts, and records aggregate overlap ratios.【F:src/diaremot/pipeline/stages/summaries.py†L26-L87】
- **Why:** Provides structured interruption metrics for inclusion in speaker rollups and QC reports, while tolerating missing optional modules by warning instead of failing.【F:src/diaremot/pipeline/stages/summaries.py†L30-L86】
- **How it is used:** Saves aggregate and per-speaker overlap data on the pipeline state for downstream conversation analysis and rollups.【F:src/diaremot/pipeline/stages/summaries.py†L85-L87】

## 10. Conversation Analysis

- **What it does:** Runs higher-level discourse analytics (turn-taking balance, pace, coherence) on the final segment list and total duration, with graceful fallbacks to neutral metrics on error.【F:src/diaremot/pipeline/stages/summaries.py†L90-L123】
- **Why:** Supplies the conversational summary metrics needed for reports and HTML summaries without derailing the pipeline when heuristics break.【F:src/diaremot/pipeline/stages/summaries.py†L94-L123】
- **How it is used:** Stores `ConversationMetrics` in `state.conv_metrics`, which the outputs and summary generators consume.【F:src/diaremot/pipeline/stages/summaries.py†L94-L123】

## 11. Speaker Rollups

- **What it does:** Builds per-speaker aggregates (talk time, affects, interruptions) using the enriched segments and overlap stats, with fallbacks to empty lists when builders fail.【F:src/diaremot/pipeline/stages/summaries.py†L126-L145】
- **Why:** Provides the structured data for `speakers_summary.csv` and summary dashboards.【F:src/diaremot/pipeline/stages/summaries.py†L130-L145】
- **How it is used:** Saves the rollup list on `state.speakers_summary` for the outputs stage to write to disk and for report generators to render.【F:src/diaremot/pipeline/stages/summaries.py†L130-L145】

## 12. Outputs

- **What it does:** Delegates to the pipeline’s output writer to emit CSV/JSON/HTML artefacts, and marks the cache directory as complete by touching a `.done` file.【F:src/diaremot/pipeline/stages/summaries.py†L148-L169】
- **Why:** Centralises final artefact generation and cache completion so reruns can quickly detect finished work.【F:src/diaremot/pipeline/stages/summaries.py†L148-L169】
- **How it is used:** Consumes all prior state (segments, rollups, health metrics, SED info) to produce the user-facing deliverables and to signal cache completion for resumable workflows.【F:src/diaremot/pipeline/stages/summaries.py†L148-L169】

## Cross-Stage Harmonization (2025-03)

- **Shared cache metadata helpers:** Preprocess, diarization, ASR, and background SED now funnel cache reads/writes through `matches_pipeline_cache`/`build_cache_payload`, ensuring every stage validates the same trio of identifiers (version, audio hash, preprocessing signature) before reusing artefacts.【F:src/diaremot/pipeline/stages/utils.py†L49-L95】【F:src/diaremot/pipeline/stages/preprocess.py†L118-L213】【F:src/diaremot/pipeline/stages/diarize.py†L238-L267】【F:src/diaremot/pipeline/stages/asr.py†L274-L299】
- **Paralinguistics normalization upstream:** The paralinguistics stage back-fills duration, word-count, pause, and WPM metrics for every transcript segment so the affect combiner no longer recomputes these basics on the hot path.【F:src/diaremot/pipeline/stages/paralinguistics.py†L18-L94】【F:src/diaremot/pipeline/stages/affect.py†L636-L757】
- **SED timeline hydration:** Background SED retains in-memory `timeline_events` alongside the cached JSON pointer, hydrating legacy caches when necessary so affect assembly can intersect events without extra disk I/O.【F:src/diaremot/pipeline/stages/preprocess.py†L360-L552】【F:src/diaremot/pipeline/stages/affect.py†L440-L520】
- **Overlap availability signal:** The overlap/interruptions stage flags whether data was actually computed, propagating `overlap_available` into downstream summaries and CSVs to differentiate “not computed” from genuine zero overlap.【F:src/diaremot/pipeline/stages/base.py†L30-L45】【F:src/diaremot/pipeline/stages/summaries.py†L26-L100】【F:src/diaremot/pipeline/outputs.py†L446-L485】【F:src/diaremot/summaries/speakers_summary_builder.py†L372-L477】

