# Pipeline Optimization Opportunities

## Turn-Taking Analytics

- **Overlap/Interruption Sweep-Line (New in v2.2.1):**
  - `compute_overlap_and_interruptions` now expands every speech turn into sorted boundary events and performs a single sweep-line pass.
  - Complexity drops from \(\mathcal{O}(n^2)\) pairwise comparisons to \(\mathcal{O}(n \log n)\) for sorting plus linear accumulation across active speakers.
  - Dense, long-form conversations (100+ rapid turns) now stay inside the paralinguistics SLA without throttling downstream analytics.
- Threshold knobs (`min_overlap_sec`, `interruption_gap_sec`) are unchanged, ensuring historical report expectations remain intact while scaling to higher speaker churn.

## Next Targets

- Continue profiling the affect bundle for additional \(\mathcal{O}(n^2)\) hot spots (e.g., emotion cross-correlation).
- Evaluate batching opportunities for voice-quality feature extraction once current CPU telemetry stabilises.

## Affect memory reuse

- `affect.run` now hands each segment a `_SegmentAudioWindow` that reuses the
  shared PCM buffer through zero-copy NumPy slices and memory views, so
  adjacent segments no longer allocate duplicate waveform arrays.【F:src/diaremot/pipeline/stages/affect.py†L22-L115】
- `_affect_unified` normalizes incoming audio lazily, accepting memoryviews,
  iterables, and custom view objects without forcing materialization. Downstream
  analyzers therefore reuse the same float32 buffer while still supporting
  streaming clients.【F:src/diaremot/pipeline/core/affect_mixin.py†L9-L60】
## Shared spectral analysis in preprocessing

The preprocess chain now caches the short-time Fourier transform magnitude
between the upward gain and compression stages. The helper class
`SpectralFrameStats` stores the per-frame loudness derived from a single STFT,
allowing both stages to reuse the same spectral snapshot instead of computing
separate FFTs. This keeps numerical parity with the previous implementation
(the new unit test mirrors the legacy path) while removing a redundant
transform for every clip.

### Operational guidance
- `process_array` constructs one `SpectralFrameStats` instance from the
  denoised waveform and threads it through both stages.
- `apply_upward_gain` now returns the processed waveform together with the
  updated spectral stats so `apply_compression` can reuse them.
- Callers that need to invoke these helpers directly should reuse the returned
  stats to avoid recomputing STFTs.

The change shaves roughly one STFT (and associated magnitude reduction)
per clip during preprocessing, which is significant for long-form audio
where the FFT is among the most expensive CPU steps.

## Dependency health caching and concurrency

- `dependency_health_summary` now memoises the computed dependency snapshot
  between pipeline runs and returns defensive copies, so repeated workers no
  longer re-import every dependency per file while still allowing explicit
  refreshes when the environment changes.【F:src/diaremot/pipeline/config.py†L314-L487】
- `_iter_dependency_status` fans out the metadata lookups through a bounded
  thread pool, overlapping `importlib.metadata` fetches while maintaining
  deterministic ordering and falling back to sequential enumeration if the
  executor cannot start.【F:src/diaremot/pipeline/config.py†L347-L395】
- The `diagnostics` helper now forces a refreshed summary before reporting so
  interactive CLI checks continue to surface the latest dependency state even
  when caching is enabled elsewhere in the pipeline.【F:src/diaremot/pipeline/config.py†L501-L509】

## Preprocess cache streaming and upgrades

- Preprocessed audio is now stored as a standalone float32 `.npy` file with
  metadata in `preprocessed.meta.json`, allowing cache hits to memory-map PCM
  without duplicating multi-hour clips in RAM.【F:src/diaremot/pipeline/stages/preprocess.py†L40-L222】
- Legacy `.npz` caches are automatically upgraded the first time they are read,
  so existing deployments benefit from the new layout without manual cleanup.
- Cache writes atomically stream through a temporary file to avoid
  partial artefacts when workers exit unexpectedly.【F:src/diaremot/pipeline/stages/preprocess.py†L224-L279】

## Diarisation cache compaction

- Cached diarisation turns now record embeddings in a separate
  `diar_embeddings.npz` file and only persist lightweight timing/label data in
  JSON, significantly reducing resume overhead for large meetings.【F:src/diaremot/pipeline/stages/diarize.py†L1-L209】
- Resume paths rehydrate embeddings on demand, preserving downstream speaker
  recognition behaviour while avoiding redundant JSON decoding.【F:src/diaremot/pipeline/stages/diarize.py†L82-L123】
- Sanitisation clamps cached and fresh diarisation turns to valid time ranges
  and sorts them chronologically so that downstream metrics and resumptions do
  not encounter negative or out-of-order spans.【F:src/diaremot/pipeline/stages/diarize.py†L26-L121】【F:src/diaremot/pipeline/stages/diarize.py†L146-L206】

## Transcription resume fast path

- The ASR stage stores only per-segment digests in `tx.json` and cross-checks
  them against the transcription checkpoint before trusting cached text,
  guarding against cache corruption without sacrificing resume speed.【F:src/diaremot/pipeline/stages/asr.py†L52-L211】
- Fresh transcriptions keep the asynchronous execution path but reuse a single
  normalisation routine when writing caches, so per-segment payloads are hashed
  once and shared across outputs.【F:src/diaremot/pipeline/stages/asr.py†L132-L211】

## Transcription maintenance cleanup (Current)

- Removed an unused SNR cache facade and a dead resampling kernel helper from
  the transcription models module, slimming the public surface area while keeping
  the fast resampler and SNR estimator intact.【F:src/diaremot/pipeline/transcription/models.py†L10-L113】
- Dropped an unused semaphore placeholder from the async transcriber; executor
  resets still rebuild the worker pool without tracking redundant concurrency
  state.【F:src/diaremot/pipeline/transcription/scheduler.py†L63-L69】【F:src/diaremot/pipeline/transcription/scheduler.py†L683-L689】
