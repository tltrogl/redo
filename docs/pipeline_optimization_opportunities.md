# Pipeline Optimization Opportunities

The notes below walk the 11 pipeline stages and highlight the most significant
performance or resilience opportunities I noticed while reviewing the current
implementation. Suggested improvements intentionally focus on changes that keep
functional behaviour stable while reducing runtime, memory pressure, or avoidable
I/O overhead.

## 1. `dependency_check`
- The stage always recomputes `dependency_health_summary()` when enabled, which
  imports every core dependency and queries version metadata sequentially via
  `_iter_dependency_status` on every run.【F:src/diaremot/pipeline/stages/dependency_check.py†L17-L48】【F:src/diaremot/pipeline/config.py†L338-L420】
  Persisting the computed summary on the `PipelineSession` (or in the
  process-wide environment bootstrap) would avoid repeated metadata scans when
  multiple files are processed in one worker.
- `_iter_dependency_status` performs blocking import + metadata lookups one by
  one; wrapping those queries in a `ThreadPoolExecutor` (bounded) would let
  slow `importlib_metadata` calls overlap, and the fallback map could still be
  reduced deterministically before logging.【F:src/diaremot/pipeline/config.py†L338-L358】

## 2. `preprocess`
- Cache loads currently rely on `np.load(..., allow_pickle=True)` followed by an
  unconditional `astype(np.float32)` copy when the cache hits.【F:src/diaremot/pipeline/stages/preprocess.py†L47-L80】 Switching the cache
  format to `np.load(..., mmap_mode="r")` or storing PCM frames in a float32
  memory-map (e.g., `np.memmap` or Zarr) would let multi-hour runs stream data
  without duplicating arrays in RAM.
- When the cache misses, `preprocess.process_file` always decodes the entire
  clip into memory before chunking, even for long inputs; deferring to
  `safe_load_audio` only per-chunk (and streaming PCM out of ffmpeg) would avoid
  the first full decode for files that will be re-chunked anyway.【F:src/diaremot/pipeline/audio_preprocessing.py†L27-L84】【F:src/diaremot/pipeline/preprocess/io.py†L52-L114】
- The signal chain re-computes STFT magnitudes twice—once for the upward gain and
  once for compression—over the same hop size.【F:src/diaremot/pipeline/preprocess/chain.py†L83-L133】 Sharing the initial
  `librosa.stft` result (or migrating to `scipy.signal.stft` with Reuse) would
  eliminate one expensive transform per stage.
- Background cache probing scans every configured cache root synchronously for
  both diar and transcription artefacts.【F:src/diaremot/pipeline/stages/preprocess.py†L172-L196】 Reordering the checks to stop as soon as both
  artefacts are found (or precomputing available roots at startup) reduces slow
  network/disk traversals when shared caches point to remote storage.

## 3. `background_sed`
- Cache validation re-materialises full timeline payloads even when only the
  `timeline_event_count` is required for stats.【F:src/diaremot/pipeline/stages/preprocess.py†L201-L248】 Persisting the count separately would
  avoid deserialising large event lists for cache hits.
- Timeline inference processes windows sequentially and always allocates numpy
  arrays for the entire audio clip, even when `sed_mode="auto"` disables the
  timeline for quiet files.【F:src/diaremot/pipeline/stages/preprocess.py†L243-L268】 A cheap pre-pass on the cached noise score could
  skip allocating buffers when the threshold is not met.
- The SED call path always copies the entire waveform into memory; exposing a
  chunked iterator on the `tagger` would let us stream long files without the
  double buffering that `state.y` already imposes.【F:src/diaremot/pipeline/stages/preprocess.py†L229-L267】

## 4. `diarize`
- Resume logic rehydrates the full cached JSON (potentially containing embeddings)
  before deciding whether diarisation is required.【F:src/diaremot/pipeline/stages/diarize.py†L70-L123】 Storing embeddings in a separate
  file or only materialising turn timing/labels in the primary cache would
  reduce JSON load overhead.
- Fresh diarisation always materialises every turn before any downstream stage
  runs; exposing a generator or incremental callback from `diar.diarize_audio`
  would let the ASR scheduler begin processing earlier, overlapping expensive
  inference stages.【F:src/diaremot/pipeline/stages/diarize.py†L124-L173】
- The `vad_unstable` heuristic scales with `duration_s` but still counts flips
  in Python; moving that logic into the diariser (where VAD frames already live)
  would save a second full pass across the turn list.【F:src/diaremot/pipeline/stages/diarize.py†L145-L152】

## 5. `transcribe`
- The stage rebuilds the diarised turn list into `tx_in` dictionaries on every
  call even when resume caches will immediately short-circuit; capturing the
  hashed turn schema in the diarisation checkpoint would avoid the redundant
  conversions.【F:src/diaremot/pipeline/stages/asr.py†L37-L59】
- `transcribe_segments` is invoked serially even though the `Transcriber`
  façade already exposes an async engine with configurable worker pools.【F:src/diaremot/pipeline/stages/asr.py†L60-L79】【F:src/diaremot/pipeline/transcription_module.py†L32-L80】 Wiring the pipeline config to opt into
  `enable_async=True` (and raising `max_workers` when CPU allows) would overlap
  decoder batches and reduce wall-clock time for dense diarisation outputs.
- Cache writes duplicate the full normalised segment list as JSON; storing
  per-segment digests alongside the diar cache would let us detect unchanged
  audio without serialising the entire payload repeatedly.【F:src/diaremot/pipeline/stages/asr.py†L129-L146】

## 6. `paralinguistics`
- The stage always copies the full waveform into a float32 numpy array before
  delegating to `_extract_paraling`, even if transcription failed or the
  downstream module only needs short clips.【F:src/diaremot/pipeline/stages/paralinguistics.py†L13-L23】 Short-circuiting early when
  `state.norm_tx` is empty (or when the module advertises streaming support)
  would avoid the duplicate allocation.
- `_extract_paraling` currently receives the entire transcript list; exposing a
  batched iterator or chunk-level processing API would reduce peak memory when
  analysing multi-hour meetings.

## 7. `affect_and_assemble`
- For every segment, the stage slices `state.y` to build a fresh numpy array for
  audio affect despite SER/VAD models usually working on mel batches; caching
  resampled windows or reusing the paralinguistics clips would reduce repeated
  slicing allocations.【F:src/diaremot/pipeline/stages/affect.py†L67-L143】
- Timeline intersections rely on Python bisect loops across every event, which
  can become O(N·log M) per segment when dense SED timelines are enabled.【F:src/diaremot/pipeline/stages/affect.py†L145-L192】 Materialising the timeline
  as numpy arrays (start/end/confidence) would allow vectorised overlap checks.
- Intent/text emotion JSON is re-encoded for every row; caching the raw dict and
  only serialising once during output generation would trim CPU time on large
  transcripts.【F:src/diaremot/pipeline/stages/affect.py†L109-L168】

## 8. `overlap_interruptions`
- `compute_overlap_and_interruptions` runs a nested sweep across every turn pair
  (`for k in range(j, i)`), making the stage quadratic in the number of turns.【F:src/diaremot/affect/paralinguistics/analysis.py†L159-L212】 Switching to a proper sweep-line over sorted boundaries (maintaining
  the active speaker set) would bring the computation down to O(N log N).
- The Python loops also allocate intermediary dicts for every overlap; pushing
  aggregation into numpy arrays or `bisect`-based counters would reduce GC
  pressure in long conversations.【F:src/diaremot/affect/paralinguistics/analysis.py†L159-L212】

## 9. `conversation_analysis`
- The function repeatedly iterates Python dictionaries and lists to build
  speaker metrics, then recomputes derived stats (entropy, turns per minute) in
  pure Python.【F:src/diaremot/summaries/conversation_analysis.py†L31-L180】 Vectorising the turn table with pandas/NumPy or precomputing per-speaker totals
  once would shrink CPU time for transcript-heavy meetings.
- Topic coherence and energy flow both rescan the segment list; caching tokenised
  text or precomputing per-window aggregates during affect assembly would let
  this stage reuse existing artefacts.【F:src/diaremot/summaries/conversation_analysis.py†L187-L240】

## 10. `speaker_rollups`
- `build_speakers_summary` parses JSON strings (`text_emotions_*`) and updates
  per-speaker dictionaries for every segment, which scales poorly with large
  meetings.【F:src/diaremot/summaries/speakers_summary_builder.py†L50-L143】 Storing affect outputs as structured dicts (or caching parsed results from the
  affect stage) would avoid per-segment `json.loads` calls.
- Aggregations compute medians/means using Python lists collected per speaker;
  using `numpy` accumulators (running sums, counts, sum of squares) would avoid
  storing entire per-segment histories in memory.【F:src/diaremot/summaries/speakers_summary_builder.py†L91-L143】

## 11. `outputs`
- `_write_outputs` eagerly writes every artefact regardless of configuration,
  including the HTML/PDF renderers that may be expensive; threading a per-output
  enable switch (or a lazy manifest) would let batch workflows skip formats they
  do not need.【F:src/diaremot/pipeline/core/output_mixin.py†L18-L83】
- The stage serialises the full segment list multiple times—CSV, JSONL, human
  transcript—each re-iterating over Python dicts.【F:src/diaremot/pipeline/core/output_mixin.py†L18-L83】 Writing once into an Arrow table (or Pandas
  DataFrame) and exporting to the required formats would reduce repeated Python
  traversal and give columnar compression for long transcripts.

## Cross-cutting suggestions
- Several stages (preprocess, diarize, transcribe) update checkpoints with large
  blobs of JSON or numpy data; switching the checkpoint backend to LMDB or
  sqlite would reduce filesystem churn and enable transactional updates.
- Many stages duplicate per-run logging to JSONL via `corelog`; batching log
  writes (or using structured logging with async sinks) would reduce the IO
  cost in multi-run services.
