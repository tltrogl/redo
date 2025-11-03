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
