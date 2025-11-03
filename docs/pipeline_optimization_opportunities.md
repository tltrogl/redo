# Pipeline Optimization Opportunities

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
