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
