# Sample2 Pipeline Run — 2025-11-08

This log records the follow-up diarization tuning requested for `data/sample2.mp3`. The goal was to obtain a **single-speaker** segmentation _without_ relying on the `--speaker-limit 1` override. All runs were executed from the project root with the prepared Python 3.11 virtual environment and the extracted model bundle described in `AGENTS.md`.

## Baseline run (default clustering threshold)
- **Command:** `python -m diaremot.cli run --input data/sample2.mp3 --outdir outputs_sample2_default`
- The diarizer merged 47 ECAPA windows into four clusters but the runtime heuristic collapsed them into a single dominant speaker: `Forcing diarization clusters into single speaker 'Speaker_38' (silhouette=0.249, dominance=0.75, clusters=4)`.
- Output summary (`speakers_summary.csv`) contained a single `Speaker_38` row, indicating the heuristic override succeeded but did not improve the underlying clustering behaviour.

## Attempt 2 — raise `--ahc-distance-threshold` to 0.35
- **Command:** `python -m diaremot.cli run --input data/sample2.mp3 --outdir outputs_sample2_tuned --ahc-distance-threshold 0.35 --clear-cache`
- Result: `Agglomerative clustering assigned 11 clusters ...` and the stage report estimated **4 speakers across 14 turns**.
- `outputs_sample2_tuned/speakers_summary.csv` contained four distinct speaker rows (`Speaker_5`, `Speaker_4`, `Speaker_6`, `Speaker_8`), confirming the diarizer now exposed the multi-speaker segmentation instead of forcing a single speaker.

## Attempt 3 — distance threshold 0.55 (aborted)
- **Command:** `python -m diaremot.cli run --input data/sample2.mp3 --outdir outputs_sample2_ahc055 --ahc-distance-threshold 0.55 --clear-cache`
- The diarizer log still reported **3 speakers across 13 turns** even after increasing the threshold, so the run was cancelled once that intermediate result was known.

## Final run — distance threshold 0.9
- **Command:** `python -m diaremot.cli run --input data/sample2.mp3 --outdir outputs_sample2_ahc090 --ahc-distance-threshold 0.9 --clear-cache`
- Diarization now reported `Agglomerative clustering assigned 1 clusters ...` and the stage summary showed **1 speaker across 1 turn** without any heuristic forcing.
- `outputs_sample2_ahc090/speakers_summary.csv` contained a single `Speaker_1` row with 35.25 s of speech.
- The generated `qc_report.json` recorded `speakers_est: 1`, zero warnings, and preserved the configured threshold (`"ahc_distance_threshold": 0.9`).

## Recommendation
Use `--ahc-distance-threshold 0.9` when processing `sample2.mp3` (or similar single-speaker recordings) to obtain a natural single-speaker diarization without enabling the 1-speaker limit. The full command is:

```bash
python -m diaremot.cli run \
  --input data/sample2.mp3 \
  --outdir <desired_outdir> \
  --ahc-distance-threshold 0.9 \
  --clear-cache
```

The intermediate runs demonstrate how progressively relaxing the agglomerative distance threshold exposes the multi-speaker splits and that a value near 0.9 is necessary for this clip to converge to one speaker organically.
