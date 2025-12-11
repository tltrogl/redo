# Gate DB Integration Plan (PreprocessConfig)

## Goal
Thread the `gate_db` setting from CLI/config into the preprocessing stack so noise‑gate behavior is configurable.

## Steps
1. **Config model**  
   - In `src/diaremot/pipeline/preprocess/config.py`, add a field to `PreprocessConfig`:  
     ```python
     gate_db: float | None = None  # default: disabled unless provided
     ```

2. **Factory wiring**  
   - In `src/diaremot/pipeline/core/component_factory.py`, when constructing `PreprocessConfig` inside `build_preprocessor`, pass the new value:  
     ```python
     pp_config = PreprocessConfig(
         target_sr=pipeline_config.target_sr,
         noise_reduction=pipeline_config.noise_reduction,
         denoise_alpha_db=pipeline_config.denoise_alpha_db,
         denoise_beta=pipeline_config.denoise_beta,
         gate_db=pipeline_config.gate_db,  # NEW
         ...
     )
     ```

3. **CLI / pipeline config**  
   - Ensure the CLI flag `--gate-db` already populates `pipeline_config.gate_db`. If not, add it to the pipeline config dataclass and CLI argument parser with a sensible default (e.g., `None`).

4. **Verify**
   - Run:
     ```bash
     python -m diaremot.cli run audio/Trog.M4A \
       --noise-reduction --denoise-alpha-db 15.0 --denoise-beta 0.02 \
       --gate-db -30.0 --clear-cache
     ```
   - Smoke test:
     ```bash
     python -m diaremot.cli smoke --outdir smoke_test_gate --gate-db -30.0
     ```

## Notes
- Keeping `gate_db` optional (`None`) preserves backward compatibility.
- No behavior change occurs unless the user supplies `--gate-db` or a config value.
