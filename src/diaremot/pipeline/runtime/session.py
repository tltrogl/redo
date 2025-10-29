"""Pipeline session primitives mirroring the paralinguistics package layout."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import PipelineConfig
from ..stages import PIPELINE_STAGES, PipelineState, StageDefinition

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from ..orchestrator import AudioAnalysisPipelineV2


@dataclass(slots=True, frozen=True)
class StagePlan:
    """Immutable wrapper around a stage definition."""

    definition: StageDefinition

    @property
    def name(self) -> str:
        return self.definition.name

    def run(self, pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard) -> None:
        self.definition.runner(pipeline, state, guard)


@dataclass(slots=True)
class PipelineSession:
    """Container for per-run pipeline state and immutable configuration."""

    pipeline: AudioAnalysisPipelineV2
    config: PipelineConfig
    state: PipelineState
    stages: tuple[StagePlan, ...]
    input_audio_path: Path
    output_dir: Path

    @classmethod
    def create(
        cls,
        pipeline: AudioAnalysisPipelineV2,
        state: PipelineState,
        *,
        input_audio_path: str | Path,
        output_dir: Path,
        stage_definitions: Iterable[StageDefinition] | None = None,
    ) -> PipelineSession:
        plans = build_stage_plan(stage_definitions or PIPELINE_STAGES)
        return cls(
            pipeline=pipeline,
            config=pipeline.pipeline_config,
            state=state,
            stages=plans,
            input_audio_path=Path(input_audio_path),
            output_dir=Path(output_dir),
        )

    def as_tuple(self) -> tuple[StagePlan, ...]:
        return self.stages


def build_stage_plan(definitions: Iterable[StageDefinition]) -> tuple[StagePlan, ...]:
    """Return an ordered tuple of ``StagePlan`` entries from raw definitions."""

    plans: list[StagePlan] = [StagePlan(defn) for defn in definitions]
    return tuple(plans)


__all__ = ["PipelineSession", "StagePlan", "build_stage_plan"]
