"""Stage execution helpers following the paralinguistics runtime style."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import StageExecutionError, coerce_stage_error
from ..logging_utils import StageGuard
from .session import PipelineSession, StagePlan

if TYPE_CHECKING:  # pragma: no cover - avoid runtime import cycle
    from ..orchestrator import AudioAnalysisPipelineV2


@dataclass(slots=True)
class StageExecutor:
    """Execute an ordered collection of pipeline stages."""

    pipeline: AudioAnalysisPipelineV2
    session: PipelineSession

    def run_all(self) -> None:
        for plan in self.session.stages:
            self._run_plan(plan)

    def _run_plan(self, plan: StagePlan) -> None:
        state = self.session.state
        with StageGuard(self.pipeline.corelog, self.pipeline.stats, plan.name) as guard:
            try:
                plan.run(self.pipeline, state, guard)
            except StageExecutionError:
                raise
            except Exception as exc:  # pragma: no cover - defensive wrapper
                context = {
                    "stage": plan.name,
                    "file_id": self.pipeline.stats.file_id,
                    "has_audio": state.y is not None,
                    "has_turns": bool(getattr(state, "turns", None)),
                    "has_transcript": bool(getattr(state, "norm_tx", None)),
                    "audio_sha16": getattr(state, "audio_sha16", None),
                }
                raise coerce_stage_error(
                    plan.name,
                    f"Stage '{plan.name}' execution failed",
                    context=context,
                    cause=exc,
                ) from exc


def execute_plans(
    pipeline: AudioAnalysisPipelineV2,
    plans: Iterable[StagePlan],
    session: PipelineSession,
) -> None:
    """Convenience helper mirroring paralinguistics' batch execution helpers."""

    session.stages = tuple(plans)
    executor = StageExecutor(pipeline=pipeline, session=session)
    executor.run_all()


__all__ = ["StageExecutor", "execute_plans"]
