from __future__ import annotations

import logging

logger = logging.getLogger("diaremot.pipeline.diarization")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

__all__ = ["logger"]
