"""FastAPI wrapper for the DiaRemot pipeline when running on Cloud Run."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

try:
    from diaremot.pipeline.run_pipeline import build_pipeline_config, run_pipeline
except ImportError:  # pragma: no cover
    build_pipeline_config = None  # type: ignore[assignment]
    run_pipeline = None  # type: ignore[assignment]

logger = logging.getLogger("diaremot.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="DiaRemot Speech Intelligence API", version="2.2.0")


def _pipeline_available() -> bool:
    return callable(build_pipeline_config) and callable(run_pipeline)


@app.get("/")
def root() -> Dict[str, Any]:
    """Basic service metadata."""
    return {
        "service": "diaremot2-on",
        "version": "2.2.0",
        "status": "healthy" if _pipeline_available() else "degraded",
        "endpoints": {
            "process": "POST /process",
            "health": "GET /health",
        },
    }


@app.get("/health")
def health() -> JSONResponse:
    """Detailed health check."""
    if not _pipeline_available():
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "diaremot pipeline not available"},
        )

    model_dir = Path(os.getenv("DIAREMOT_MODEL_DIR", "/srv/models"))
    models_present = model_dir.exists() and any(model_dir.iterdir())
    return JSONResponse(
        {
            "status": "healthy" if models_present else "degraded",
            "models_available": models_present,
            "model_dir": str(model_dir),
        }
    )


def _save_upload(upload: UploadFile, target_path: Path) -> None:
    with target_path.open("wb") as target:
        shutil.copyfileobj(upload.file, target)


def _build_config(out_dir: Path) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "log_dir": str(out_dir / "logs"),
        "cache_root": str(out_dir / ".cache"),
        "ignore_tx_cache": True,
        "quiet": True,
        "noise_reduction": True,
    }
    return build_pipeline_config(overrides)  # type: ignore[arg-type]


@app.post("/process")
async def process_audio(
    audio: UploadFile = File(...),
    output_format: str = "json",
) -> Any:
    """Run the DiaRemot pipeline on an uploaded audio file."""
    if not _pipeline_available():
        raise HTTPException(status_code=503, detail="DiaRemot pipeline not available")

    if output_format not in {"json", "csv", "all"}:
        raise HTTPException(status_code=400, detail="Invalid output_format")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        input_path = temp_dir / "input_audio"
        output_dir = temp_dir / "outputs"
        output_dir.mkdir()

        try:
            _save_upload(audio, input_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to save uploaded file")
            raise HTTPException(status_code=400, detail=f"Failed to read upload: {exc}") from exc

        try:
            config = _build_config(output_dir)
            result = run_pipeline(  # type: ignore[call-arg]
                str(input_path),
                str(output_dir),
                config=config,
                clear_cache=False,
            )
        except Exception as exc:
            logger.exception("Pipeline execution failed")
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

        if output_format == "csv":
            csv_path = output_dir / "diarized_transcript_with_emotion.csv"
            if not csv_path.exists():
                raise HTTPException(status_code=500, detail="CSV output missing from pipeline")
            return FileResponse(
                csv_path,
                media_type="text/csv",
                filename=f"{audio.filename or 'transcript'}.csv",
            )

        if output_format == "all":
            archive_path = temp_dir / "outputs.zip"
            shutil.make_archive(str(archive_path.with_suffix("")), "zip", output_dir)
            return FileResponse(
                archive_path,
                media_type="application/zip",
                filename=f"{audio.filename or 'results'}.zip",
            )

        # Default: JSON response with manifest and QC summary if present.
        response: Dict[str, Any] = {
            "status": "success",
            "filename": audio.filename,
            "result": result,
        }
        qc_path = output_dir / "qc_report.json"
        if qc_path.exists():
            import json

            response["summary"] = json.loads(qc_path.read_text(encoding="utf-8"))
        return JSONResponse(response)
