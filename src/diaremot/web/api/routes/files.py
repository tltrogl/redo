"""File upload and management routes."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from diaremot.web.api.models import FileUploadResponse
from diaremot.web.api.services import get_storage

router = APIRouter(prefix="/files", tags=["Files"])


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload an audio file.

    Returns:
        File upload response with file_id and metadata.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Read file content
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # Save file
    storage = get_storage()
    file_id, file_path, metadata = storage.save_upload(file.filename, content)

    return FileUploadResponse(
        file_id=file_id,
        filename=metadata["filename"],
        size=metadata["size"],
        duration=metadata.get("duration"),
        sample_rate=metadata.get("sample_rate"),
        channels=metadata.get("channels"),
    )


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file."""
    storage = get_storage()
    success = storage.cleanup_upload(file_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    return {"success": True, "message": "File deleted"}
