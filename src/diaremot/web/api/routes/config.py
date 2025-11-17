"""Configuration API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from diaremot.pipeline.config import build_pipeline_config
from diaremot.web.api.models import ConfigSchemaResponse, ConfigValidateRequest, PresetResponse
from diaremot.web.config_schema import generate_config_schema

router = APIRouter(prefix="/config", tags=["Configuration"])


@router.get("/schema", response_model=ConfigSchemaResponse)
async def get_config_schema():
    """Get the complete configuration schema with all parameters."""
    schema = generate_config_schema()
    return ConfigSchemaResponse(**schema)


@router.get("/presets", response_model=list[PresetResponse])
async def list_presets():
    """List all available configuration presets."""
    schema = generate_config_schema()
    presets = []

    for preset_id, preset_data in schema["presets"].items():
        presets.append(
            PresetResponse(
                name=preset_id,
                label=preset_data["name"],
                description=preset_data["description"],
                overrides=preset_data["overrides"],
            )
        )

    return presets


@router.get("/presets/{preset_name}", response_model=PresetResponse)
async def get_preset(preset_name: str):
    """Get a specific preset by name."""
    schema = generate_config_schema()
    preset_data = schema["presets"].get(preset_name)

    if not preset_data:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")

    return PresetResponse(
        name=preset_name,
        label=preset_data["name"],
        description=preset_data["description"],
        overrides=preset_data["overrides"],
    )


@router.post("/validate")
async def validate_config(request: ConfigValidateRequest):
    """Validate a configuration and return the merged result."""
    try:
        validated = build_pipeline_config(request.config)
        return {"valid": True, "config": validated}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Configuration validation failed: {e}")


@router.get("/defaults")
async def get_defaults():
    """Get default configuration values."""
    defaults = build_pipeline_config({})
    return defaults
