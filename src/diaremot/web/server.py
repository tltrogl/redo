"""Development server runner for DiaRemot Web API."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def main():
    """Run the development server."""
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set default model directory if not set
    if "DIAREMOT_MODEL_DIR" not in os.environ:
        default_model_dir = Path("models").resolve()
        if default_model_dir.exists():
            os.environ["DIAREMOT_MODEL_DIR"] = str(default_model_dir)
            print(f"Using model directory: {default_model_dir}")

    # Create necessary directories
    Path("uploads").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("jobs").mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("DiaRemot Web API - Development Server")
    print("=" * 60)
    print("\nStarting server at http://localhost:8000")
    print("API Documentation: http://localhost:8000/api/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nPress CTRL+C to stop")
    print("=" * 60 + "\n")

    # Run server
    uvicorn.run(
        "diaremot.web.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )


if __name__ == "__main__":
    main()
