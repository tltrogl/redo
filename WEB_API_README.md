# DiaRemot Web API - Installation & Usage

## Installation

### Backend API

1. **Install with web dependencies:**
   ```bash
   pip install -e ".[web]"
   ```

   This installs:
   - FastAPI (web framework)
   - Uvicorn (ASGI server)
   - python-multipart (file upload support)
   - websockets (real-time progress)
   - pydantic (data validation)

2. **Verify installation:**
   ```bash
   python -c "from diaremot.web.api.app import app; print('✓ Backend ready')"
   ```

### Frontend (Optional)

```bash
cd frontend/frontend
npm install
```

## Running the API

### Development Server

**Option 1: Using the helper script**
```bash
python src/diaremot/web/server.py
```

**Option 2: Using uvicorn directly**
```bash
uvicorn diaremot.web.api.app:app --reload --port 8000
```

**Option 3: Production mode**
```bash
uvicorn diaremot.web.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access the API

- **API Documentation:** http://localhost:8000/api/docs
- **ReDoc Documentation:** http://localhost:8000/api/redoc
- **Health Check:** http://localhost:8000/health
- **OpenAPI JSON:** http://localhost:8000/api/openapi.json

## API Endpoints

### Configuration
- `GET /api/config/schema` - Get complete configuration schema (80+ parameters)
- `GET /api/config/presets` - List available presets
- `GET /api/config/presets/{name}` - Get specific preset
- `POST /api/config/validate` - Validate configuration
- `GET /api/config/defaults` - Get default values

### File Management
- `POST /api/files/upload` - Upload audio file
- `DELETE /api/files/{file_id}` - Delete uploaded file

### Job Management
- `POST /api/jobs` - Create processing job
- `GET /api/jobs/{job_id}` - Get job status
- `GET /api/jobs` - List all jobs
- `POST /api/jobs/{job_id}/cancel` - Cancel job
- `GET /api/jobs/{job_id}/results` - List result files
- `GET /api/jobs/{job_id}/results/{filename}` - Download result
- `DELETE /api/jobs/{job_id}` - Delete job outputs

### WebSocket
- `WS /api/jobs/{job_id}/progress` - Real-time progress updates
- `WS /api/jobs/subscribe` - Subscribe to all jobs

### Health
- `GET /health` - Health check with model availability
- `GET /` - Service information

## Example Usage

### 1. Upload Audio File
```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/files/upload",
        files={"file": f}
    )
file_info = response.json()
print(f"File ID: {file_info['file_id']}")
```

### 2. Create Processing Job
```python
job_response = requests.post(
    "http://localhost:8000/api/jobs",
    json={
        "filename": file_info["file_id"],
        "preset": "accurate",  # or "fast", "offline", "balanced"
        "config": {
            "enable_affect": True,
            "enable_sed": True
        }
    }
)
job = job_response.json()
print(f"Job ID: {job['job_id']}")
```

### 3. Monitor Progress (WebSocket)
```python
import websockets
import asyncio
import json

async def monitor_progress(job_id):
    uri = f"ws://localhost:8000/api/jobs/{job_id}/progress"
    async with websockets.connect(uri) as ws:
        async for message in ws:
            data = json.loads(message)
            if data["type"] == "progress":
                progress = data["progress"]
                print(f"Stage: {progress['stage']}")
                print(f"Progress: {progress['overall_progress']:.1f}%")
            elif data["type"] == "completed":
                print("Job completed!")
                break
            elif data["type"] == "error":
                print(f"Error: {data['error']}")
                break

asyncio.run(monitor_progress(job["job_id"]))
```

### 4. Download Results
```python
results = requests.get(f"http://localhost:8000/api/jobs/{job['job_id']}/results").json()
for file in results["files"]:
    print(f"- {file['filename']} ({file['size']} bytes)")
    
# Download CSV transcript
csv_url = f"http://localhost:8000/api/jobs/{job['job_id']}/results/diarized_transcript_with_emotion.csv"
response = requests.get(csv_url)
with open("transcript.csv", "wb") as f:
    f.write(response.content)
```

## Configuration

### Presets
Four built-in presets are available:
- **fast**: Quick processing with reduced accuracy
- **accurate**: Maximum accuracy (slower)
- **offline**: Works without internet (no HF models)
- **balanced**: Good balance of speed and accuracy (default)

### Custom Configuration
Override any of the 80+ parameters:
```json
{
  "vad_threshold": 0.5,
  "transcribe_language": "en",
  "enable_affect": true,
  "enable_sed": true,
  "sed_mode": "audio-only",
  "num_threads": 4
}
```

Get the full schema with descriptions:
```bash
curl http://localhost:8000/api/config/schema | jq .
```

## Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│  ┌────────────┐  ┌──────────────────┐  │
│  │   Routes   │  │    Services      │  │
│  │  (REST)    │→ │  - JobQueue      │  │
│  └────────────┘  │  - Storage       │  │
│                   └──────────────────┘  │
│  ┌────────────┐                         │
│  │ WebSocket  │                         │
│  │  Progress  │                         │
│  └────────────┘                         │
└─────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│      DiaRemot Pipeline Core             │
│   (11-stage processing pipeline)        │
└─────────────────────────────────────────┘
```

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError: No module named 'fastapi'`:
```bash
pip install -e ".[web]"
```

### Models Not Available
If health check shows `models_available: false`:
```bash
# Download and extract models
./setup.sh  # Linux/macOS
# or
.\setup.ps1  # Windows
```

### Port Already in Use
Change the port:
```bash
uvicorn diaremot.web.api.app:app --port 8080
```

### CORS Issues
The API allows all origins by default. For production, edit `src/diaremot/web/api/app.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Frontend Development

The frontend is scaffolded but not yet implemented. To develop:

```bash
cd frontend/frontend
npm run dev
```

Access at http://localhost:3000

## Production Deployment

See `CLOUDRUN_DEPLOY.md` for Cloud Run deployment or create your own:

**Docker:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e ".[web]"

CMD ["uvicorn", "diaremot.web.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - DIAREMOT_MODEL_DIR=/app/models
```

## Next Steps

1. Install dependencies: `pip install -e ".[web]"`
2. Start the server: `python src/diaremot/web/server.py`
3. Open API docs: http://localhost:8000/api/docs
4. Try the examples above
5. Build the frontend UI (see `WEB_APP_PROGRESS.md` for remaining tasks)
