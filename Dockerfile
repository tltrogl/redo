1 # Multi-stage build for diaremot2-on
2 # CPU-only speech intelligence pipeline
3 
4 ARG PYTHON_VERSION=3.11
5 
6 # Stage 1: Builder - install dependencies
7 FROM python:${PYTHON_VERSION}-slim AS builder
8 
9 # Set working directory
10 WORKDIR /build
11 
12 # Install system dependencies
13 RUN apt-get update && apt-get install -y --no-install-recommends \
    14     ffmpeg \
    15     build-essential \
    16     git \
    17     && rm -rf /var/lib/apt/lists/*
18 
19 # Create a virtual environment and install dependencies
20 ENV VIRTUAL_ENV=/opt/venv
21 RUN python -m venv $VIRTUAL_ENV
22 ENV PATH="$VIRTUAL_ENV/bin:$PATH"
23 
24 # Copy requirements first (for better caching)
25 COPY requirements.txt requirements.in ./
26 
27 # Upgrade pip and install dependencies
28 RUN pip install --upgrade pip wheel setuptools && \
    29     pip install --no-cache-dir -r requirements.txt
30 
31 # Copy source code
32 COPY pyproject.toml README.md ./\nCOPY src/ ./src/\nCOPY tests/ ./tests/\n\n# Install package\nRUN pip
33 install --no-cache-dir .
34 # Stage 2: Runtime - minimal image
35 FROM python:${PYTHON_VERSION}-slim
36 
37 # Set working directory
38 WORKDIR /app
39 
40 # Install only runtime system dependencies
41 RUN apt-get update && apt-get install -y --no-install-recommends \
    42     ffmpeg \
    43     && rm -rf /var/lib/apt/lists/*
44 
45 # Copy Python packages from builder
46 COPY --from=builder /opt/venv /opt/venv
47 
48 # Copy source code
49 COPY src/ ./src/
50 COPY pyproject.toml README.md ./\n
51 
52 # Set environment variables
53 ENV PYTHONUNBUFFERED=1 \
    54     PYTHONDONTWRITEBYTECODE=1 \
    55     PATH="/opt/venv/bin:$PATH" \
    56     # Model directory (can be overridden at runtime)
57     DIAREMOT_MODEL_DIR=/srv/models \
    58     # HuggingFace cache
59     HF_HOME=/app/.cache \
    60     HUGGINGFACE_HUB_CACHE=/app/.cache/hub \
    61     TRANSFORMERS_CACHE=/app/.cache/transformers \
    62     TORCH_HOME=/app/.cache/torch \
    63     # CPU threading
64     OMP_NUM_THREADS=4 \
    65     MKL_NUM_THREADS=4 \
    66     NUMEXPR_MAX_THREADS=4 \
    67     TOKENIZERS_PARALLELISM=false
68 
69 # Create directories
70 RUN mkdir -p /srv/models /app/outputs /app/.cache && \
    71     chmod -R 777 /app/outputs /app/.cache
72 
73 # Expose volume mount points
74 VOLUME ["/srv/models", "/app/outputs"]
75 
76 # Health check
77 HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    78     CMD python -c "import diaremot; print('OK')" || exit 1
79 
80 # Default entrypoint
81 ENTRYPOINT ["python", "-m", "diaremot.cli"]
82 
83 # Default command (shows help)
84 CMD ["--help"]
85 
86 # Labels
87 LABEL maintainer="Timothy Leigh Troglin" \
    88       version="2.2.0" \
    89       description="CPU-only speech intelligence pipeline with diarization, ASR, and affect analysis" \
    90       org.opencontainers.image.source="https://github.com/your-org/diaremot2-on" \
    91       org.opencontainers.image.documentation=
"https://github.com/your-org/diaremot2-on/blob/main/README.md" \
    92       org.opencontainers.image.vendor="DiaRemot" \
    93       org.opencontainers.image.licenses="MIT"