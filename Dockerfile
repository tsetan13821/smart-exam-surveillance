# SmartSquad demo image: the API and proctor console, proctoring recorded videos on CPU.
# A container can't use the host's webcam on Windows/macOS, so the camera source is disabled.
#   docker compose up --build        (see README: "Docker demo")

# ---- 1. Build the React console ------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. Python runtime --------------------------------------------------------
FROM python:3.11-slim

# OpenCV and MediaPipe need these shared libraries even without a display
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencies first, so code changes don't reinstall them
COPY requirements.txt ./
COPY scripts/install_ci_deps.sh scripts/install_insightface_nobuild.py scripts/
RUN bash scripts/install_ci_deps.sh

USER app
# Download the model weights at build time so the first session starts without waiting
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition']).prepare(ctx_id=-1)" \
 && python -c "from transformers import RTDetrForObjectDetection, RTDetrImageProcessor; RTDetrImageProcessor.from_pretrained('PekingU/rtdetr_r50vd'); RTDetrForObjectDetection.from_pretrained('PekingU/rtdetr_r50vd')"

COPY --chown=app:app config.py main.py register_student.py desk_rois.json ./
COPY --chown=app:app app/ app/
COPY --chown=app:app api/ api/
COPY --chown=app:app --from=frontend /build/dist frontend/dist

# All data lives in /data (a volume); videos to proctor are mounted at /videos
ENV SMARTSQUAD_DB=/data/proctoring_session.db \
    SMARTSQUAD_REGISTRATIONS_DIR=/data/registrations \
    SMARTSQUAD_LOG_DIR=/data/proctoring_logs \
    SMARTSQUAD_LEGACY_REGISTRY=/data/student_encodings.json \
    SMARTSQUAD_DEMO_VIDEOS=/videos \
    SMARTSQUAD_DISABLE_CAMERA=1
USER root
RUN mkdir -p /data /videos /app/logs && chown app:app /data /videos /app/logs
USER app
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# Listen on all interfaces inside the container; docker-compose.yml publishes it on 127.0.0.1 only
CMD ["python", "-m", "api", "--host", "0.0.0.0", "--port", "8000"]
