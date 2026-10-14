from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerignore_excludes_heavy_local_paths():
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in (".venv", ".git", "tests", "graphify-out", ".worktrees", "models"):
        assert pattern in text, f"missing {pattern!r} in .dockerignore"


def test_dockerfile_bakes_models_and_runs_uvicorn():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "MODEL_DIR=/app/models" in text
    assert "scripts/download_models.py" in text
    assert "scripts/download_liveness_model.py" in text
    assert "uvicorn" in text
    assert "0.0.0.0" in text
    assert "8000" in text


def test_compose_api_and_qdrant_limits_and_env():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "api:" in text
    assert "qdrant:" in text
    assert "mem_limit: 4g" in text
    assert 'cpus: "2.0"' in text or "cpus: '2.0'" in text
    assert "mem_limit: 2g" in text
    assert 'cpus: "1.0"' in text or "cpus: '1.0'" in text
    assert "QDRANT_URL=http://qdrant:6333" in text or "QDRANT_URL: http://qdrant:6333" in text
    assert "MODEL_DIR=/app/models" in text or "MODEL_DIR: /app/models" in text
    assert "LIVENESS_MODEL_PATH=/app/models/liveness/minifasnet_v2.onnx" in text or (
        "LIVENESS_MODEL_PATH: /app/models/liveness/minifasnet_v2.onnx" in text
    )
    assert "condition: service_healthy" in text
    assert "127.0.0.1:6333:6333" in text
    assert "127.0.0.1:6334:6334" in text
    assert '"6333:6333"' not in text and "'6333:6333'" not in text
    assert "restart: unless-stopped" in text


def test_compose_gpu_profile_api_gpu_service():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "api-gpu:" in text
    assert "profiles:" in text
    assert "gpu" in text
    assert "Dockerfile.gpu" in text
    assert "FACE_DEVICE: ${FACE_DEVICE:-auto}" in text
    assert "gpus:" in text or "nvidia" in text.lower()


def test_dockerfile_gpu_uses_onnxruntime_gpu():
    text = (ROOT / "Dockerfile.gpu").read_text(encoding="utf-8")
    assert "requirements-gpu.txt" in text
    assert "python3.11 -m venv /opt/venv" in text
    assert "ENV PATH=/opt/venv/bin:$PATH" in text
    assert "ln -sf" not in text
    req = (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
    assert "onnxruntime-gpu" in req
    assert "onnxruntime>=" not in req.replace("onnxruntime-gpu", "")
    assert "download_models.py" in text
    assert "uvicorn" in text
