"""Download InsightFace buffalo_sc into MODEL_DIR (default ./models)."""
import os
from pathlib import Path

from insightface.app import FaceAnalysis


def main() -> None:
    root = Path(os.environ.get("MODEL_DIR", "./models"))
    root.mkdir(parents=True, exist_ok=True)
    app = FaceAnalysis(
        name="buffalo_sc", root=str(root), providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print(f"models ready under {root}")


if __name__ == "__main__":
    main()
