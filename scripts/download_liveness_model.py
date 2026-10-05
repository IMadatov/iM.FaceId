"""Download MiniFASNet-V2 ONNX (Silent-Face-Anti-Spoofing) into MODEL_DIR/liveness."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx"
    "/resolve/main/minifasnet_v2.onnx"
)


def main() -> None:
    model_dir = Path(os.environ.get("MODEL_DIR", "./models"))
    out_dir = model_dir / "liveness"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "minifasnet_v2.onnx"
    url = os.environ.get("LIVENESS_MODEL_URL", DEFAULT_URL)
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    print(f"liveness model ready: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
