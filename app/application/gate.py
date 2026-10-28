from dataclasses import dataclass

import numpy as np

from app.application.params import GateParams
from app.domain.checks.base import Frame
from app.domain.checks.chains import build_single_frame_chain
from app.domain.face import DetectedFace
from app.domain.ports.liveness import LivenessScorer
from app.domain.ports.metrics import FaceImageMetrics


@dataclass(frozen=True)
class FaceGate:
    """Runs the single-frame chain and returns the anti-spoof score.

    Returns None without running any check when liveness is off — the
    behaviour of the old _run_quality_and_liveness. Quality gating is
    deliberately coupled to the liveness flag; do not decouple them here.
    """

    metrics: FaceImageMetrics
    scorer: LivenessScorer

    def run(
        self,
        image_bgr: np.ndarray,
        detected: DetectedFace,
        *,
        liveness: bool,
        gate: GateParams,
    ) -> float | None:
        if not liveness:
            return None
        frame = Frame(
            image_bgr=image_bgr,
            detected=detected,
            metrics=self.metrics,
            scorer=self.scorer,
        )
        for check in build_single_frame_chain(gate):
            check.verify(frame)
        return frame.liveness_score
