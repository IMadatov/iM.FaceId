from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedDevice:
    requested: str
    effective: str
    providers: list[str]
    ctx_id: int
    fallback_reason: str | None = None


_CPU = "CPUExecutionProvider"
_CUDA = "CUDAExecutionProvider"
_VALID = frozenset({"auto", "cpu", "cuda"})


def resolve_face_device(
    requested: str, *, available_providers: list[str] | None = None
) -> ResolvedDevice:
    if requested not in _VALID:
        raise ValueError(
            f"Invalid FACE_DEVICE={requested!r}; expected one of: auto, cpu, cuda"
        )
    if available_providers is None:
        import onnxruntime as ort

        available_providers = list(ort.get_available_providers())

    cuda_ok = _CUDA in available_providers

    if requested == "cpu" or (requested in {"auto", "cuda"} and not cuda_ok):
        reason = None
        if requested != "cpu" and not cuda_ok:
            reason = f"{_CUDA} not available in {available_providers}"
        return ResolvedDevice(
            requested=requested,
            effective="cpu",
            providers=[_CPU],
            ctx_id=-1,
            fallback_reason=reason,
        )

    return ResolvedDevice(
        requested=requested,
        effective="cuda",
        providers=[_CUDA, _CPU],
        ctx_id=0,
        fallback_reason=None,
    )
