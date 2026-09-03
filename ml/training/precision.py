"""Hardware-aware mixed-precision selection shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PrecisionName = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class PrecisionSelection:
    name: PrecisionName
    compute_capability: tuple[int, int] | None
    bf16_runtime_reported: bool

    @property
    def trainer_fp16(self) -> bool:
        return self.name == "fp16"

    @property
    def trainer_bf16(self) -> bool:
        return self.name == "bf16"

    def torch_dtype(self, torch: Any) -> Any:
        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.name]


def choose_precision_name(
    *,
    cuda_available: bool,
    compute_capability: tuple[int, int] | None,
    bf16_runtime_reported: bool,
) -> PrecisionName:
    """Choose BF16 only for Ampere-or-newer CUDA devices."""
    if not cuda_available:
        return "fp32"
    if compute_capability is None:
        raise ValueError("CUDA compute capability is required for a CUDA device")
    major, _minor = compute_capability
    if major >= 8 and bf16_runtime_reported:
        return "bf16"
    return "fp16"


def select_precision(torch: Any, device_index: int = 0) -> PrecisionSelection:
    if not torch.cuda.is_available():
        return PrecisionSelection(
            name="fp32",
            compute_capability=None,
            bf16_runtime_reported=False,
        )
    capability = tuple(torch.cuda.get_device_capability(device_index))
    bf16_reported = bool(torch.cuda.is_bf16_supported())
    return PrecisionSelection(
        name=choose_precision_name(
            cuda_available=True,
            compute_capability=capability,
            bf16_runtime_reported=bf16_reported,
        ),
        compute_capability=capability,
        bf16_runtime_reported=bf16_reported,
    )
