from __future__ import annotations

import pytest
import torch

from ml.training.stability import (
    NonFiniteTrainingError,
    StabilityMonitorCallback,
)


def test_stability_monitor_fails_at_nonfinite_backpropagated_gradient() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    monitor = StabilityMonitorCallback(torch, model, "fp16")

    try:
        with pytest.raises(NonFiniteTrainingError, match="--precision fp32"):
            (model.weight.sum() * torch.tensor(float("nan"))).backward()
    finally:
        monitor.close()


def test_stability_monitor_fails_at_nonfinite_loss() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    monitor = StabilityMonitorCallback(torch, model, "fp32")

    try:
        with pytest.raises(NonFiniteTrainingError, match="Non-finite loss"):
            monitor.record_loss(torch.tensor(float("inf")))
    finally:
        monitor.close()
