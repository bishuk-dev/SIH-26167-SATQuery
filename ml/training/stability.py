"""Fail-fast numerical checks for Phase 2B LoRA training."""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback


class NonFiniteTrainingError(RuntimeError):
    """Raised before training can continue with NaN or infinite state."""


class StabilityMonitorCallback(TrainerCallback):
    def __init__(self, torch: Any, model: Any, precision: str) -> None:
        self._torch = torch
        self._precision = precision
        self._initial_parameters = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self._hook_handles = [
            parameter.register_hook(self._gradient_hook(name))
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.microbatch_losses: list[float] = []
        self.gradient_norms: list[float] = []
        self.optimizer_steps = 0

    def record_loss(self, loss: Any) -> None:
        value = loss.detach().float()
        if not self._torch.isfinite(value).all().item():
            self._fail("loss", "training_step")
        self.microbatch_losses.append(float(value.item()))

    def on_pre_optimizer_step(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        model = kwargs["model"]
        gradients = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if not self._torch.isfinite(parameter.detach()).all().item():
                self._fail("trainable parameter", name)
            if parameter.grad is not None:
                gradient = parameter.grad.detach().float()
                if not self._torch.isfinite(gradient).all().item():
                    self._fail("accumulated gradient", name)
                gradients.append(gradient.norm(2))
        if not gradients:
            raise NonFiniteTrainingError(
                "No LoRA gradients were present before the optimizer step"
            )
        norm = self._torch.stack(gradients).norm(2)
        if not self._torch.isfinite(norm).item():
            self._fail("gradient norm", f"optimizer_step_{state.global_step + 1}")
        self.gradient_norms.append(float(norm.item()))

    def on_optimizer_step(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        model = kwargs["model"]
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and not self._torch.isfinite(
                parameter.detach()
            ).all().item():
                self._fail("updated trainable parameter", name)
        self.optimizer_steps += 1

    def verify_parameters_changed(self, model: Any) -> bool:
        for name, parameter in model.named_parameters():
            if name not in self._initial_parameters:
                continue
            current = parameter.detach().float().cpu()
            if not self._torch.equal(current, self._initial_parameters[name]):
                return True
        raise NonFiniteTrainingError(
            "Training completed without changing any trainable LoRA parameter"
        )

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def report(self, *, parameters_changed: bool) -> dict[str, Any]:
        return {
            "optimizer_steps_checked": self.optimizer_steps,
            "microbatches_checked": len(self.microbatch_losses),
            "losses": self.microbatch_losses,
            "gradient_norms": self.gradient_norms,
            "trainable_parameters_changed": parameters_changed,
        }

    def _gradient_hook(self, name: str):
        def require_finite(gradient: Any) -> Any:
            if not self._torch.isfinite(gradient.detach()).all().item():
                self._fail("backpropagated gradient", name)
            return gradient

        return require_finite

    def _fail(self, quantity: str, location: str) -> None:
        fallback = (
            " Rerun the stability smoke with --precision fp32 on this GPU."
            if self._precision == "fp16"
            else ""
        )
        raise NonFiniteTrainingError(
            f"Non-finite {quantity} detected at {location} using "
            f"{self._precision}; training stopped before continuing.{fallback}"
        )
