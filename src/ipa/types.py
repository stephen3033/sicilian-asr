from dataclasses import dataclass

import torch

from common.errors import ResourceError


@dataclass
class AcousticOutput:
    tokens: list[str]
    blank_id: int
    preprocessing: dict
    log_probs: torch.Tensor
    lengths: torch.Tensor

    def validate(self) -> None:
        p, lengths = self.log_probs, self.lengths
        if (
            p.device.type != "cpu"
            or p.dtype != torch.float32
            or p.ndim != 3
            or p.shape[-1] != len(self.tokens)
        ):
            raise ResourceError(
                "Expected float32 CPU [batch, frames, vocabulary] probabilities"
            )
        if len(set(self.tokens)) != len(self.tokens) or not 0 <= self.blank_id < len(
            self.tokens
        ):
            raise ResourceError("Invalid token inventory or blank ID")
        if (
            lengths.device.type != "cpu"
            or lengths.dtype not in (torch.int32, torch.int64)
            or lengths.shape != (p.shape[0],)
        ):
            raise ResourceError("Invalid frame lengths shape/type")
        if not p.shape[0] or not torch.all((lengths > 0) & (lengths <= p.shape[1])):
            raise ResourceError("Invalid frame lengths")
        if not torch.isfinite(p).all():
            raise ResourceError("Nonfinite acoustic probabilities")
        valid = torch.arange(p.shape[1])[None, :] < lengths[:, None]
        if p.logsumexp(-1)[valid].abs().max().item() > 1e-4:
            raise ResourceError(
                "Acoustic outputs are not normalized natural-log probabilities"
            )
