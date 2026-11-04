"""U-Net for 2-channel SAR water segmentation, with a dropout layer for MC dropout."""
from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class FloodNet(nn.Module):
    def __init__(self, encoder: str = "resnet34", encoder_weights: str | None = "imagenet",
                 dropout: float = 0.2):
        super().__init__()
        self.unet = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                             in_channels=2, classes=1)
        head = self.unet.segmentation_head
        self.unet.segmentation_head = nn.Sequential(nn.Dropout2d(dropout), head)
        self.config = {"encoder": encoder, "dropout": dropout}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns water logits, shape (B, H, W)."""
        return self.unet(x)[:, 0]


def save_checkpoint(path, model: FloodNet, stats, extra: dict | None = None):
    torch.save({"state_dict": model.state_dict(), "config": model.config,
                "stats": stats.to_dict(), "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    from .data import NormStats
    ck = torch.load(path, map_location=device, weights_only=False)
    m = FloodNet(encoder=ck["config"]["encoder"], encoder_weights=None,
                 dropout=ck["config"]["dropout"])
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    return m, NormStats.from_dict(ck["stats"]), ck.get("extra", {})
