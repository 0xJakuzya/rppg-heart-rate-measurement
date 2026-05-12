import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config

class PhysNet(nn.Module):
    """PhysNet for multi-ROI rPPG windows.

    Input shape: [batch, time, roi, channels, height, width].
    Output shape: [batch, time].
    """
    def __init__(self):
        super().__init__()
        stem_channels = config.PHYSNET_STEM_CHANNELS
        enc1_channels, enc2_channels, enc3_channels = config.PHYSNET_ENCODER_CHANNELS
        bottleneck_channels = config.PHYSNET_BOTTLENECK_CHANNELS
        decoder_channels = config.PHYSNET_DECODER_CHANNELS

        self.stem = nn.Sequential(
            nn.Conv3d(3, stem_channels, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(stem_channels),
            nn.ELU(inplace=True),
        )

        self.enc1 = nn.Sequential(
            nn.Conv3d(stem_channels, enc1_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(enc1_channels),
            nn.ELU(inplace=True),
            nn.Conv3d(enc1_channels, enc1_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(enc1_channels),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )

        self.enc2 = nn.Sequential(
            nn.Conv3d(enc1_channels, enc2_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(enc2_channels),
            nn.ELU(inplace=True),
            nn.Conv3d(enc2_channels, enc2_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(enc2_channels),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )

        self.enc3 = nn.Sequential(
            nn.Conv3d(enc2_channels, enc3_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(enc3_channels),
            nn.ELU(inplace=True),
            nn.Conv3d(enc3_channels, enc3_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(enc3_channels),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv3d(enc3_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(bottleneck_channels),
            nn.ELU(inplace=True),
            nn.Conv3d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(bottleneck_channels),
            nn.ELU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(bottleneck_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(decoder_channels),
            nn.ELU(inplace=True),
        )

        self.head = nn.Conv3d(decoder_channels, 1, kernel_size=1)

    @staticmethod
    def make_mosaic(x: torch.Tensor) -> torch.Tensor:
        batch, time, roi, channels, height, width = x.shape
        expected_roi = config.ROI_MOSAIC_ROWS * config.ROI_MOSAIC_COLS
        if roi != expected_roi:
            raise ValueError(f"Expected {expected_roi} ROI patches, got {roi}.")
        x = x.reshape(batch,time, config.ROI_MOSAIC_ROWS, config.ROI_MOSAIC_COLS, channels, height, width)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(batch, time, channels, config.ROI_MOSAIC_ROWS * height, config.ROI_MOSAIC_COLS * width)
        return x.permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_time = x.shape[1]
        features = self.make_mosaic(x)
        features = self.stem(features)
        features = self.enc1(features)
        features = self.enc2(features)
        features = self.enc3(features)
        features = self.bottleneck(features)
        features = F.interpolate(features, size=(target_time, features.shape[-2], features.shape[-1]),
                                 mode="trilinear", align_corners=False)
        features = self.decoder(features)
        features = F.adaptive_avg_pool3d(features, (target_time, 1, 1))
        features = self.head(features)
        return features.view(features.shape[0], target_time)
