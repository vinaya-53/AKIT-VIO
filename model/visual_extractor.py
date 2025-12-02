# models/visual_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualFeatureExtractor(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.projector = nn.Linear(128, feat_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = F.adaptive_avg_pool2d(h, (1, 1))
        h = h.view(h.size(0), -1)
        return self.projector(h)
