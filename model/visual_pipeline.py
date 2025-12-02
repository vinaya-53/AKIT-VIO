# models/visual_pipeline.py
import torch.nn as nn

from model.enhance import UnderwaterEnhancer
from model.visual_extractor import VisualFeatureExtractor

class VisualPipeline(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.enhancer = UnderwaterEnhancer()
        self.extractor = VisualFeatureExtractor(feat_dim)

    def forward(self, img):
        enhanced = self.enhancer(img)
        feat = self.extractor(enhanced)
        return enhanced, feat
