# models/enhance.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnderwaterEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.color_scale = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.sharpen_strength = nn.Parameter(torch.tensor(0.3))
        self.dehaze_strength = nn.Parameter(torch.tensor(0.8))

    def gray_world_correction(self, img):
        mean_rgb = img.mean(dim=[2, 3], keepdim=True)
        mean_gray = mean_rgb.mean(dim=1, keepdim=True)
        corrected = img * (mean_gray / (mean_rgb + 1e-6))
        return torch.clamp(corrected, 0, 1)

    def dehaze(self, img):
        min_channel = torch.min(img, dim=1, keepdim=True)[0]
        dark = -F.max_pool2d(-min_channel, kernel_size=15, stride=1, padding=7)
        t = 1 - self.dehaze_strength * dark
        out = (img - dark) / (t + 1e-6)
        return torch.clamp(out, 0, 1)

    def contrast_enhance(self, img):
        eps = 1e-4
        mean = F.avg_pool2d(img, 25, stride=1, padding=12)
        sq_mean = F.avg_pool2d(img * img, 25, stride=1, padding=12)
        std = torch.sqrt(torch.clamp(sq_mean - mean * mean, min=eps))
        enhanced = (img - mean) / (std + eps)
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + eps)
        return enhanced

    def sharpen(self, img):
        blur = F.avg_pool2d(img, kernel_size=3, stride=1, padding=1)
        highfreq = img - blur
        return torch.clamp(img + self.sharpen_strength * highfreq, 0, 1)

    def forward(self, x):
        x = self.gray_world_correction(x)
        x = self.dehaze(x)
        x = self.contrast_enhance(x)
        x = self.sharpen(x)
        x = x * self.color_scale.view(1, 3, 1, 1)
        return torch.clamp(x, 0, 1)
