import torch
import torch.nn as nn
import torch.nn.functional as F

class UnderwaterEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Trainable parameters for fine-tuning
        self.sharpen_strength = nn.Parameter(torch.tensor(1.5)) # Increased for ripple edges
        self.contrast_clip = nn.Parameter(torch.tensor(0.02))   # Differentiable clip limit
        self.gamma = nn.Parameter(torch.tensor(1.2))            # To combat overexposure

    def local_contrast_enhance(self, x):
        # Calculate local mean and variance across channels
        # x is (B, 3, H, W)
        mean = F.avg_pool2d(x, kernel_size=13, stride=1, padding=6)
        sq_mean = F.avg_pool2d(x**2, kernel_size=13, stride=1, padding=6)
        var = torch.relu(sq_mean - mean**2)
        std = torch.sqrt(var + 1e-5)
        
        x = (x - mean) / (std + self.contrast_clip)
        
        # FIX: Robust min-max scaling for 3-channel tensors
        b, c, h, w = x.shape
        x_view = x.view(b, c, -1)
        x_min = x_view.min(2)[0].view(b, c, 1, 1)
        x_max = x_view.max(2)[0].view(b, c, 1, 1)
        
        return (x - x_min) / (x_max - x_min + 1e-6)

    def adaptive_sharpen(self, x):
        # Laplacian kernel
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        # Apply per channel
        # Use groups=3 to process R, G, and B independently
        kernel = kernel.repeat(3, 1, 1, 1)
        edges = F.conv2d(x, kernel, padding=1, groups=3)
            
        return torch.clamp(x + self.sharpen_strength * edges, 0, 1)
    def gamma_correction(self, x):
        """Combats 'washed out' images by re-mapping midtones."""
        return torch.pow(x + 1e-6, self.gamma)

    def forward(self, x):
        # 1. Fix overexposure first
        x = self.gamma_correction(x)
        
        # 2. Local contrast enhancement (Replaces global Dehaze/GrayWorld for better keypoints)
        x = self.local_contrast_enhance(x)
        
        # 3. Aggressive sharpening to define ripple peaks
        x = self.adaptive_sharpen(x)
        
        return x