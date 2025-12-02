import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# ============================================================
# ORB Keypoint Extractor Wrapper
# ============================================================

class ORBKeypointExtractor:
    """
    Wrapper around OpenCV-ORB
    Outputs:
        coords: (B, K, 2)
        desc:   (B, K, D)
        mask:   (B, K)
    """

    def __init__(self, max_keypoints=1000, descriptor_size=32):
        if cv2 is None:
            raise RuntimeError("Install OpenCV: pip install opencv-python")

        self.max_kpts = max_keypoints
        self.descriptor_size = descriptor_size

        # ---- Improved ORB parameters ----
        self.orb = cv2.ORB_create(
            nfeatures=max_keypoints,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=5
        )

    def __call__(self, imgs):
        """
        imgs: numpy (H,W,3) or (B,H,W,3)
              OR torch (B,3,H,W)
        """
        # -------- Convert torch -> numpy batch --------
        if isinstance(imgs, torch.Tensor):
            imgs_np = imgs.detach().cpu().numpy()
            imgs_np = (imgs_np * 255).astype(np.uint8)
            imgs_np = np.transpose(imgs_np, (0, 2, 3, 1))
        else:
            imgs_np = imgs
            if imgs_np.ndim == 3:
                imgs_np = imgs_np[None, ...]  # make batch=1

        B = imgs_np.shape[0]
        K = self.max_kpts
        D = self.descriptor_size

        coords = np.zeros((B, K, 2), dtype=np.float32)
        descs = np.zeros((B, K, D), dtype=np.float32)
        mask = np.zeros((B, K), dtype=np.bool_)

        # -------- Process each frame --------
        for i in range(B):
            img = imgs_np[i]

            # ---- ROBUST GRAYSCALE ----
            if img.ndim == 3:
                ch = img.shape[2]
                if ch == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif ch == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                elif ch == 1:
                    gray = img[:, :, 0]
                else:
                    raise ValueError(f"Unexpected channel count: {img.shape}")
            elif img.ndim == 2:
                gray = img
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

            # ---- ORB detection ----
            kps, dc = self.orb.detectAndCompute(gray, None)

            if kps is None or len(kps) == 0:
                continue

            # Sort strongest → first
            idx = np.argsort([-kp.response for kp in kps])
            n_take = min(len(kps), K)
            idx = idx[:n_take]

            # Fill coords + mask
            for j, k in enumerate(idx):
                kp = kps[k]
                coords[i, j, 0] = kp.pt[0]
                coords[i, j, 1] = kp.pt[1]
                mask[i, j] = True

            # Fill descriptors
            if dc is not None:
                dsel = dc[idx]
                if dsel.shape[1] != D:
                    padded = np.zeros((n_take, D), dtype=np.float32)
                    padded[:, :dsel.shape[1]] = dsel
                    dsel = padded
                descs[i, :n_take] = dsel

        return (
            torch.from_numpy(coords).float(),
            torch.from_numpy(descs).float(),
            torch.from_numpy(mask)
        )


# ============================================================
# Simple Dense Descriptor (+ automatic coord rescaling)
# ============================================================

class SimpleDenseDescriptor(nn.Module):
    """
    CNN feature map (H/8 x W/8 x C)
    + grid_sample to get per-keypoint descriptors
    """

    def __init__(self, out_dim=64):
        super().__init__()
        self.out_dim = out_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, 3, padding=1, stride=2)
        )

    def forward(self, imgs, kpt_coords=None):
        """
        imgs: (B,3,H,W)
        kpt_coords: (B,K,2) in original pixel coordinates
        """
        feat = self.backbone(imgs)  # (B,C,Hf,Wf)

        if kpt_coords is None:
            return feat

        B, C, Hf, Wf = feat.shape
        B2, K, _ = kpt_coords.shape

        assert B == B2, "Batch mismatch"

        H, W = imgs.shape[2], imgs.shape[3]

        # rescale original pixel coords → feature map coords
        kx = kpt_coords[..., 0] * (Wf / W)
        ky = kpt_coords[..., 1] * (Hf / H)

        # convert to [-1,1]
        nx = (kx / (Wf - 1)) * 2 - 1
        ny = (ky / (Hf - 1)) * 2 - 1

        grid = torch.stack([nx, ny], dim=-1).view(B, K, 1, 2)

        sampled = F.grid_sample(
            feat,
            grid,
            mode="bilinear",
            align_corners=True
        )

        # (B, C, K, 1) → (B, K, C)
        return sampled.squeeze(-1).permute(0, 2, 1)
