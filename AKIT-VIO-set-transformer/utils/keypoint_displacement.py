import os
import glob
import numpy as np
import torch
from PIL import Image
import pandas as pd
import cv2
from utils.innovation import (
    compute_innovation_from_triangulation,
    match_orb_descriptors,
    undistort_fisheye_points
)

import numpy as np
import torch

def avg_keypoint_displacement_from_lists(
    kpA, descA,
    kpB, descB,
    K,
    depth_map,          # ← DEPTH, not distortion
    ratio_test=0.9,
    min_depth=0.1
):
    # --- Tensor → NumPy ---
    for name, var in [('kpA', kpA), ('kpB', kpB), ('descA', descA), ('descB', descB)]:
        if torch.is_tensor(var):
            locals()[name] = var.detach().cpu().numpy()

    kpA_s = kpA.reshape(-1, 2)
    kpB_s = kpB.reshape(-1, 2)
    d1 = descA.reshape(-1, descA.shape[-1])
    d2 = descB.reshape(-1, descB.shape[-1])

    matches = match_orb_descriptors(d1, d2, cross_check=True, ratio_test=ratio_test)
    if len(matches) == 0:
        return 0.0, 0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    disp_sum = 0.0
    valid = 0

    for m in matches:
        u1, v1 = kpA_s[m.queryIdx]
        u2, v2 = kpB_s[m.trainIdx]

        ui, vi = int(round(u1)), int(round(v1))
        if ui < 0 or vi < 0 or ui >= depth_map.shape[1] or vi >= depth_map.shape[0]:
            continue

        z = depth_map[vi, ui]
        if z < min_depth:
            continue

        # Back-project to metric 3D (camera frame)
        P1 = np.array([(u1 - cx) * z / fx,
                       (v1 - cy) * z / fy,
                        z])

        P2 = np.array([(u2 - cx) * z / fx,
                       (v2 - cy) * z / fy,
                        z])

        disp_sum += np.linalg.norm(P2 - P1)
        valid += 1

    if valid == 0:
        return 0.0, 0

    return disp_sum / valid, valid

