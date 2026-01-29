import os
import glob
import numpy as np
import torch
from PIL import Image
import pandas as pd
import cv2

def load_camera_intrinsics(path="camera_intrinsics.yml", img_shape=None):
    # Try OpenCV FileStorage first
    if os.path.exists(path):
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        cm_node = fs.getNode("camera_matrix")
        if not cm_node.empty():
            cam_m = cm_node.mat()
            d_node = fs.getNode("distortion")
            if not d_node.empty():
                D = d_node.mat().flatten()
            else:
                D = np.zeros((4,), dtype=float)
            fs.release()
            return cam_m, D, True
        fs.release()

    # fallback: use calibrated default if img_shape provided else safe identity-like
    if img_shape is None:
        K = np.eye(3, dtype=float)
        K[0, 0] = K[1, 1] = 500.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        D = np.zeros((4,), dtype=float)
        return K, D, False

    # Hardcoded calibrated fallback (replace with your calibration if available)
    K = np.array([
        [1296.6667, 0.0, 501.5039],
        [0.0, 1300.8313, 276.1617],
        [0.0, 0.0, 1.0]
    ], dtype=float)
    D = np.array([-0.065499670739455, 0.052973131052699, 0.0, 0.0], dtype=float)
    return K, D, False

