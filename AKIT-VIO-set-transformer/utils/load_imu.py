import os
import glob
import numpy as np
import torch
from PIL import Image
import pandas as pd
import cv2

def load_imu_txt(path="imu/imu.txt", gyro_in_deg=False):
    df = pd.read_csv(path, comment='#', header=None)
    df.columns = [
        "timestamp",
        "wx", "wy", "wz",
        "ax", "ay", "az"
    ]
    imu_vals = df[["wx", "wy", "wz", "ax", "ay", "az"]].values.astype(np.float64)
    if gyro_in_deg:
        imu_vals[:, 0:3] = imu_vals[:, 0:3] * (np.pi / 180.0)
    imu_data = torch.tensor(imu_vals.astype(np.float32))
    timestamps = torch.tensor(df["timestamp"].values.astype(np.float64))
    return imu_data, timestamps


def get_imu_between(imu_ts, imu_data, t_start, t_end):
    # imu_ts and t_start/t_end are seconds (float or tensor)
    mask = (imu_ts >= t_start) & (imu_ts <= t_end)
    return imu_data[mask], imu_ts[mask]


# -------------------------
# Rotation estimate util
# -------------------------
def estimate_rotation_from_imu(ts0, ts1, imu_ts_np, imu_data_np):
    idx = np.where((imu_ts_np >= ts0) & (imu_ts_np <= ts1))[0]
    if len(idx) < 2:
        return 0.0
    w = imu_data_np[idx, 0:3]  # rad/s
    ts = imu_ts_np[idx]
    ang = 0.0
    for j in range(len(idx) - 1):
        dt = ts[j + 1] - ts[j]
        wm = 0.5 * (np.linalg.norm(w[j]) + np.linalg.norm(w[j + 1]))
        ang += wm * dt
    return float(ang)