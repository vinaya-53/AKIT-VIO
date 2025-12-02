# main.py
import os
import glob
import numpy as np
import torch
from PIL import Image
import pandas as pd
import cv2

from model.visual_pipeline import VisualPipeline
from model.imu_extractor import IMUFeatureExtractor
from model.keypoint_extractor import ORBKeypointExtractor, SimpleDenseDescriptor
from model.innovation import (
    compute_innovation_from_triangulation,
    match_orb_descriptors,
    undistort_fisheye_points
)


# -------------------------
# Data loading utilities
# -------------------------
def load_tiff_images(folder="images/", max_images=50):
    image_paths = glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.tiff"))
    if len(image_paths) == 0:
        raise RuntimeError("No images found in " + folder)

    def extract_ts(path):
        name = os.path.basename(path)
        ts_raw = int(name.split("_")[0])
        return ts_raw

    image_paths = sorted(image_paths, key=extract_ts)[:max_images]

    # detect magnitude of raw timestamps and convert to seconds
    first_raw = extract_ts(image_paths[0])
    digits = len(str(first_raw))
    if digits >= 18:
        scale = 1e9   # nanoseconds -> seconds
    elif digits >= 15:
        scale = 1e6   # microseconds -> seconds
    elif digits >= 12:
        scale = 1e3   # milliseconds -> seconds
    else:
        scale = 1.0   # already seconds

    images = []
    timestamps = []
    for path in image_paths:
        fname = os.path.basename(path)
        ts_raw = int(fname.split("_")[0])
        ts_sec = ts_raw / scale
        timestamps.append(ts_sec)
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.tensor(arr).permute(2, 0, 1)  # [C,H,W]
        images.append(tensor)

    images = torch.stack(images)  # [N, 3, H, W]
    timestamps = torch.tensor(timestamps, dtype=torch.float64)
    return images, timestamps


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


# -------------------------
# Average keypoint displacement (helper)
# -------------------------
def avg_keypoint_displacement_from_lists(kpA, descA, kpB, descB, K, D, ratio_test=0.9):
    if torch.is_tensor(kpA):
        kpA = kpA.detach().cpu().numpy()
    if torch.is_tensor(kpB):
        kpB = kpB.detach().cpu().numpy()
    if torch.is_tensor(descA):
        descA = descA.detach().cpu().numpy()
    if torch.is_tensor(descB):
        descB = descB.detach().cpu().numpy()

    kpA_s = kpA[0] if (kpA.ndim == 3 and kpA.shape[0] == 1) else kpA.reshape(-1, 2)
    kpB_s = kpB[0] if (kpB.ndim == 3 and kpB.shape[0] == 1) else kpB.reshape(-1, 2)
    d1 = descA[0] if (descA.ndim == 3 and descA.shape[0] == 1) else descA.reshape(-1, descA.shape[-1])
    d2 = descB[0] if (descB.ndim == 3 and descB.shape[0] == 1) else descB.reshape(-1, descB.shape[-1])

    matches = match_orb_descriptors(d1, d2, cross_check=True, ratio_test=ratio_test)
    if len(matches) == 0:
        return 0.0, 0
    ptsA = np.array([kpA_s[m.queryIdx] for m in matches])
    ptsB = np.array([kpB_s[m.trainIdx] for m in matches])

    try:
        ptsA_u = undistort_fisheye_points(ptsA, K, D)
        ptsB_u = undistort_fisheye_points(ptsB, K, D)
    except Exception:
        ptsA_u, ptsB_u = ptsA, ptsB

    disp = np.linalg.norm(ptsA_u - ptsB_u, axis=1)
    return float(np.mean(disp)), len(matches)


# -------------------------
# Camera intrinsics loader (module-level)
# -------------------------
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


# -------------------------
# Main processing
# -------------------------
def main(
    img_folder="images/",
    imu_path="imu/imu.txt",
    max_images=50,
    orb_max_kpts=800,
    window_size=5,
    min_matches=8,
    parallax_thresh_px=3.0
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # networks / extractors
    vis_pipeline = VisualPipeline(feat_dim=256).to(device)
    imu_net = IMUFeatureExtractor(imu_feat_dim=128).to(device)
    orb = ORBKeypointExtractor(max_keypoints=orb_max_kpts)
    dense_net = SimpleDenseDescriptor(out_dim=64).to(device)

    # load data
    images, img_ts = load_tiff_images(img_folder, max_images=max_images)
    imu_data, imu_ts = load_imu_txt(imu_path, gyro_in_deg=False)

    # align camera timestamps to IMU reference by simple offset (first-sample)
    offset = float(imu_ts[0].item() - img_ts[0].item())
    img_ts_aligned = img_ts + offset

    # extract visual features for all frames (one pass)
    kpt_coords_list = []
    kpt_descs_list = []
    kpt_mask_list = []
    dense_desc_list = []

    for img in images:
        img_b = img.unsqueeze(0).to(device)  # [1,3,H,W]
        enhanced, vis_feat = vis_pipeline(img_b)

        enhanced_np = (enhanced[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        coords, descs, mask = orb(enhanced_np)
        kpt_coords_list.append(coords)   # torch tensors (1,K,2)
        kpt_descs_list.append(descs)
        kpt_mask_list.append(mask)

        dense_desc = dense_net(enhanced.to(device), coords.to(device))
        dense_desc_list.append(dense_desc.detach().cpu())

    # load intrinsics
    img_shape = images.shape[1:]
    K, D, loaded = load_camera_intrinsics("camera_intrinsics.yml", img_shape=img_shape)
    if not loaded:
        print("⚠ Intrinsics file missing: falling back to default calibrated values.")

    # initial pose placeholders
    R_pred = np.eye(3)
    p_pred = np.zeros(3)

    N = len(images)
    # sliding windows
    for i in range(0, N - (window_size - 1)):
        first = i
        last = i + (window_size - 1)

        # quick parallax check between first and last (avg displacement)
        avg_disp, nmatches = avg_keypoint_displacement_from_lists(
            kpt_coords_list[first], kpt_descs_list[first],
            kpt_coords_list[last], kpt_descs_list[last],
            K, D, ratio_test=0.9
        )
        if nmatches < min_matches or avg_disp < parallax_thresh_px:
            continue

        # slice IMU between the two aligned timestamps
        t0 = float(img_ts_aligned[first].item())
        t1 = float(img_ts_aligned[last].item())
        imu_seg, imu_seg_ts = get_imu_between(imu_ts, imu_data, t0, t1)
        imu_seg_np = imu_seg.numpy() if len(imu_seg) > 0 else np.zeros((0, 6))
        imu_seg_ts_np = imu_seg_ts.numpy() if len(imu_seg_ts) > 0 else np.zeros((0,))

        rot_imu = estimate_rotation_from_imu(t0, t1, imu_seg_ts_np, imu_seg_np) if len(imu_seg_ts_np) > 1 else 0.0
        imu_weight = float(np.clip(rot_imu * 10.0, 0.1, 1.0))

        # compute innovation using triangulation between first and last
        innov, obs2, pts3d, info = compute_innovation_from_triangulation(
            kpt_coords_list[first],
            kpt_coords_list[last],
            kpt_descs_list[first],
            kpt_descs_list[last],
            K, R_pred, p_pred,
            min_matches=min_matches,
            ratio_test=True
        )

        # apply IMU-based weight
        if innov.size != 0:
            innov = innov * imu_weight

        print(f"Window {first}->{last}: matches={nmatches}, avg_disp={avg_disp:.2f}px, imu_samples={len(imu_seg)}, rot_imu={rot_imu:.4f}, innov_shape={innov.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
