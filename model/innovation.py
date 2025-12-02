# models/innovation.py
import numpy as np
import torch
import cv2

# ------------------------------
# Helper: match ORB descriptors
# ------------------------------
def match_orb_descriptors(desc1, desc2, cross_check=True, ratio_test=0.95):
    """
    Match ORB descriptors using BFMatcher + ratio test.
    desc1, desc2 expected shapes: [K, 32] or [1,K,32]
    ratio_test: float (0.0-1.0). If None or <=0, ratio test is skipped.
    """
    # ---- Ensure descriptors are numpy uint8 ----
    if torch.is_tensor(desc1):
        desc1 = desc1.squeeze(0).cpu().numpy()
    if torch.is_tensor(desc2):
        desc2 = desc2.squeeze(0).cpu().numpy()

    if desc1 is None or desc2 is None or desc1.size == 0 or desc2.size == 0:
        return []

    # MUST convert to uint8 for ORB matcher (safe cast)
    desc1 = desc1.astype(np.uint8)
    desc2 = desc2.astype(np.uint8)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # KNN match (may return lists with <2 neighbors for small sets)
    knn = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in knn:
        # pair can be [] or [m] or [m,n]
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if ratio_test is None or ratio_test <= 0:
            # accept all (or rely on cross-check later)
            good.append(m)
        else:
            if float(m.distance) < float(ratio_test) * float(n.distance):
                good.append(m)

    # As a fallback, if no matches found and cross_check==True, try crossCheck mode
    if len(good) == 0 and cross_check:
        bf2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            plain = bf2.match(desc1, desc2)
            good = sorted(plain, key=lambda x: x.distance)
        except Exception:
            good = []

    return good




# ------------------------------
# Triangulation for two views
# ------------------------------
def triangulate_two_views(kp1, kp2, K, min_matches=20, reproj_threshold=3.0):
    """
    Triangulate matched keypoints between two frames.
    Inputs:
        kp1: keypoint coords in image 1, shape (N1, 2) numpy/torch (x,y)
        kp2: keypoint coords in image 2, shape (N2, 2)
        K: camera intrinsics (3x3) numpy or torch
    Returns:
        pts3d_world: (M,3) numpy array of triangulated points w.r.t. frame1 camera (or world if you choose)
        obs1: (M,2) image coords in frame1 (observations)
        obs2: (M,2) image coords in frame2
        mask: boolean mask length M (True = valid)
        pose_rel: (R, t) relative pose from frame1 -> frame2 (R: 3x3, t: 3x1)
    Notes:
        - Uses findEssentialMat + recoverPose + triangulatePoints.
        - pts are returned in homogeneous coordinates converted to 3D (camera1 frame).
    """
    if isinstance(kp1, torch.Tensor):
        kp1_np = kp1.detach().cpu().numpy()
    else:
        kp1_np = np.asarray(kp1)
    if isinstance(kp2, torch.Tensor):
        kp2_np = kp2.detach().cpu().numpy()
    else:
        kp2_np = np.asarray(kp2)

    assert kp1_np.ndim == 2 and kp2_np.ndim == 2

    # match between two frames using simple brute-force Hamming if descriptors provided externally.
    # Here we assume the caller passed already-matched pairs (kp coords aligned) OR used descriptors to get indices.
    # For convenience, if lengths equal, try to align by index.
    # The proper way is to pass matched index pairs; this helper focuses on geometry given matched pairs.

    # Convert K to numpy
    K_np = K.detach().cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K)

    # Use essential matrix to get relative pose.
    # We need matched points arrays of shape (N,2); use RANSAC to be robust.
    if kp1_np.shape[0] < min_matches or kp2_np.shape[0] < min_matches:
        return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,), dtype=bool), (np.eye(3), np.zeros((3,1)))

    E, mask = cv2.findEssentialMat(kp1_np, kp2_np, cameraMatrix=K_np, method=cv2.RANSAC, prob=0.999, threshold=reproj_threshold)
    if E is None:
        return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,), dtype=bool), (np.eye(3), np.zeros((3,1)))

    # recover relative pose (up to scale)
    _, R, t, mask_pose = cv2.recoverPose(E, kp1_np, kp2_np, cameraMatrix=K_np, mask=mask)

    # build projection matrices for triangulation: P1 = K [I | 0], P2 = K [R | t]
    P1 = K_np @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K_np @ np.hstack((R, t))

    # Triangulate matched points (use only inliers)
    inlier_idx = (mask.ravel() > 0)
    if np.count_nonzero(inlier_idx) < min_matches:
        return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,), dtype=bool), (R, t)

    pts1_in = kp1_np[inlier_idx]
    pts2_in = kp2_np[inlier_idx]

    # Need points as homogeneous (2xN)
    pts1_h = pts1_in.T
    pts2_h = pts2_in.T

    pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)  # 4xN
    pts3d = (pts4d[:3, :] / (pts4d[3, :] + 1e-12)).T  # N x 3

    obs1 = pts1_in  # (N,2)
    obs2 = pts2_in  # (N,2)

    return pts3d, obs1, obs2, np.ones((pts3d.shape[0],), dtype=bool), (R, t)


# ------------------------------
# Project 3D points using predicted pose and return pixel residuals
# ------------------------------
def project_points(K, R_pred, p_pred, pts3d_world):
    """
    Project points into camera frame given predicted pose.
    Inputs:
        K: 3x3 intrinsics (numpy or torch)
        R_pred: 3x3 rotation matrix (world -> camera) or camera orientation
        p_pred: 3-vector camera position in world coords
        pts3d_world: (N,3) numpy array of 3D points in world coordinates
    Returns:
        preds_uv: (N,2) numpy projected pixel coords
    Notes:
        This function assumes pts3d_world are in the same global frame as p_pred.
    """
    # convert to numpy
    if isinstance(K, torch.Tensor):
        K_np = K.detach().cpu().numpy()
    else:
        K_np = np.asarray(K)
    if isinstance(R_pred, torch.Tensor):
        R_np = R_pred.detach().cpu().numpy()
    else:
        R_np = np.asarray(R_pred)
    if isinstance(p_pred, torch.Tensor):
        p_np = p_pred.detach().cpu().numpy()
    else:
        p_np = np.asarray(p_pred)

    # Transform pts to camera frame: X_cam = R * (X_world - p)
    X_cam = (R_np @ (pts3d_world.T - p_np.reshape(3, 1))).T  # N x 3

    # Perspective project
    x = X_cam[:, 0]
    y = X_cam[:, 1]
    z = X_cam[:, 2] + 1e-12

    u = K_np[0, 0] * (x / z) + K_np[0, 2]
    v = K_np[1, 1] * (y / z) + K_np[1, 2]

    preds = np.stack([u, v], axis=1)
    return preds

def undistort_fisheye_points(pts, K, D):
    pts = pts.reshape(-1, 1, 2).astype(np.float32)
    und = cv2.fisheye.undistortPoints(pts, K, D)
    und = und.reshape(-1, 2)
    und[:, 0] = und[:, 0] * K[0, 0] + K[0, 2]
    und[:, 1] = und[:, 1] * K[1, 1] + K[1, 2]
    return und

# ------------------------------
# Compute innovation vector y = z_obs - h(x)
# ------------------------------
def compute_innovation_from_triangulation(kp1_coords, kp2_coords, kp1_desc, kp2_desc, 
                                         K, R_pred, p_pred,
                                         min_matches=30, ratio_test=True):

    # Convert torch → numpy
    if isinstance(kp1_coords, torch.Tensor):
        kp1_coords = kp1_coords.detach().cpu().numpy()
    if isinstance(kp2_coords, torch.Tensor):
        kp2_coords = kp2_coords.detach().cpu().numpy()
    if isinstance(kp1_desc, torch.Tensor):
        kp1_desc = kp1_desc.detach().cpu().numpy()
    if isinstance(kp2_desc, torch.Tensor):
        kp2_desc = kp2_desc.detach().cpu().numpy()

    # Squeeze shapes (1,K,2) → (K,2)
    kp1_coords_s = kp1_coords[0] if kp1_coords.ndim == 3 and kp1_coords.shape[0] == 1 else kp1_coords.reshape(-1, 2)
    kp2_coords_s = kp2_coords[0] if kp2_coords.ndim == 3 and kp2_coords.shape[0] == 1 else kp2_coords.reshape(-1, 2)

    # Descriptors
    desc1 = kp1_desc[0] if kp1_desc.ndim == 3 and kp1_desc.shape[0] == 1 else kp1_desc.reshape(-1, kp1_desc.shape[-1])
    desc2 = kp2_desc[0] if kp2_desc.ndim == 3 and kp2_desc.shape[0] == 1 else kp2_desc.reshape(-1, kp2_desc.shape[-1])

    # Match descriptors
    matches = match_orb_descriptors(desc1, desc2, cross_check=True, ratio_test=ratio_test)

    if len(matches) < min_matches:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 3)), {
            "matches": matches
        }

    # Build matched 2D pixel coord arrays
    pts1 = np.array([kp1_coords_s[m.queryIdx] for m in matches])
    pts2 = np.array([kp2_coords_s[m.trainIdx] for m in matches])

    # Fisheye (equidistant) distortion parameters
    D = np.array([
        -0.065499670739455,
         0.052973131052699,
         0.0,
         0.0
    ])

    # Undistort BOTH sets of matched keypoints
    pts1_und = undistort_fisheye_points(pts1, K, D)
    pts2_und = undistort_fisheye_points(pts2, K, D)

    # ---- TRIANGULATION (returns UNDISTORTED observations) ----
    pts3d, obs1u, obs2u, inlier_mask, pose_rel = triangulate_two_views(
        pts1_und, pts2_und, K
    )

    if pts3d.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 3)), {
            "matches": matches,
            "pose_rel": pose_rel
        }

    # Filter only triangulation inliers
    obs2u_inliers = obs2u[inlier_mask]        # (N_inliers,2)
    pts3d_inliers = pts3d                     # already filtered inside triangulate adapter

    # Predict projection of the 3D points into frame2
    preds_uv = project_points(K, R_pred, p_pred, pts3d_inliers)

    # Shapes MUST match: both (N_inliers,2)
    innovation = obs2u_inliers - preds_uv

    return innovation, obs2u_inliers, pts3d_inliers, {
        "matches": matches,
        "pose_rel": pose_rel,
        "preds_uv": preds_uv,
        "inliers": inlier_mask
    }
