#innovation.py
import numpy as np
import torch
import cv2

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
    if isinstance(kp1, torch.Tensor):
        kp1_np = kp1.detach().cpu().numpy()
    else:
        kp1_np = np.asarray(kp1)
    if isinstance(kp2, torch.Tensor):
        kp2_np = kp2.detach().cpu().numpy()
    else:
        kp2_np = np.asarray(kp2)

    assert kp1_np.ndim == 2 and kp2_np.ndim == 2
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

def triangulate_points_world(pts1, pts2, R1, t1, R2, t2, K):
    """
    pts1, pts2 : Nx2 pixel coordinates
    R1,t1      : cam1 → world
    R2,t2      : cam2 → world
    """

    # World → camera projection
    P1 = K @ np.hstack((R1.T, -R1.T @ t1.reshape(3,1)))
    P2 = K @ np.hstack((R2.T, -R2.T @ t2.reshape(3,1)))

    pts1_h = pts1.T
    pts2_h = pts2.T

    pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    Pw = (pts4d[:3] / (pts4d[3] + 1e-12)).T

    # Cheirality check (in cam1 frame)
    Xc1 = (R1.T @ (Pw.T - t1.reshape(3,1))).T
    valid = Xc1[:,2] > 0.1

    return Pw[valid], pts2[valid]




# ------------------------------
# Project 3D points using predicted pose and return pixel residuals
# ------------------------------
def project_points(K, pts3d_cam):
    """
    Project camera-frame 3D points directly.
    pts3d_cam: (N,3) in CAMERA frame
    """
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()

    X = pts3d_cam[:, 0]
    Y = pts3d_cam[:, 1]
    Z = pts3d_cam[:, 2] + 1e-12

    u = K[0,0] * (X / Z) + K[0,2]
    v = K[1,1] * (Y / Z) + K[1,2]

    return np.stack([u, v], axis=1)


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
    preds_uv = project_points(K,pts3d_inliers)
    # Shapes MUST match: both (N_inliers,2)
    innovation = obs2u_inliers - preds_uv
    print("Z stats:", 
      pts3d_inliers[:,2].min(),
      pts3d_inliers[:,2].max())

    return innovation, obs2u_inliers, pts3d_inliers, {
        "matches": matches,
        "pose_rel": pose_rel,
        "preds_uv": preds_uv,
        "inliers": inlier_mask
    }

def linear_multiview_triangulation(
    pts,
    Ps,
    K,
    cond_thresh=1e8,
    min_parallax_deg=0.5,
    max_reproj_err=3.0
):
    if len(pts) < 2:
        return None, "too_few_views"

    Kinv = np.linalg.inv(K)

    # --- Bearing vectors ---
    bearings = []
    for (u, v) in pts:
        xn = Kinv @ np.array([u, v, 1.0])
        bearings.append(xn / np.linalg.norm(xn))

    # --- Parallax selection ---
    valid_idx = []
    for i in range(len(bearings)):
        for j in range(i + 1, len(bearings)):
            cos_angle = np.dot(bearings[i], bearings[j])
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            if angle > np.deg2rad(min_parallax_deg):
                valid_idx.append(i)
                valid_idx.append(j)

    valid_idx = sorted(set(valid_idx))
    if len(valid_idx) < 2:
        return None, "low_parallax"

    # --- Build normalized A ---
    A = []
    for i in valid_idx:
        u, v = pts[i]
        P = Ps[i]
        xn = Kinv @ np.array([u, v, 1.0])

        r1 = xn[0] * P[2] - P[0]
        r2 = xn[1] * P[2] - P[1]

        r1 /= np.linalg.norm(r1)
        r2 /= np.linalg.norm(r2)

        A.append(r1)
        A.append(r2)

    A = np.asarray(A)

    # --- SVD + conditioning ---
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    if S[-1] < 1e-12:
        return None, "rank_deficient"

    cond = S[0] / S[-1]
    if cond > cond_thresh:
        return None, "ill_conditioned"

    X = Vt[-1]
    Pw = X[:3] / X[3]

    # --- Cheirality + reprojection ---
    reproj_errs = []
    for (u, v), P in zip(pts, Ps):
        Pc = P @ np.hstack((Pw, 1.0))
        if Pc[2] <= 0:
            return None, "behind_camera"

        proj = (K @ (Pc[:3] / Pc[2]))[:2]
        reproj_errs.append(np.linalg.norm(proj - np.array([u, v])))

    if np.mean(reproj_errs) > max_reproj_err:
        return None, "high_reprojection"

    # --- Depth sanity ---
    depths = [(P @ np.hstack((Pw, 1.0)))[2] for P in Ps]
    if not (0.1 < np.mean(depths) < 1e3):
        return None, "bad_depth"

    return Pw, "ok"


def reprojection_residual(Pw, Rk, pk, z, K):
    Pc = Rk @ (Pw - pk)

    if Pc[2] <= 0:
        return None, None

    z_hat = (K @ (Pc / Pc[2]))[:2]
    r = z - z_hat

    return r, Pc


def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def reprojection_jacobian(Pc, K):
    X, Y, Z = Pc
    fx, fy = K[0,0], K[1,1]

    J_proj = np.array([
        [fx/Z, 0, -fx*X/(Z*Z)],
        [0, fy/Z, -fy*Y/(Z*Z)]
    ])

    J_se3 = np.hstack((np.eye(3), -skew(Pc)))
    return J_proj @ J_se3
