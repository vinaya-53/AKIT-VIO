#geometry.py
import numpy as np
import cv2

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    
def match_orb(kp1, des1, kp2, des2, ratio=0.9):
    # Safety checks
    if des1 is None or des2 is None:
        return None

    if len(des1) < 2 or len(des2) < 2:
        return None

    # ORB descriptors must be uint8
    if des1.dtype != np.uint8:
        des1 = des1.astype(np.uint8)
    if des2.dtype != np.uint8:
        des2 = des2.astype(np.uint8)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    pts1, pts2, idx_pairs = [], [], []

    for m, n in matches:
        if m.distance < ratio * n.distance:
            pts1.append(kp1[m.queryIdx])
            pts2.append(kp2[m.trainIdx])
            idx_pairs.append((m.queryIdx, m.trainIdx))  # 🔑 store indices

    if len(pts1) < 8:
        return None

    return np.asarray(pts1, dtype=np.float64), np.asarray(pts2, dtype=np.float64), idx_pairs



def triangulate_points(kp1, kp2, R1, p1, R2, p2, K):
    """
    Args:
        kp1, kp2: Keypoints from image 1 and 2 (Nx2)
        R1, p1: Camera 1 Pose (Rotation and Position in World)
        R2, p2: Camera 2 Pose (Rotation and Position in World)
        K: Camera Intrinsics (3x3)
    """
    # Create World-to-Camera 1 Projection Matrix
    R_c1_w = R1.T
    t_c1_w = -R_c1_w @ p1.reshape(3, 1)
    P1 = K @ np.hstack((R_c1_w, t_c1_w))

    # Create World-to-Camera 2 Projection Matrix
    R_c2_w = R2.T
    t_c2_w = -R_c2_w @ p2.reshape(3, 1)
    P2 = K @ np.hstack((R_c2_w, t_c2_w))

    # Triangulate using OpenCV (expects points as 2xN)
    pts4 = cv2.triangulatePoints(
        P1.astype(np.float64), 
        P2.astype(np.float64),
        kp1.T.astype(np.float64),
        kp2.T.astype(np.float64)
    )

    # Convert from Homogeneous to 3D World Coordinates
    Pw = (pts4[:3] / (pts4[3] + 1e-12)).T
    return Pw

def project_to_so3(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def reprojection_residuals(Pw, kp, R, p, K):
    y = []

    for X, z in zip(Pw, kp):
        Xc = R.T @ (X - p)
        if Xc[2] <= 0:
            continue

        uv = K @ (Xc / Xc[2])
        y.append(z - uv[:2])

    if len(y) == 0:
        return None

    return np.concatenate(y)

def build_reprojection_residual_and_jacobian(Pw, kp, x, K):
    '''Pw : Nx3 world points
    kp : Nx2 observed pixels
    x  : predicted EKF state (pose)
    '''

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    Rcw, tcw = x.R, x.t   # world → camera

    y, H = [], []

    for Xw, z in zip(Pw, kp):

        # ---------- WORLD → CAMERA ----------
        Xc = Rcw @ (Xw - tcw)
        X, Y, Z = Xc

        if Z <= 0.1:
            continue

        # ---------- Projection ----------
        u = fx * X / Z + cx
        v = fy * Y / Z + cy

        # ---------- Innovation ----------
        y.append([z[0] - u, z[1] - v])

        # ---------- Jacobian ----------
        J_proj = np.array([
            [fx/Z,      0, -fx*X/(Z*Z)],
            [0,      fy/Z, -fy*Y/(Z*Z)]
        ])

        J_se3 = np.hstack((-Rcw, Rcw @ skew(Xw - tcw)))
        H.append(J_proj @ J_se3)

    if len(y) == 0:
        return None, None

    y = np.array(y).reshape(-1,1)
    H = np.vstack(H)

    print("mean |r|:", np.mean(np.linalg.norm(y.reshape(-1,2), axis=1)))

    return y, H
def get_interpolated_imu(imu_ts, imu_data, t_start, t_end):
    """
    Always returns IMU samples spanning [t_start, t_end]
    with interpolated boundary samples.
    """
    if t_end <= t_start:
        return None, None

    # 1. Samples strictly inside
    mask = (imu_ts > t_start) & (imu_ts < t_end)
    ts_inside = imu_ts[mask]
    data_inside = imu_data[mask]

    def lerp(t, t0, t1, d0, d1):
        alpha = (t - t0) / (t1 - t0)
        return d0 + alpha * (d1 - d0)

    # ---- interpolate start ----
    idx1 = np.searchsorted(imu_ts, t_start)
    if idx1 == 0 or idx1 >= len(imu_ts):
        return None, None

    t0, t1 = imu_ts[idx1 - 1], imu_ts[idx1]
    d0, d1 = imu_data[idx1 - 1], imu_data[idx1]
    imu_start = lerp(t_start, t0, t1, d0, d1)

    # ---- interpolate end ----
    idx2 = np.searchsorted(imu_ts, t_end)
    if idx2 == 0 or idx2 >= len(imu_ts):
        return None, None

    t0, t1 = imu_ts[idx2 - 1], imu_ts[idx2]
    d0, d1 = imu_data[idx2 - 1], imu_data[idx2]
    imu_end = lerp(t_end, t0, t1, d0, d1)

    # ---- combine ----
    final_ts = np.concatenate(([t_start], ts_inside, [t_end]))
    final_data = np.vstack(([imu_start], data_inside, [imu_end]))

    return final_data, final_ts


# ================= INITIAL GRAVITY LEVELING =================

def get_initial_rotation(gravity_body):
    # 1. Force the input into a 1D array of shape (3,)
    gb = np.array(gravity_body).flatten() 
    
    if gb.shape[0] != 3:
        raise ValueError(f"Expected 3 accel values, got {gb.shape[0]}")

    print("-" * 30)
    print(f"[INIT] Raw Accel Mean: {gb}")
    accel_norm = np.linalg.norm(gb)
    print(f"[INIT] Accel Magnitude: {accel_norm:.4f} m/s^2")

    # 2. Normalize to get the z-world basis vector (direction of gravity in body frame)
    zw = gb / accel_norm
    
    # 3. Pick temp axis to create a coordinate system
    if np.abs(zw[0]) < 0.9:
        temp_axis = np.array([0, 1, 0])
    else:
        temp_axis = np.array([0, 0, 1])
        
    # 4. Gram-Schmidt process to create orthogonal basis
    xw = np.cross(temp_axis, zw)
    xw /= np.linalg.norm(xw)
    
    yw = np.cross(zw, xw)
    yw /= np.linalg.norm(yw)
    
    # 5. Build R_wb (Body-to-World)
    R_wb = np.column_stack((xw, yw, zw))
    
    # --- DIAGNOSTIC VERIFICATION ---
    # Verification A: Orthogonality (R^T * R should be Identity)
    ortho_check = np.linalg.norm(np.eye(3) - R_wb.T @ R_wb)
    
    # Verification B: Gravity Alignment
    # In World frame, gravity should be [0, 0, norm]
    # R_bw @ gravity_body should result in [0, 0, magnitude]
    g_world_check = R_wb.T @ gb 
    
    print(f"[INIT] Orthogonality Error: {ortho_check:.2e}")
    print(f"[INIT] Gravity in World Frame (check): {g_world_check}")
    print(f"[INIT] Initial R_wb:\n{R_wb}")
    print("-" * 30)
    
    return R_wb

def gravity_alignment_rotation(accel_mean):
    """
    Compute rotation R such that:
        R @ accel_mean ≈ [0, 0, |g|]
    Assumes accel_mean measured while device is static.
    """

    # Normalize measured gravity
    g_meas = accel_mean / np.linalg.norm(accel_mean)

    # World gravity direction (+Z up)
    g_world = np.array([0.0, 0.0, 1.0])

    # Axis of rotation
    v = np.cross(g_meas, g_world)
    s = np.linalg.norm(v)
    c = np.dot(g_meas, g_world)

    # Handle already-aligned case
    if s < 1e-8:
        return np.eye(3)

    # Skew-symmetric matrix
    vx = np.array([
        [ 0.0, -v[2],  v[1]],
        [ v[2],  0.0, -v[0]],
        [-v[1],  v[0],  0.0]
    ])

    # Rodrigues rotation formula
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    # Ensure numerical orthonormality
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    return R
