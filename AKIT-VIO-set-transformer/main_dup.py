import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from HDF5_Logger.vio_hdf5_logger import VIOHDF5Logger
from model.visual_pipeline import VisualPipeline
from model.keypoint_extractor import ORBKeypointExtractor, SimpleDenseDescriptor
from model.ekf.ekf_se3 import EKFSE3, State, NoiseParams
from utils.load_img import load_tiff_images
from utils.load_imu import load_imu_txt, get_imu_between
from utils.camera_intrinsics import load_camera_intrinsics
from model.ekf.noise import NoiseParams
from utils.innovation import reprojection_residual, reprojection_jacobian
from utils.innovation import linear_multiview_triangulation
from utils.geometry import (match_orb,get_interpolated_imu,get_initial_rotation,gravity_alignment_rotation)
from HDF5_Logger.txt_logger import TextLogger
try:
    import rospy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, Image # Added Image
    from cv_bridge import CvBridge         # Added for Gazebo -> OpenCV conversion
    gazebo_enabled = True
except ImportError:
    gazebo_enabled = False
    print("ROS/Gazebo not detected. Skipping simulation publishing.")
# This is T_ic (Camera to IMU/Body)
R_bc = np.array([
    [-0.31825765, -0.94771498,  0.02341775],
    [ 0.94669219, -0.316421  ,  0.06042894],
    [-0.04985954,  0.04140137,  0.99789777]
])
t_bc = np.array([-0.00267583, -0.01257466, 0.02344868])

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def main(img_folder="images/", imu_path="imu/imu.txt", max_images=50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    visual_pipe = VisualPipeline(feat_dim=256).to(device).eval()
    orb_extractor = ORBKeypointExtractor(max_keypoints=2000)

    images, img_ts = load_tiff_images(img_folder, max_images=max_images)
    imu_data, imu_ts = load_imu_txt(imu_path)
    print(f"Loaded {len(images)} images, {imu_data.shape[0]} IMU samples")
    if img_ts[0] > 1e10: # Likely nanoseconds or high-res
        img_ts = img_ts / 10.0 # Adjust based on your specific log scale

    img_ts = img_ts + (imu_ts[0] - img_ts[0])
    if img_ts[-1] > imu_ts[-1]:
        print("⚠ Image timestamps exceed IMU range")
    K_cam, _, _ = load_camera_intrinsics(
        "camera_intrinsics.yml",
        images[0].shape[1:]
    )
    K_cam = K_cam[:3, :3].astype(np.float64)
    fx, fy, cx, cy = K_cam[0,0], K_cam[1,1], K_cam[0,2], K_cam[1,2]
    K4 = (fx, fy, cx, cy)
    x0 = State(R=np.eye(3), p=np.zeros(3), v=np.zeros(3), bg=np.zeros(3), ba=np.zeros(3))
    ekf = EKFSE3(x0, np.eye(15) * 0.01)
    noise = NoiseParams()
    
    # Use calibration gravity in IMU frame
    g_calib = np.array([-7.14713539, 4.12814959, -5.3021634])
    R_init = gravity_alignment_rotation(g_calib)
    ekf.x.R = R_init
    gravity_body = ekf.x.R @ g_calib
    print("Gravity in world frame after alignment:", gravity_body)
    print("Should be ~ [0,0,9.81]")

    print("Check R_init vs EKF rotation:\n", np.allclose(R_init, ekf.x.R))

    print("Camera z-axis:", R_bc[:3,2])  # cam0 → imu
    print("Gravity after alignment:", R_init @ g_calib)

    print("Initial EKF rotation aligned to gravity:\n", ekf.x.R)
    print("Gravity vector in body frame:", ekf.x.R.T @ g_calib)
    print("Body z-axis:", ekf.x.R[:,2])

    ekf.x.v[:]  = 0.0
    ekf.x.ba[:] = 0.0
    ekf.x.bg[:] = 0.0
    ekf.P = np.eye(15) * 1e-4
    
    # ---------- FEATURE EXTRACTION ----------
    kps, descs = [], []
    for i, img in enumerate(images):
        img_t = torch.from_numpy(img).float() if isinstance(img, np.ndarray) else img.float()
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0)
        img_t = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            enhanced, _ = visual_pipe(img_t)
            coords, desc, _ = orb_extractor(enhanced)
        pts = coords[0].cpu().numpy()
        des = desc[0].cpu().numpy()

        if pts.shape[0] < 20:
            print(f"[Frame {i}] Too few ORB points")
            kps.append(np.empty((0,2)))
            descs.append(None)
            continue

        kps.append(pts.astype(np.float32))
        descs.append(des.astype(np.uint8))
        if i < 5:
            print(f"[Frame {i}] ORB keypoints:", pts.shape[0])

    VISION_STRIDE = 6
    MIN_TRACK_LEN = 6
    consecutive_vision_skips = 0
    vision_success_pts = []
    last_r = None
    last_S = None
    n_matches = 0

    logger = VIOHDF5Logger("logs/vio_run_001.h5")
    txt_logger = TextLogger("logs/vio_results.txt")
    if gazebo_enabled:
        rospy.init_node('vio_sim', anonymous=True)
        odom_pub = rospy.Publisher("/vio/odom", Odometry, queue_size=10)
        imu_pub = rospy.Publisher("/vio/imu", Imu, queue_size=10)
        print("Gazebo ROS publishers initialized.")

    # ---------- MSCKF FEATURE TRACKS ----------
    feature_tracks = {}        
    feature_id_map = {}       
    next_feature_id = 0
    last_vision_p = ekf.x.p.copy()
    last_vision_r = ekf.x.R.copy()
    consecutive_vision_skips = 0
    
    if torch.is_tensor(imu_data):
        initial_accels_raw = imu_data[:20, 3:6].detach().cpu().numpy()
    else:
        initial_accels_raw = imu_data[:20, 3:6]
    raw_accel_mean = np.mean(initial_accels_raw, axis=0)

    gravity_dir = raw_accel_mean / np.linalg.norm(raw_accel_mean)
    ekf.x.R = gravity_alignment_rotation(gravity_dir)
    ekf.x.ba[:] = 0.0          # start with ZERO bias
    ekf.x.bg[:] = 0.0
    raw_gyro_mean = imu_data[:20, 0:3].detach().cpu().numpy() if torch.is_tensor(imu_data) else imu_data[:20, 0:3]
    initial_bg = np.mean(raw_gyro_mean, axis=0)
    ekf.x.bg[:] = initial_bg

    print(f"Initial Gyro Bias (bg): {initial_bg}")
        
    # ========= EKF LOG STORAGE =========
    traj_world = []       # EKF position history
    traj_time = []        # timestamps
    P_trace_hist = []     # covariance trace
    vel_hist = []         # velocity magnitude

    # ================= MAIN LOOP =================
    for i in range(0, len(images) - VISION_STRIDE):
        print(f"\n===== FRAME {i} → {i + VISION_STRIDE} =====")

        R_prev = ekf.x.R.copy()
        p_prev = ekf.x.p.copy()
        v_prev = ekf.x.v.copy()
        
        # ---------- IMU PREDICTION ----------
        imu_seg, imu_ts_seg = get_interpolated_imu(
            imu_ts,
            imu_data,
            img_ts[i],
            img_ts[i + 1]
        )

        if imu_seg is None:
            print(f"[WARN] IMU interpolation failed at frame {i}")
            continue

        # COVARIANCE HEALTH CHECK
        imu_np = imu_seg if isinstance(imu_seg, np.ndarray) else imu_seg.detach().cpu().numpy()
        dt_sum = 0.0
        MAX_DT = 0.01        # 100 Hz integration
        MAX_FRAME_DT = 0.1   # drop protection

        for j in range(len(imu_ts_seg) - 1):
            dt = float(imu_ts_seg[j + 1] - imu_ts_seg[j])
            if dt > 0.5: 
                dt /= 1e9
            if dt <= 0.0 or dt > MAX_FRAME_DT:
                print(f"[IMU] Skipped dt={dt:.3f}s")
                continue

            steps = int(np.ceil(dt / MAX_DT))
            sub_dt = dt / steps

            omega = imu_np[j, 0:3]
            acc   = imu_np[j, 3:6]

            for _ in range(steps):
                ekf.predict(
                    omega=omega,
                    acc=acc,
                    dt=sub_dt,
                    Q=noise
                )
            dt_sum += dt
            print(f"Δp_imu={ekf.x.p - p_prev}")
            p_trace = np.trace(ekf.P)
            if not np.isfinite(ekf.P).all() or p_trace <= 0:
                print("🚨 EKF CRASHED (NaN/Non-PSD) → Emergency Reset")
                ekf.P = np.eye(15) * 0.1
                ekf.x.v *= 0.999
                continue

            # ---------- STATIONARY DETECTION (ZUPT) ----------
            acc_mag = np.linalg.norm(imu_np[-1, 3:6])
            if (9.75 <= acc_mag <= 9.87): # Check if near 1G
                ekf.x.v[:] = 0.0
                # Zero Velocity Update (Constrain covariance)
                H_zupt = np.zeros((3, 15)); H_zupt[:, 3:6] = np.eye(3)
                K_z = ekf.P @ H_zupt.T @ np.linalg.inv(H_zupt @ ekf.P @ H_zupt.T + np.eye(3)*1e-4)
                ekf.P = (np.eye(15) - K_z @ H_zupt) @ ekf.P
                print(f"[ZUPT] Active. Trace: {np.trace(ekf.P):.2f}")
            # Soft reset if covariance blows up
            if (not np.isfinite(ekf.P).all()) or (np.trace(ekf.P) <= 0):
                print("🚨 EKF covariance NaN detected → SOFT RESET")
                ekf.P[:] = np.eye(15) * 0.1
               # ekf.x.v[:] = 0.0
                ekf.x.bg[:] = 0.0
                ekf.x.ba[:] = 0.0
                baseline_accum = 0.0
                baseline_xy_accum = 0.0
                consecutive_vision_skips = 0
                continue
        print(f"Δp={ekf.x.p - p_prev}, Δv={ekf.x.v - v_prev}")
        print(dt_sum)
        # --- STABILIZED INFLATION ---
        p_trace = np.trace(ekf.P)
        if p_trace > 200.0:
            ekf.x.v *= 0.95
            inflation_factor = 1.05 if p_trace < 200.0 else 1.0
            ekf.P[0:9, 0:9] *= inflation_factor
            ekf.P[9:15, 9:15] *= 1.01 # Keep biases very stable
            ekf.P[3:6, 3:6] = np.eye(3) * min(ekf.P[3,3], 5.0) 
            ekf.P = 0.5 * (ekf.P + ekf.P.T)

        # ---------- COMPUTE BASELINE SINCE LAST VISION ----------
        delta_p_since_vision = ekf.x.p - last_vision_p
        baseline = np.linalg.norm(delta_p_since_vision)
        baseline_xy = np.linalg.norm(delta_p_since_vision[:2])
        baseline_ratio = baseline_xy / (baseline + 1e-6)
        
        # ---------- DECIDE IF VISION UPDATE SHOULD RUN ----------
        force_update = (consecutive_vision_skips >= 5)
        if baseline < 0.001 and not force_update:
            run_vision = False
            consecutive_vision_skips += 1
            print("[VISION] Skipped due to low baseline")
            continue

        if consecutive_vision_skips >= 3:
            print("⚠️ EMERGENCY: No vision updates. Damping velocity to zero.")
            ekf.x.v *= 0.8  # Friction-like damping
            ekf.P[3:6, 3:6] *= 1.1 # Increase velocity uncertainty

        acc_mag = np.linalg.norm(imu_np[-1, 3:6])
        stationary = (9.75 <= acc_mag <= 9.87) 

        if stationary:
            ekf.x.v[:] = 0.0
            H_zupt = np.zeros((3, 15))
            H_zupt[:, 3:6] = np.eye(3)
            R_zupt = np.eye(3) * 1e-4
            S = H_zupt @ ekf.P @ H_zupt.T + R_zupt
            K_zupt = ekf.P @ H_zupt.T @ np.linalg.inv(S)
            ekf.P = (np.eye(15) - K_zupt @ H_zupt) @ ekf.P
            print(f"[ZUPT] Velocity locked. P_trace: {np.trace(ekf.P):.3e}")
        #-------------gazebo----------------------
        if gazebo_enabled:
            for j in range(len(imu_ts_seg) - 1):
                omega = imu_np[j, 0:3]
                acc   = imu_np[j, 3:6]

                imu_msg = Imu()
                imu_msg.header.stamp = rospy.Time.now()
                imu_msg.angular_velocity.x = omega[0]
                imu_msg.angular_velocity.y = omega[1]
                imu_msg.angular_velocity.z = omega[2]
                imu_msg.linear_acceleration.x = acc[0]
                imu_msg.linear_acceleration.y = acc[1]
                imu_msg.linear_acceleration.z = acc[2]
                imu_pub.publish(imu_msg)
        # ---------- RUN VISION UPDATE ----------
        matches = []
        pts1 = None
        pts2 = None
        j = i + VISION_STRIDE
        if j >= len(img_ts): break
        result = match_orb(kps[i], descs[i], kps[j], descs[j])

        if result is None or len(result[2]) < 30:
            print(f"FAILED MATCH: {0 if result is None else len(result[2])} matches found.")
            consecutive_vision_skips += 1 # Critical: increment this
            continue
        pts1, pts2, matches = result

        # --- Fisheye distortion ---
        D = np.array(
            [-0.065499670739455,
            0.052973131052699,
            0.0,
            0.0],
            dtype=np.float64
        )
        
        K = np.asarray(K_cam, dtype=np.float64)
        K = K.reshape(3, 3)
        D = np.asarray(D, dtype=np.float64).reshape(4, 1)

        pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 1, 2)
        pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 1, 2)
        pts1_ud = cv2.fisheye.undistortPoints(
            pts1.reshape(-1, 1, 2), K, D
        )
        pts2_ud = cv2.fisheye.undistortPoints(
            pts2.reshape(-1, 1, 2), K, D
        )
        E, mask_e = cv2.findEssentialMat(
            pts1_ud, pts2_ud,
            np.eye(3),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1e-3
        )

        if E is None:
            print("Essential matrix failed")
            vision_success = 0
            continue
        points, R_cv, t_cv, mask_pose = cv2.recoverPose(
            E, pts1_ud, pts2_ud,cameraMatrix=np.eye(3),mask=mask_e
        )

        in_front_ratio = points / len(pts1_ud)
        print(f"[DEBUG] Points in front: {points}/{len(pts1_ud)}")


        if in_front_ratio < 0.5:
            print("❌ Reject: cheirality failure")
            consecutive_vision_skips += 1
            # If we miss too many, we must increase IMU noise to stay 'search' mode
            ekf.P[0:6, 0:6] *= 1.2
            vision_success = 0
            continue

        R_imu = R_bc @ R_cv 

        # COORDINATE ALIGNMENT & DIRECTION CHECK
        t_body = R_bc @ t_cv.flatten()
        dp_body = ekf.x.R.T @ (ekf.x.p - p_prev) 

        if np.linalg.norm(dp_body) > 1e-4:
            v_vision = t_body / (np.linalg.norm(t_body) + 1e-9)
            v_imu = dp_body / (np.linalg.norm(dp_body) + 1e-9)
            dot = np.dot(v_vision, v_imu)
            print(f"[DEBUG] Body-Frame Dot Product: {dot:.3f}")
            if dot < -0.5:
                print("⚠️ Direction Conflict: Applying 180-degree correction to Vision")
                t_cv = -t_cv
        # UPDATE FEATURE TRACKS (REWRITTEN FOR CONSISTENCY)
        for pid, qid in matches:
            key = (i, pid)
            if key not in feature_id_map:
                fid = next_feature_id
                next_feature_id += 1
                feature_id_map[key] = fid
                feature_tracks[fid] = []
            else:
                fid = feature_id_map[key]

            R_wc = ekf.x.R @ R_bc 
            p_wc = ekf.x.p + ekf.x.R @ t_bc 
            feature_tracks[fid].append(
                (j, kps[j][qid], R_wc.copy(), p_wc.copy())
            )
        H_stacked = []
        r_stacked = []

        for fid in list(feature_tracks.keys()):
            track = feature_tracks[fid]
            if len(track) < MIN_TRACK_LEN: continue
            pts_obs = [] 
            Ps = []      
            for frame_id, uv, Rk, pk in track:
                pts_obs.append(uv)
        
            P_mat = np.hstack((Rk.T, -Rk.T @ pk.reshape(3,1)))
            Ps.append(P_mat)
            Pw, tri_status = linear_multiview_triangulation(pts_obs, Ps, K_cam)
            if Pw is None or tri_status != "success":
                del feature_tracks[fid]; continue
            r, Pc = reprojection_residual(Pw, R_wc, p_wc, track[-1][1], K_cam)
            # GATING: Reject outliers before stacking
            if r is None or np.linalg.norm(r) > 20.0: # Pixel threshold
                del feature_tracks[fid]; continue

            # Calculate Jacobian for this feature
            J_p, J_theta = reprojection_jacobian(Pc, K_cam)
            H_f = np.zeros((2, 15))
            H_f[:, 0:3] = J_p @ ekf.x.R.T
            H_f[:, 6:9] = J_theta @ R_bc.T - H_f[:, 0:3] @ skew(ekf.x.R @ t_bc)

            # Accumulate
            H_stacked.append(H_f)
            r_stacked.append(r)
            del feature_tracks[fid]

        # ---------- PERFORM THE STACKED EKF UPDATE ONCE ----------
        if len(H_stacked) > 0:
            H = np.vstack(H_stacked)
            res = np.hstack(r_stacked)
            R_total = np.eye(len(res)) * (1.0**2) # 1.0 pixel noise
            
            # Standard EKF Math
            S = H @ ekf.P @ H.T + R_total
            K = ekf.P @ H.T @ np.linalg.inv(S)
            last_r = res
            last_S = S
            n_matches = len(H_stacked)
            dx = K @ res
            # Apply capped correction
            dx_limit = 0.5 
            if np.linalg.norm(dx[0:3]) > dx_limit:
                dx *= (dx_limit / np.linalg.norm(dx[0:3]))

            # State Update
            ekf.x.p += dx[0:3]
            ekf.x.v += dx[3:6]
            dR, _ = cv2.Rodrigues(dx[6:9])
            ekf.x.R = ekf.x.R @ dR
            ekf.x.ba += 0.5 * dx[9:12]
            ekf.x.bg += 0.5 * dx[12:15]

            # Joseph Form Covariance Update
            I = np.eye(15)
            IKH = I - K @ H
            ekf.P = IKH @ ekf.P @ IKH.T + K @ R_total @ K.T
            ekf.P = 0.5 * (ekf.P + ekf.P.T) # Ensure symmetry
            ekf.P += np.eye(15) * 1e-9
            vision_success_pts.append(ekf.x.p.copy())
            last_vision_p = ekf.x.p.copy()
            consecutive_vision_skips = 0
            innovation_norm = np.linalg.norm(r)
            print(
                f"[UPDATE] fid={fid} SUCCESS | "
                f"innovation={innovation_norm:.4f}"
            )

            last_vision_p = ekf.x.p.copy()
            consecutive_vision_skips = 0
            # Innovation whitening
            L = np.linalg.cholesky(S + 1e-9 * np.eye(S.shape[0]))
            r_normed = np.linalg.solve(L, r)
            frame_id, z, Rk, pk = track[-1]
            track_age = frame_id - track[0][0]
            reproj_err = np.linalg.norm(r)

            #-----gazebo publishing------------------
            if gazebo_enabled:
                odom = Odometry()
                odom.header.stamp = rospy.Time.now()
                odom.pose.pose.position.x = ekf.x.p[0]
                odom.pose.pose.position.y = ekf.x.p[1]
                odom.pose.pose.position.z = ekf.x.p[2]

                # Convert rotation matrix to quaternion
                R = ekf.x.R
                qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
                qx = (R[2,1] - R[1,2]) / (4*qw + 1e-12)
                qy = (R[0,2] - R[2,0]) / (4*qw + 1e-12)
                qz = (R[1,0] - R[0,1]) / (4*qw + 1e-12)
                odom.pose.pose.orientation.x = qx
                odom.pose.pose.orientation.y = qy
                odom.pose.pose.orientation.z = qz
                odom.pose.pose.orientation.w = qw

                # Optionally, publish velocity
                odom.twist.twist.linear.x = ekf.x.v[0]
                odom.twist.twist.linear.y = ekf.x.v[1]
                odom.twist.twist.linear.z = ekf.x.v[2]

                odom_pub.publish(odom)
            # ---------- LOG for set transformer ----------
            logger.log_measurement(
            frame=i,
            fid=fid,
            innovation=r,
            innovation_norm=np.linalg.norm(r),
            innovation_whitened=r_normed,
            S_trace=np.trace(S),
            S_cond=np.linalg.cond(S),
            baseline=baseline,
            baseline_xy=baseline_xy,
            baseline_ratio=baseline_ratio,
            num_views=len(track),
            track_age=track_age,
            triangulation_status=tri_status,
            H_norm=np.linalg.norm(H_f),
            H_pos=np.linalg.norm(H_f[:, :3]),
            H_rot=np.linalg.norm(H_f[:, 6:9]),
            reproj_err=reproj_err,
            P_trace=np.trace(ekf.P),
            P_pos_trace=np.trace(ekf.P[:3, :3]),
            P_rot_trace=np.trace(ekf.P[3:6, 3:6]),
            delta_t= img_ts[j] - img_ts[i]
        )
        
        traj_world.append(ekf.x.p.copy())
        traj_time.append(img_ts[i])
        P_trace_hist.append(np.trace(ekf.P))
        vel_hist.append(np.linalg.norm(ekf.x.v))
        print(
            f"[DEBUG] skips={consecutive_vision_skips}, "
            f"P_trace={np.trace(ekf.P):.3e}"
        )

        txt_logger.log(img_ts[i], ekf.x, ekf.P, n_matches, last_r, last_S)
        
        # Reset for next frame so we don't log the same vision update twice
        last_r = None
        last_S = None
        n_matches = 0
    
        print(f"Frame {i}: EKF p_prev={p_prev}, EKF p_curr={ekf.x.p}")
    logger.close()
    print("All frames logged successfully!")
    # After the loop finishes
    txt_logger.close()

if __name__ == "__main__":
    main()
