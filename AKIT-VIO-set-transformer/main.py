#!/usr/bin/env python3
import numpy as np
import torch
import cv2
import rospy
import os
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R_tool
# Preserve your specific imports
from HDF5_Logger.vio_hdf5_logger import VIOHDF5Logger
from model.visual_pipeline import VisualPipeline
from model.keypoint_extractor import ORBKeypointExtractor
from model.ekf.ekf_se3 import EKFSE3, State, NoiseParams
from utils.camera_intrinsics import load_camera_intrinsics
from utils.innovation import reprojection_residual, reprojection_jacobian, linear_multiview_triangulation
from utils.geometry import (match_orb, gravity_alignment_rotation, skew)
from HDF5_Logger.txt_logger import TextLogger

class GazeboVIONode:
    def __init__(self):
        # ---------- 1. SETTINGS & CALIBRATION (PRESERVED) ----------
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Updated R_bc: Maps Camera Frame to Body Frame for a Downward Camera
        self.R_bc = np.array([
    [ 0, -1,  0],  # Camera X is Body -Y (Left)
    [ 0,  0, -1],  # Camera Y is Body -Z (Down)
    [ 1,  0,  0]   # Camera Z is Body X (Forward)
])

        # t_bc usually remains small, but you can update xyz based on your XACRO (1.2 0 0)
        self.t_bc = np.array([1.2, 0.0, 0.0])
        self.VISION_STRIDE = 12
        self.MIN_TRACK_LEN = 2

        # ---------- 2. MODELS & LOGGERS (PRESERVED) ----------
        self.visual_pipe = VisualPipeline(feat_dim=256).to(self.device).eval()
        self.orb_extractor = ORBKeypointExtractor(max_keypoints=2000)
        self.bridge = CvBridge()
        
        if not os.path.exists("logs"): os.makedirs("logs")
        self.logger = VIOHDF5Logger("logs/vio_run_gazebo.h5")
        self.txt_logger = TextLogger("logs/vio_results_gazebo.txt")

        # Camera Intrinsics - Matching Gazebo default 640x480
        # Change from (480, 640) to (492, 768)
        self.K_cam, _, _ = load_camera_intrinsics("camera_intrinsics.yml", (492, 768))
        self.K_cam = self.K_cam[:3, :3].astype(np.float64)

        # ---------- 3. EKF INITIALIZATION (PRESERVED) ----------
    
        x0 = State(R=np.eye(3), p=np.zeros(3), v=np.zeros(3), bg=np.zeros(3), ba=np.zeros(3))
        # Explicitly set gravity to positive because your EKF subtracts it (acc - g)
        self.ekf = EKFSE3(x0, np.eye(15) * 0.1, gravity=np.array([0, 0, 9.80665]))
        self.noise = NoiseParams()
        self.initialized = False

        # MSCKF Feature Tracks state
        self.feature_tracks = {}
        self.feature_id_map = {}
        self.next_feature_id = 0
        self.consecutive_vision_skips = 0
        self.last_vision_p = self.ekf.x.p.copy()
        
        # Buffers for history
        self.kps_buf = []
        self.descs_buf = []
        self.ts_buf = []
        self.imu_history = [] # Stores (t, data)
        self.R_buf = []
        self.p_buf = []

        # ---------- 4. ROS INTERFACE ----------
        self.odom_pub = rospy.Publisher("/vio/odom", Odometry, queue_size=10)
        self.imu_sub = rospy.Subscriber("/lauv/imu", Imu, self.imu_callback)
        self.img_sub = rospy.Subscriber("/lauv/lauv/camerafront/camera_image", Image, self.image_callback)
        self.gt_sub = rospy.Subscriber("/lauv/pose_gt", Odometry, self.gt_callback) 
# Note: Double check if your topic is /lauv/pose_gt or /lauv/ground_truth
# #----------logging txt file-------------------
        # New Relative Start Flags
        self.gt_pose = None
        self.origin_initialized = False
        self.p_offset = np.zeros(3) 

        # Loggers
        self.debug_log = open("logs/vio_debug_comparison.txt", "w")
        self.debug_log.write("timestamp, x_vio, y_vio, z_vio, x_gt, y_gt, z_gt, error_pos, n_matches, innov_norm\n")
        rospy.on_shutdown(self.shutdown_hook)

    def imu_callback(self, msg):
        # Standard IMU Data Array
        data = np.array([
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        ])
        self.imu_history.append((msg.header.stamp.to_sec(), data))

    def image_callback(self, msg):
        current_t = msg.header.stamp.to_sec()
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # A. IMU Prediction
        if len(self.imu_history) >= 2:
            relevant_imu = [x for x in self.imu_history if x[0] <= current_t]
            if len(relevant_imu) > 1:
                for j in range(len(relevant_imu) - 1):
                    t0, d0 = relevant_imu[j]
                    t1, d1 = relevant_imu[j+1]
                    dt = t1 - t0
                    if 0 < dt < 0.1:
                        self.ekf.predict(omega=d0[0:3], acc=d0[3:6], dt=dt, Q=self.noise)
                self.imu_history = [x for x in self.imu_history if x[0] >= relevant_imu[-1][0]]

        # B. Sync Pose (Forward-looking camera)
        R_wc = self.ekf.x.R @ self.R_bc
        p_wc = self.ekf.x.p + self.ekf.x.R @ self.t_bc

        # C. Vision Processing
        img_t = torch.from_numpy(cv_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            enhanced, _ = self.visual_pipe(img_t)
            coords, desc, _ = self.orb_extractor(enhanced)

        self.kps_buf.append(coords[0].cpu().numpy().astype(np.float32))
        self.descs_buf.append(desc[0].cpu().numpy().astype(np.uint8))
        self.ts_buf.append(current_t)
        self.R_buf.append(R_wc.copy())
        self.p_buf.append(p_wc.copy())

        # D. Process Update
        if len(self.ts_buf) > (self.VISION_STRIDE + 1):
            did_reset = self.process_vision_update()
            if did_reset: return 

            self.kps_buf.pop(0)
            self.descs_buf.pop(0)
            self.ts_buf.pop(0)
            self.R_buf.pop(0)
            self.p_buf.pop(0)

        self.publish_odom(msg.header.stamp)

    def process_vision_update(self):
        idx_i, idx_j = 0, self.VISION_STRIDE
        
        # 1. Baseline Safety Check
        baseline = np.linalg.norm(self.p_buf[idx_j] - self.p_buf[idx_i])
        if baseline > 4.0: 
            rospy.logwarn(f"[VIO] Resetting: Baseline {baseline:.2f}m is unrealistic.")
            self.ekf.x.v = np.zeros(3)
            self.clear_buffers()
            return True

        # 2. Matching
        result = match_orb(self.kps_buf[idx_i], self.descs_buf[idx_i], 
                          self.kps_buf[idx_j], self.descs_buf[idx_j])
        if result is None: 
            print("[VIO DEBUG] Matcher returned None - No features found.")
            return False
        
        pts1, pts2, matches = result
        num_matches = len(matches)

        # 3. Triangulation Matrices (Projection)
        P1 = self.K_cam @ np.hstack((self.R_buf[idx_i].T, -self.R_buf[idx_i].T @ self.p_buf[idx_i].reshape(3,1)))
        P2 = self.K_cam @ np.hstack((self.R_buf[idx_j].T, -self.R_buf[idx_j].T @ self.p_buf[idx_j].reshape(3,1)))

        # 4. Perform Triangulation
        p_homog = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        Pw_all = (p_homog[:3] / (p_homog[3] + 1e-12)).T

        H_stacked, r_stacked = [], []
        depth_fail_count = 0

        # 5. Filter Points and Build Jacobians
        for i, Pw in enumerate(Pw_all):
            # Transform World point to Camera j local frame
            Pc_j = self.R_buf[idx_j].T @ (Pw - self.p_buf[idx_j])
            
            # Cheirality Check: Z must be positive (forward in Gazebo camera)
            if Pc_j[2] < 0.5 or Pc_j[2] > 40.0:
                depth_fail_count += 1
                continue

            # Calculate Residual
            r, Pc = reprojection_residual(Pw, self.R_buf[idx_j], self.p_buf[idx_j], pts2[i], self.K_cam)
            
            if r is not None and np.linalg.norm(r) < 20.0:
                J_res = reprojection_jacobian(Pc, self.K_cam)
                if J_res:
                    J_p, J_theta = J_res[0][:, :3], J_res[1][:, :3]
                    
                    # Define H_f (The Measurement Jacobian for this point)
                    H_f = np.zeros((2, 15))
                    # Link to EKF Position (indices 0:3 in your state vector)
                    H_f[:, 0:3] = J_p @ self.ekf.x.R.T
                    # Link to EKF Rotation/Theta (indices 6:9)
                    H_f[:, 6:9] = (J_theta @ self.R_bc.T) - (H_f[:, 0:3] @ skew(self.ekf.x.R @ self.t_bc))
                    
                    H_stacked.append(H_f)
                    r_stacked.append(r)

        # --- DEBUG PRINTS ---
        valid_points = len(H_stacked)
        print(f"[VIO DEBUG] Matches: {num_matches} | Triangulated: {len(Pw_all)} | DepthFail: {depth_fail_count} | Valid for EKF: {valid_points}")

        if valid_points > 12:
            self.apply_ekf_update(np.vstack(H_stacked), np.hstack(r_stacked), valid_points, self.ts_buf[idx_j])
        else:
            print(f"[VIO DEBUG] Update Skipped - Insufficient valid points ({valid_points})")
            
        return False

    def clear_buffers(self):
        self.kps_buf.clear(); self.descs_buf.clear(); self.ts_buf.clear()
        self.R_buf.clear(); self.p_buf.clear()
        
    
    def apply_ekf_update(self, H, res, n_matches, current_t):
        innov_norm = np.linalg.norm(res) / n_matches
        if innov_norm > 20.0:
            print(f"[VIO WARNING] Innovation too high ({innov_norm:.2f}). Skipping update.")
            return
        num_obs = len(res)
        # Use a slightly higher base noise (2.0) for Gazebo underwater stability
        R_val = 2.0 + (num_obs / 1000.0) 
        R_total = np.eye(num_obs) * (R_val**2)
        
        S = H @ self.ekf.P @ H.T + R_total
        try:
            K = self.ekf.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
            
        dx = K @ res
        print(f"[EKF UPDATE] Innovation: {innov_norm:.4f} | Pos Correction: {np.linalg.norm(dx[0:3]):.4f}m | Vel Correction: {np.linalg.norm(dx[3:6]):.4f}m/s")
        # --- REFINED CAPPING LOGIC ---
        # Cap Position Jump (0.15m) and Velocity Jump (0.1m/s)
        max_steps = {"pos": 0.15, "vel": 0.10, "rot": 0.05}
        
        p_norm = np.linalg.norm(dx[0:3])
        if p_norm > max_steps["pos"]:
            dx[0:3] *= (max_steps["pos"] / p_norm)
            rospy.logwarn(f"[VIO] Capping Pos Jump: {p_norm:.2f}m")

        v_norm = np.linalg.norm(dx[3:6])
        if v_norm > max_steps["vel"]:
            dx[3:6] *= (max_steps["vel"] / v_norm)

        # Apply State Update
        self.ekf.x.p += dx[0:3]
        self.ekf.x.v += dx[3:6]
        self.ekf.x.bg += dx[9:12] # Gyro bias
        self.ekf.x.ba += dx[12:15] # Accel bias
        
        dR, _ = cv2.Rodrigues(dx[6:9])
        self.ekf.x.R = self.ekf.x.R @ dR
        
        # --- LOGGING ---
        if self.gt_pose is not None:
            vio_p = self.ekf.x.p
            gt_p = self.gt_pose
            pos_error = np.linalg.norm(vio_p - gt_p)
            innov_norm = np.linalg.norm(res)

            log_line = (f"{current_t:.4f}, {vio_p[0]:.4f}, {vio_p[1]:.4f}, {vio_p[2]:.4f}, "
                        f"{gt_p[0]:.4f}, {gt_p[1]:.4f}, {gt_p[2]:.4f}, {pos_error:.4f}, "
                        f"{n_matches}, {innov_norm:.4f}\n")
            self.debug_log.write(log_line)
            self.debug_log.flush()

        # Joseph Form Covariance Update
        I = np.eye(15)
        IKH = I - K @ H
        self.ekf.P = IKH @ self.ekf.P @ IKH.T + K @ R_total @ K.T
        self.ekf.P = 0.5 * (self.ekf.P + self.ekf.P.T)
        self.consecutive_vision_skips = 0
    
    def publish_odom(self, stamp):
        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = "world"
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = self.ekf.x.p
        R = self.ekf.x.R
        qw = np.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2
        o.pose.pose.orientation.w = qw
        o.pose.pose.orientation.x = (R[2,1] - R[1,2]) / (4*qw + 1e-12)
        o.pose.pose.orientation.y = (R[0,2] - R[2,0]) / (4*qw + 1e-12)
        o.pose.pose.orientation.z = (R[1,0] - R[0,1]) / (4*qw + 1e-12)
        self.odom_pub.publish(o)

    from scipy.spatial.transform import Rotation as R_tool

    def gt_callback(self, msg):
        # 1. Extract Position
        curr_gt_p = np.array([msg.pose.pose.position.x, 
                            msg.pose.pose.position.y, 
                            msg.pose.pose.position.z])
        
        # 2. Extract Orientation (Crucial for Gravity Cancellation)
        q = msg.pose.pose.orientation
        # scipy expects [x, y, z, w]
        actual_R = R_tool.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        if not self.origin_initialized:
            # SYNC THE EKF TO REALITY
            self.ekf.x.p = curr_gt_p.copy()
            self.ekf.x.R = actual_R.copy() # No more R=Identity!
            
            # Initialize velocity to 0 or GT velocity if available
            self.ekf.x.v = np.zeros(3) 
            
            self.origin_initialized = True
            rospy.loginfo(f"VIO Initialized! Orientation aligned with Gravity.")

        self.gt_pose = curr_gt_p

    def shutdown_hook(self):
        rospy.loginfo("Shutting down VIO Node... Saving logs.")
        self.debug_log.close()
        self.txt_logger.close()
        self.logger.close()

if __name__ == '__main__':
    rospy.init_node('vio_gazebo_live')
    node = GazeboVIONode()
    rospy.spin()