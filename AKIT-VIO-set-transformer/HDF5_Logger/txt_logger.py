import numpy as np
class TextLogger:
        def __init__(self, filename="logs/vio_results.txt"):
            self.file = open(filename, "w")
            header = (
                "timestamp,pos_x,pos_y,pos_z,"
                "roll,pitch,yaw,vel_x,vel_y,vel_z"
                ",P_trace\n"
            )
            self.file.write(header)

        def log(self, timestamp, state, P, num_matches=0, r_vec=None, S_mat=None):
            R = state.R
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            if not singular:
                roll = np.arctan2(R[2, 1], R[2, 2])
                pitch = np.arctan2(-R[2, 0], sy)
                yaw = np.arctan2(R[1, 0], R[0, 0])
            else:
                roll = np.arctan2(-R[1, 2], R[1, 1])
                pitch = np.arctan2(-R[2, 0], sy)
                yaw = 0

            # --- 2. Calculate Diagnostics ---
            inno_norm = np.linalg.norm(r_vec) if r_vec is not None else 0.0
            s_trace = np.trace(S_mat) if S_mat is not None else 0.0
            p_trace = np.trace(P)

            # --- 3. Format CSV Row ---
            log_str = (
                f"{timestamp:.6f},"
                f"{state.p[0]:.4f},{state.p[1]:.4f},{state.p[2]:.4f},"  # Position
                f"{np.degrees(roll):.2f},{np.degrees(pitch):.2f},{np.degrees(yaw):.2f}," # Orientation
                f"{state.v[0]:.4f},{state.v[1]:.4f},{state.v[2]:.4f},"  # Velocity
                #f"{state.bg[0]:.6f},{state.bg[1]:.6f},{state.bg[2]:.6f}," # Gyro Bias
                f"{p_trace:.6f}\n"    # Filter total uncertainty
            )
            
            self.file.write(log_str)
            self.file.flush() 

        def close(self):
            self.file.close()