import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as Rot
from model.ekf.noise import NoiseParams
from utils.geometry import project_to_so3
def so3_exp(phi):
    return Rot.from_rotvec(phi).as_matrix()

@dataclass
class State:
    R: np.ndarray
    p: np.ndarray
    v: np.ndarray
    bg: np.ndarray
    ba: np.ndarray

def skew(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

class StateSE3:
    def __init__(self):
        self.R = np.eye(3)
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)

class EKFSE3:
    def __init__(self, x0: State, P0: np.ndarray, gravity: np.ndarray = np.array([0,0,9.80665])):
        """
        x0  : initial State
        P0  : initial covariance (15x15)
        gravity : np.array([0,0,-9.81])
        """
        self.x = x0
        self.P = P0.copy()
        self.g = gravity
        self.predict_count = 0  # Counter for debug
    
    def _normalize_rotation(self):
        U, _, Vt = np.linalg.svd(self.x.R)
        self.x.R = U @ Vt


    def predict(self, omega, acc, dt, Q: NoiseParams):
        self.predict_count +=1 # Counter for debug
        x = self.x

        # Subtract biases
        omega_unbiased = omega - x.bg
        acc_unbiased   = acc - x.ba

        # ---- DEBUG: Check for zero dt ----
        if dt <= 0:
            print(f"[WARN] Zero or negative dt: {dt}. Skipping prediction.")
            return

        # ---- Rotation update ----
        dR = so3_exp(omega_unbiased * dt)
        x.R = x.R @ dR

        # ---- Gravity-aligned world acceleration ----
        a_world = x.R @ acc_unbiased - self.g
        # ---- DEBUG: Detect frozen physics ----
        if np.linalg.norm(a_world) < 1e-9 and np.linalg.norm(x.v) < 1e-9:
            print(f"[DEBUG] Physics frozen. acc={acc_unbiased}, g={self.g}, dt={dt}")
        else:
            if self.predict_count % 100 == 0:
                print(f"[EKF Predict] Net Accel: {np.linalg.norm(a_world):.4f} m/s^2 | Vel: {np.linalg.norm(x.v):.2f} m/s")
            # Normal integration
            x.p = x.p + x.v * dt + 0.5 * a_world * dt**2
            x.v = x.v + a_world * dt

        # ---- Rotation normalization ----
        self._normalize_rotation()

        # ---- Process noise matrix ----
        Qk = np.zeros((15, 15))
        Qk[0:3, 0:3]   = np.eye(3) * Q.gyro_noise**2 * dt
        Qk[3:6, 3:6]   = np.eye(3) * Q.accel_noise**2 * dt
        Qk[9:12, 9:12] = np.eye(3) * Q.gyro_bias_rw**2 * dt
        Qk[12:15,12:15]= np.eye(3) * Q.accel_bias_rw**2 * dt

        # ---- State transition matrix ----
        F = np.eye(15)
        F[0:3, 0:3] -= skew(omega_unbiased) * dt
        F[0:3, 9:12] = -np.eye(3) * dt

        F[3:6, 0:3] = -x.R @ skew(acc_unbiased) * dt
        F[3:6, 12:15] = -x.R * dt

        F[6:9, 3:6] = np.eye(3) * dt  # Position integrates velocity

        # ---- Covariance update ----
        self.P = F @ self.P @ F.T + Qk

        # ---- COVARIANCE SANITY BOUND (NOT CLIP) ----
        trace_limit = 1e5
        P_trace = np.trace(self.P)
        if P_trace > trace_limit:
            self.P *= (trace_limit / P_trace)

        # ---- Positive definiteness check (no clipping) ----
        if not np.all(np.isfinite(self.P)):
            print("🚨 EKF covariance NaN detected → HARD RESET")
            self.reset()
            return

        if np.trace(self.P) > 1e12:
            print(f"[WARN] EKF P trace exploding: {np.trace(self.P):.2e}")

        # ---- Ensure R stays in SO(3) ----
        self.x.R = project_to_so3(self.x.R)
        
    

    
    # ---------------- GENERIC EKF UPDATE ----------------
    def update_generic(self, y, H, R):
    # Innovation covariance
        S = H @ self.P @ H.T + R

    # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

    # State increment
        delta = (K @ y).reshape(-1)

    # Covariance update
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P

    # Apply state increment ONCE
        self._apply_delta(delta)

    # Normalize rotation after update
        self._normalize_rotation()

        return K, delta


    def _apply_delta(self, dx):
        dtheta = dx[0:3]
        dv     = dx[3:6]
        dp     = dx[6:9]
        dbg    = dx[9:12]
        dba    = dx[12:15]

        self.x.R = self.x.R @ so3_exp(dtheta)
        self.x.v += dv
        self.x.p += dp
        self.x.bg += dbg
        self.x.ba += dba


    # ---------------- VISION UPDATE ----------------
    def update_from_reprojection(self, Pw, uv, K4, pix_sigma=1.0):

        fx, fy, cx, cy = K4
        Rwb, pw = self.x.R, self.x.p

        rows, H_rows = [], []

        for i in range(Pw.shape[0]):

            Pc = Rwb.T @ (Pw[i] - pw)
            X, Y, Z = Pc

            if Z < 0.5:
                continue

            # predicted pixel
            u_hat = fx * X / Z + cx
            v_hat = fy * Y / Z + cy

            # ---- NORMALIZED RESIDUAL ----
            r = np.array([
                uv[i,0] - u_hat,
                uv[i,1] - v_hat
            ])

            rows.append(r)

            # projection Jacobian
            Jp = np.array([
                [fx / Z, 0, -fx * X / (Z * Z)],
                [0, fy / Z, -fy * Y / (Z * Z)]
            ])


            H = np.zeros((2, 15))
            H[:, 0:3] = Jp @ (-skew(Pc))   # rotation
            H[:, 6:9] = -Jp                # ✅ POSITION UPDATE ENABLED

            H_rows.append(H)

        if len(rows) < 5:
            return False, None, None

        y = np.concatenate(rows)
        H = np.vstack(H_rows)

        # measurement noise (normalized)
        R = (pix_sigma) ** 2 * np.eye(len(y))

        # ---- MAHALANOBIS GATING ----
        S = H @ self.P @ H.T + R
        d2 = y.T @ np.linalg.inv(S) @ y
        dof = len(y)
        print("mean |r| px:", np.mean(np.linalg.norm(np.array(rows).reshape(-1,2), axis=1)))

        if d2 > 20.0 * (dof // 2):
            print("Vision update rejected (Mahalanobis gate)")
            return False, None, None

        K_gain, delta = self.update_generic(y, H, R)

        # ---- SAFETY CHECK ----
        if np.linalg.norm(delta[0:3]) > 0.2:
            print("Large rotation correction rejected")
            return False, None, None

        if np.linalg.norm(delta[6:9]) > 2.0:
            print("Large position correction rejected")
            return False, None, None


        return True, K_gain, delta
