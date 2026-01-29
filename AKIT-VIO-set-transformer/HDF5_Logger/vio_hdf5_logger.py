import h5py
import numpy as np
import os


class VIOHDF5Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = h5py.File(path, "w")

        # ---------- ROOT GROUP ----------
        self.grp = self.f.create_group("vio")
        print("[DEBUG] Created root group '/vio'")

        # ---------- FRAME-LEVEL DATASETS ----------
        self._create_ds("frame_id", (0,), np.int32)

        self._create_ds("ekf_p", (0, 3), np.float32)
        self._create_ds("ekf_v", (0, 3), np.float32)
        self._create_ds("ekf_R", (0, 3, 3), np.float32)

        self._create_ds("P_diag", (0, 15), np.float32)
        self._create_ds("P_trace", (0,), np.float32)
        self._create_ds("P_pos_trace", (0,), np.float32)
        self._create_ds("P_rot_trace", (0,), np.float32)

        self._create_ds("num_matches", (0,), np.int32)
        self._create_ds("num_triangulated", (0,), np.int32)

        self._create_ds("baseline", (0,), np.float32)
        self._create_ds("baseline_xy", (0,), np.float32)
        self._create_ds("baseline_ratio", (0,), np.float32)

        self._create_ds("forward_deg", (0,), np.uint8)
        self._create_ds("vision_success", (0,), np.uint8)
        self._create_ds("innovation_norm", (0,), np.float32)

        self.N = 0
        print("[DEBUG] Frame-level datasets created")

        # ---------- MEASUREMENT-LEVEL GROUP ----------
        self.meas = self.grp.create_group("measurement")
        print("[DEBUG] Created measurement group '/vio/measurement'")

        self._create_meas_ds("frame", (), np.int32)
        self._create_meas_ds("fid", (), np.int32)
        self._create_meas_ds("vision_success", (), np.uint8)

        self._create_meas_ds("innovation", (3,), np.float32)
        self._create_meas_ds("innovation_norm", (), np.float32)
        self._create_meas_ds("innovation_whitened", (3,), np.float32)

        self._create_meas_ds("reproj_err", (), np.float32)

        self._create_meas_ds("S_trace", (), np.float32)
        self._create_meas_ds("S_cond", (), np.float32)

        self._create_meas_ds("H_norm", (), np.float32)
        self._create_meas_ds("H_pos", (), np.float32)
        self._create_meas_ds("H_rot", (), np.float32)

        self._create_meas_ds("num_views", (), np.int32)
        self._create_meas_ds("track_age", (), np.int32)
        self._create_meas_ds("triangulation_status", (), np.uint8)

        self._create_meas_ds("baseline", (), np.float32)
        self._create_meas_ds("baseline_ratio", (), np.float32)

        self._create_meas_ds("delta_t", (), np.float32)

        self.M = 0
        print("[DEBUG] Measurement-level datasets created")

    # ----------------- FRAME-LEVEL -----------------
    def _create_ds(self, name, shape, dtype):
        self.grp.create_dataset(
            name,
            shape=shape,
            maxshape=(None,) + shape[1:],
            dtype=dtype,
            chunks=True
        )

    def _append_ds(self, name, value):
        ds = self.grp[name]
        ds.resize((self.N + 1,) + ds.shape[1:])
        ds[self.N] = value

    def log(
        self,
        frame_id,
        ekf,
        num_matches,
        num_triangulated,
        baseline,
        baseline_xy,
        forward_deg,
        vision_success,
        innovation_norm
    ):
        baseline_ratio = baseline_xy / baseline if baseline > 1e-6 else 0.0

        self._append_ds("frame_id", frame_id)

        self._append_ds("ekf_p", ekf.x.p)
        self._append_ds("ekf_v", ekf.x.v)
        self._append_ds("ekf_R", ekf.x.R)

        self._append_ds("P_diag", np.diag(ekf.P))
        self._append_ds("P_trace", np.trace(ekf.P))
        self._append_ds("P_pos_trace", np.trace(ekf.P[:3, :3]))
        self._append_ds("P_rot_trace", np.trace(ekf.P[3:6, 3:6]))

        self._append_ds("num_matches", num_matches)
        self._append_ds("num_triangulated", num_triangulated)

        self._append_ds("baseline", baseline)
        self._append_ds("baseline_xy", baseline_xy)
        self._append_ds("baseline_ratio", baseline_ratio)

        self._append_ds("forward_deg", forward_deg)
        self._append_ds("vision_success", vision_success)
        self._append_ds("innovation_norm", innovation_norm)

        self.N += 1

    # ----------------- MEASUREMENT-LEVEL -----------------
    def _create_meas_ds(self, name, shape, dtype):
        self.meas.create_dataset(
            name,
            shape=(0,) + shape,
            maxshape=(None,) + shape,
            dtype=dtype,
            chunks=True
        )

    def _append_meas(self, name, value):
        ds = self.meas[name]
        ds.resize((self.M + 1,) + ds.shape[1:])
        ds[self.M] = value

    def log_measurement(
        self,
        frame,
        vision_success,
        fid,
        innovation,
        innovation_norm,
        innovation_whitened,
        reproj_err,
        S_trace,
        S_cond,
        H_norm,
        H_pos,
        H_rot,
        num_views,
        track_age,
        triangulation_status,
        baseline,
        baseline_ratio,
        delta_t
    ):
        self._append_meas("frame", frame)
        self._append_meas("fid", fid)
        self._append_meas("vision_success",vision_success)
        self._append_meas("innovation", innovation)
        self._append_meas("innovation_norm", innovation_norm)
        self._append_meas("innovation_whitened", innovation_whitened)

        self._append_meas("reproj_err", reproj_err)

        self._append_meas("S_trace", S_trace)
        self._append_meas("S_cond", S_cond)

        self._append_meas("H_norm", H_norm)
        self._append_meas("H_pos", H_pos)
        self._append_meas("H_rot", H_rot)

        self._append_meas("num_views", num_views)
        self._append_meas("track_age", track_age)
        self._append_meas("triangulation_status", triangulation_status)

        self._append_meas("baseline", baseline)
        self._append_meas("baseline_ratio", baseline_ratio)

        self._append_meas("delta_t", delta_t)

        self.M += 1
    def get_frame_measurements(self, frame_id):
        """
        Returns all measurement-level entries for a given frame_id
        as a list of dictionaries.
        """
        meas = self.meas

        frame_ids = meas["frame"][:]   # shape (M,)
        idxs = np.where(frame_ids == frame_id)[0]

        frame_measurements = []

        for i in idxs:
            m = {
                "innovation_norm": meas["innovation_norm"][i],
                "innovation_whitened": meas["innovation_whitened"][i],
                "S_trace": meas["S_trace"][i],
                "S_cond": meas["S_cond"][i],
                "baseline_ratio": meas["baseline_ratio"][i],
                "reproj_err": meas["reproj_err"][i],
            }
            frame_measurements.append(m)

        return frame_measurements


    # ----------------- CLOSE -----------------
    def close(self):
        self.f.close()
        print("[DEBUG] HDF5 file closed")
        
