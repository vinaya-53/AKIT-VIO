import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
import numpy as np

from set_transformer import SetTransformer


# ================= PATH =================
H5_PATH = Path(__file__).resolve().parents[2] / "logs" / "vio_run_001.h5"
assert H5_PATH.exists()

with h5py.File(H5_PATH, "r") as f:
    print("Total frames:", f["vio"]["frame_id"].shape[0])
with h5py.File(H5_PATH, "r") as f:
    for k in f["vio"].keys():
        print(k, f["vio"][k].shape)

# ================= DATASET =================
class VIOSetDataset(Dataset):
    def __init__(self, h5_path, max_set_size=32):
        self.h5 = h5py.File(str(h5_path), "r")
        self.meas = self.h5["vio"]["measurement"]
        self.length = self.meas["frame"].shape[0]
        self.max_set_size = max_set_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Build one set (pad / truncate)
        X = []

        for k in range(idx):
            xk = [
                self.meas["innovation_norm"][k],
                np.linalg.norm(self.meas["innovation_whitened"][k]),
                self.meas["S_trace"][k],
                np.log(self.meas["S_cond"][k] + 1e-6),

                self.meas["baseline"][k],
                self.meas["baseline_ratio"][k],
                self.meas["num_views"][k],
                self.meas["track_age"][k],

                self.meas["H_norm"][k],
                self.meas["H_pos"][k],
                self.meas["H_rot"][k],

                self.meas["P_trace"][k],
                self.meas["P_pos_trace"][k],
                self.meas["P_rot_trace"][k],
            ]
            X.append(xk)
            if len(X) >= self.max_set_size:
                break

        X = torch.tensor(X, dtype=torch.float32)

        # Target noise scales (self-supervised)
        innov = self.meas["innovation_norm"][idx]
        cond = self.meas["S_cond"][idx]

        y = torch.tensor([
            1.0 + 0.5 * innov,
            1.0 + 0.2 * innov,
            1.0 + 0.3 * innov,
            1.0 + 0.1 * cond
        ], dtype=torch.float32)

        return X, y


# ================= SETUP =================
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = VIOSetDataset(H5_PATH)
loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

model = SetTransformer(
    input_dim=14,
    hidden_dim=128,
    output_dim=4
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(40):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # set-wise normalization
        X = (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True) + 1e-6)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch:03d} | Loss {total_loss / len(loader):.6f}")

torch.save(model.state_dict(), "set_transformer_noise.pth")
