# models/imu_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class IMUFeatureExtractor(nn.Module):
    """
    Extract features from a window of IMU samples.
    Input: imu_seq -> (B, T, 6)  (accel x3, gyro x3)
    Output: feature vector (B, imu_feat_dim)
    """
    def __init__(self, imu_feat_dim=128, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=6,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.projector = nn.Sequential(
            nn.Linear(2 * hidden_size, imu_feat_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(imu_feat_dim)
        )

    def forward(self, imu_seq, seq_lens=None):
        """
        imu_seq: (B, T, 6), torch.float32
        seq_lens: optional (B,) lengths if variable length windows are used
        """
        # If variable lengths are supplied, pack for efficiency
        if seq_lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(imu_seq, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.rnn(packed)
        else:
            out, (h_n, c_n) = self.rnn(imu_seq)
        # h_n: (num_layers * num_directions, B, hidden_size)
        # take last layer's forward and backward hidden states and concat
        # indexing: last forward = h_n[-2], last backward = h_n[-1]
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=-1)  # (B, 2*hidden_size)
        feat = self.projector(h_cat)  # (B, imu_feat_dim)
        return feat


# Helper: simple normalization (zero-mean, per-sequence)
def normalize_imu(imu_seq):
    """
    imu_seq: numpy or torch (B, T, 6) or (T,6)
    Returns same shape torch tensor, normalized per-sample (zero mean, std)
    """
    if not isinstance(imu_seq, torch.Tensor):
        imu_seq = torch.from_numpy(imu_seq).float()
    mean = imu_seq.mean(dim=1, keepdim=True)
    std = imu_seq.std(dim=1, keepdim=True) + 1e-6
    return (imu_seq - mean) / std
