# set_transformer.py
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dk = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, Q, K, V):
        B, N, D = Q.shape
        q = self.q(Q).view(B, N, self.num_heads, self.dk).transpose(1, 2)
        k = self.k(K).view(B, -1, self.num_heads, self.dk).transpose(1, 2)
        v = self.v(V).view(B, -1, self.num_heads, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        attn = torch.softmax(scores, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous()
        out = out.view(B, N, D)
        return self.o(out)

class ISAB(nn.Module):
    def __init__(self, dim, num_heads, num_inds):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inds, dim))
        self.mha1 = MultiHeadAttention(dim, num_heads)
        self.mha2 = MultiHeadAttention(dim, num_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, X):
        B = X.size(0)
        I = self.I.repeat(B, 1, 1)
        H = self.ln1(self.mha1(I, X, X))
        return self.ln2(self.mha2(X, H, H))

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mha = MultiHeadAttention(dim, num_heads)
        self.ln = nn.LayerNorm(dim)

    def forward(self, X):
        B = X.size(0)
        S = self.S.repeat(B, 1, 1)
        return self.ln(self.mha(S, X, X))

class SetTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.embed = nn.Linear(input_dim, hidden_dim)

        self.enc = nn.Sequential(
            ISAB(hidden_dim, num_heads=4, num_inds=16),
            ISAB(hidden_dim, num_heads=4, num_inds=16),
        )

        self.pma = PMA(hidden_dim, num_heads=4, num_seeds=1)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()   # 🔑 positivity for noise scales
        )

    def forward(self, X):
        X = self.embed(X)
        X = self.enc(X)
        X = self.pma(X).squeeze(1)
        return self.head(X)
