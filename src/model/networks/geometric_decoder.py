import torch.nn as nn


class GeometricDecoder(nn.Module):
    def __init__(self, hidden_dim=512, output_dim=16):
        """
        A 3-layer MLP with hidden dimensions [512, 1024] producing an 8-dim Gaussian mean.
        """
        super(GeometricDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        # x: [B, T, hidden_dim]
        return self.mlp(x)  # [B, T, 8]
