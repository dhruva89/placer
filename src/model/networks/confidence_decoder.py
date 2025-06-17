import torch
import torch.nn as nn


class ConfidenceDecoder(nn.Module):
    def __init__(self, hidden_dim=512):
        super(ConfidenceDecoder, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # x: [B, T, hidden_dim]
        x = self.norm(x)
        conf = torch.sigmoid(self.fc(x))  # [B, T, 1]
        return conf.squeeze(-1)  # [B, T]
