import torch.nn as nn


class InputToOutputMapper(nn.Module):
    def __init__(self, hidden_dim=512, num_classes=30):
        super(InputToOutputMapper, self).__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes + 1)

    def forward(self, x):
        # x: [B, T, hidden_dim]
        return self.fc(self.ln(x))  # [B, T, max_furniture_pieces + 1]
