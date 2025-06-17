import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        mlp_layers = [
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        ]
        self.sequential = nn.Sequential(*mlp_layers)

    def forward(self, geo):
        return self.sequential(geo)
