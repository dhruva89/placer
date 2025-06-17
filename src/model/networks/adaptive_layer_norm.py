import torch
from torch import Tensor, nn

LAYER_NORM_EPS = 1e-5


class AdaptiveLayerNorm(nn.Module):

    def __init__(
            self, n_embd: int, max_timestep: int
    ):
        super().__init__()
        self.emb = nn.Embedding(max_timestep, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layer_norm = nn.LayerNorm(n_embd, eps=LAYER_NORM_EPS, elementwise_affine=False)

    def forward(self, x: Tensor, timestep: Tensor):

        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)  # B, 1, 2*n_embd
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layer_norm(x) * (1 + scale) + shift
        return x
