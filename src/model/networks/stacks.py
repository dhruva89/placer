import torch.nn as nn


class PreNorm(nn.Module):
    """
    LayerNorm before the supplied function (attention or feed-forward).
    Supports arbitrary args/kwargs to wrap nn.MultiheadAttention, FeedForward, etc.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x_norm = self.norm(x)
        out = self.fn(x_norm, *args, **kwargs)
        # nn.MultiheadAttention returns (attn_out, attn_weights)
        if isinstance(out, tuple):
            return out[0]
        return out


class FeedForward(nn.Module):
    """
    Simple 2-layer MLP with GELU and dropout.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class RoomTransformerStack(nn.Module):
    """
    Pure self-attention stack for room tokens (Pre-Norm, residual).
    """

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
            ff = FeedForward(dim, mlp_dim, dropout)
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn),  # self-attn
                PreNorm(dim, ff)  # feed-forward
            ]))

    def forward(self, x):
        # x: [B, S_ctx, dim]
        for attn_layer, ff_layer in self.layers:
            # self-attention
            x = x + attn_layer(x, x, x)
            # feed-forward
            x = x + ff_layer(x)
        return x


class FurnitureTransformerStack(nn.Module):
    """
    Cross-attention stack: first self-attend among furniture tokens,
    then cross-attend to room tokens each layer.
    """

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
            cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
            ff = FeedForward(dim, mlp_dim, dropout)
            self.layers.append(nn.ModuleDict({
                'self_attn': PreNorm(dim, self_attn),
                'cross_attn': PreNorm(dim, cross_attn),
                'ff': PreNorm(dim, ff)
            }))

    def forward(self, x, context, key_padding_mask=None):
        # x: [B, num_furn_tokens, dim], context: [B, S_ctx, dim]
        for layer in self.layers:
            # 1) self-attention among furniture tokens
            x = x + layer['self_attn'](x, x, x)
            # 2) cross-attention from furniture (Q) to room (K,V)
            x = x + layer['cross_attn'](x, context, context)
            # 3) feed-forward
            x = x + layer['ff'](x)
        return x
