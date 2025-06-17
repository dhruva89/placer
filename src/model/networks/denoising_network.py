import torch
import torch.nn as nn
from diffusers.models.embeddings import SinusoidalPositionalEmbedding

from src.model.networks.adaptive_layer_norm import AdaptiveLayerNorm
from src.model.networks.geometric_decoder import GeometricDecoder
from src.model.networks.mlp import MLP
from src.model.networks.semantic_decoder import InputToOutputMapper


class SelfAttention(nn.Module):
    """Multi‐head self‐attention (Transformer encoder style)."""

    def __init__(self, n_embd, n_head, dropout=0.1, batch_first=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=batch_first
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Returns only the attended output, not the attention weights.
        return self.mha(
            x, x, x,
            need_weights=False,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )[0]


class CrossAttention(nn.Module):
    """Multi‐head cross‐attention (Transformer decoder style)."""

    def __init__(self, n_embd, n_head, dropout=0.1, batch_first=True, kv_embd=None):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=batch_first,
            kdim=kv_embd,
            vdim=kv_embd
        )

    def forward(self, q, kv, attn_mask=None, key_padding_mask=None):
        # q:   [B, n, n_embd]      (queries from the current token set)
        # kv:  [B, S_ctx, kv_embd] (keys/values from the context)
        return self.mha(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )[0]


class Block(nn.Module):
    """
    One transformer block that:
     1) Self‐attends to the n “output slots” (after time‐conditioned LN).
     2) Cross‐attends to the room context.
     3) Runs a feed‐forward MLP.
    """

    def __init__(
            self,
            n_embd: int,
            n_head: int,
            conditioning_embedding_dim: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            num_timesteps: int = 1000
    ):
        super().__init__()
        self.attn1 = SelfAttention(n_embd, n_head, dropout=dropout)
        self.attn2 = CrossAttention(
            n_embd,
            n_head,
            dropout,
            True,
            conditioning_embedding_dim
        )
        # AdaptiveLayerNorm will house a time‐embedding (sinusoidal, learned, etc.)
        # so that each timestep t uses a different LN scale+shift.
        self.layer_norm_1 = AdaptiveLayerNorm(n_embd, num_timesteps)
        self.layer_norm_2 = AdaptiveLayerNorm(n_embd, num_timesteps)
        self.layer_norm_3 = nn.LayerNorm(n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, timestep,
                conditioning,  # [B,S_ctx,d]
                mask1=None, mask2=None):
        # 1) Time‐conditioned layernorm, then self‐attend
        x = x + self.attn1(self.layer_norm_1(x, timestep), key_padding_mask=mask1)
        # 2) Time‐conditioned layernorm, then cross‐attend to “conditioning” (the input context)
        x = x + self.attn2(
            self.layer_norm_2(x, timestep),
            conditioning,
            key_padding_mask=mask2
        )
        # 3) Feed‐forward block
        x = x + self.mlp(self.layer_norm_3(x))
        return x


class DenoisingNetwork(nn.Module):
    """
    Transformer decoder for the MixedDiffusion model.
    - It decodes **n** output slots (each with a discrete‐semantic token and continuous geometry).
    - It attends to a context built from **m** input slots.
    - Output heads:
        • sem_logits → [B, n, num_furniture_classes+1]
        • geo_pred   → [B, n, 6]
        • ptr_logits → [B, n, m]     (pointer into input slots; no “NEW” category here)
    """

    def __init__(
            self,
            hidden_dim: int = 512,
            num_layers: int = 8,
            num_heads: int = 4,
            num_furniture_classes: int = 33,
            max_input_slots: int = 20,  # m_max
            max_output_slots: int = 20,  # n_max
            num_timesteps: int = 1000
    ):
        super().__init__()
        self.max_input_slots = max_input_slots
        self.max_output_slots = max_output_slots

        # === 1) Embedding tables ===
        # 1a) Embed the **noisy** discrete-semantic token per output slot ∈ [0..C-1 or PAD=C].
        self.semantic_embedding = nn.Embedding(
            num_embeddings=num_furniture_classes + 2,  # +1 for PAD +1 for Mask
            embedding_dim=hidden_dim // 2
        )

        # 1b) Embed the 6‐D geometry via an MLP → hidden_dim//2
        self.geometric_mlp = MLP(
            6,
            hidden_dim // 2
        )

        # === 2) Transformer body ===
        # We will form a [sem_emb || geo_emb] of size hidden_dim, add slot_id_emb → hidden_dim,
        # then pass through a stack of Blocks that also cross‐attend to “context” (m input slots).
        self.tf_blocks = nn.Sequential(
            *[
                Block(
                    n_embd=hidden_dim,
                    n_head=num_heads,
                    conditioning_embedding_dim=hidden_dim,
                    dim_feedforward=2048,
                    dropout=0.1,
                    num_timesteps=num_timesteps
                )
                for _ in range(num_layers)
            ]
        )

        # === 3) Output heads ===
        # 3a) Semantic head: maps hidden_dim → (num_furniture_classes + 1)
        self.input_to_output_mapper = InputToOutputMapper(
            hidden_dim=hidden_dim,
            num_classes=num_furniture_classes
        )
        # Produces [B, n, num_furniture_classes+1]

        # 3b) Geometry head: maps hidden_dim → 6 (ε̂ for geometry)
        self.geometric_decoder = GeometricDecoder(
            hidden_dim=hidden_dim,
            output_dim=6
        )
        # Produces [B, n, 6]

        # 3c) **Pointer head**: maps hidden_dim → m (pointer into input slots).
        #     No “NEW” class; each output must point to one of the m inputs.
        self.ptr_decoder = nn.Linear(
            in_features=hidden_dim,
            out_features=max_input_slots + 1  # one logit per input slot
        )

        self.slot_pos_emb = SinusoidalPositionalEmbedding(
            embed_dim=hidden_dim,
            max_seq_length=self.max_output_slots
        )

    def forward(
            self,
            x_sem: torch.Tensor,
            x_geo: torch.Tensor,
            time: torch.Tensor,
            context: torch.Tensor,
            input_mask: torch.Tensor,
            output_mask: torch.Tensor,
    ):
        """
        Args:
          x_sem:    [B, n]     int64, each in {0..C-1, PAD=C}, the *noisy* semantic tokens at time t.
          x_geo:    [B, n, 6]  float32, the *noisy* geometry at time t.
          time:     [B]        int64, the shared diffusion timestep.
          context:  [B, S_ctx, hidden_dim]  (computed by MixedDiffusion.build_context from m input slots)

        Returns:
          sem_logits:  [B, n, C+1]  raw logits for discrete diffusion on semantics.
          geo_pred:    [B, n, 6]    predicted ε̂ for geometry.
          ptr_logits:  [B, n, m]    raw logits pointing each output slot to one of the m input slots.
        """

        # 1) Embed semantics + geometry
        sem_emb = self.semantic_embedding(x_sem)  # → [B, n, hidden_dim * 3//2]
        geo_emb = self.geometric_mlp(x_geo)  # → [B, n, hidden_dim * 3//2]

        # 3) Concatenate [sem_emb ‖ geo_emb] → hidden_dim, then add slot_emb
        x = torch.cat([sem_emb, geo_emb], dim=-1)

        # add sinusoidal slot‐index encoding
        x = self.slot_pos_emb(x)

        slot_xyz = (x_geo[..., :3] + x_geo[..., 3:]) * 0.5

        # 4) Run through Transformer blocks (self‐attend + cross‐attend to context)
        for block in self.tf_blocks:
            x = block(x, time, context, output_mask, input_mask)

        # 5a) Semantic logits
        sem_logits = self.input_to_output_mapper(x)
        #    shape = [B, n, num_furniture_classes+1]

        # 5b) Geometry prediction
        geo_pred = self.geometric_decoder(x)  # [B, n, 6]

        # 5c) Pointer logits (no diffusion; trained by CE)
        ptr_logits = self.ptr_decoder(x)  # [B, n, m + 1]

        return sem_logits, geo_pred, ptr_logits
