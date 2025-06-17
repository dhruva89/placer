import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.schedulers.scheduling_vq_diffusion import index_to_log_onehot

from src.model.d3pm_scheduler import CustomVQDiffusionScheduler
from src.model.mixed_loss import MixedLoss
from src.model.networks.denoising_network import DenoisingNetwork
from src.model.networks.pointmlp import PointMLPEncoder
from src.model.networks.pointnet_encoder import PointNetPlusPlusEncoder
from src.model.networks.sem_embedding import InputSemanticClassEmbedding
from src.model.networks.stacks import FurnitureTransformerStack
from src.utils import split_integer


class MixedDiffusion(nn.Module):
    def __init__(
            self,
            device: str,
            batch_size: int,
            diffusion_semantic_kwargs: dict,
            diffusion_geometric_kwargs: dict,
            diffusion_steps: int = 1000,
            num_furniture_classes: int = 33,
            max_input_slots: int = 20,
            max_output_slots: int = 20
    ):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.diffusion_steps = diffusion_steps

        self.num_furniture_classes = num_furniture_classes
        self.max_input_slots = max_input_slots
        self.max_output_slots = max_output_slots

        self.hidden_dim = 512

        # === 1) Context encoder (unchanged) ===
        self.encoder = PointNetPlusPlusEncoder()  # → [B, hidden_dim, S_ctx]
        self.furniture_encoder = PointMLPEncoder(embed_dim=16)
        self.emb_dims_sizes = split_integer(self.hidden_dim, 2)
        self.furniture_proj = nn.Linear(16 * 2 * 2 * 2 * 2, self.emb_dims_sizes[0])
        self.semantic_embedding = InputSemanticClassEmbedding(
            num_furniture_classes,
            self.emb_dims_sizes[1]
        )

        self.furn_stack = FurnitureTransformerStack(
            dim=self.hidden_dim,
            depth=4,
            heads=8,
            mlp_dim=self.hidden_dim * 4,
            dropout=0.1
        )

        self.context_pos_emb = SinusoidalPositionalEmbedding(
            embed_dim=self.hidden_dim,
            max_seq_length=self.max_input_slots + (self.hidden_dim // 8)
        )

        # === 2) Denoiser core (unchanged) ===
        self.denoiser = DenoisingNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=8,
            num_heads=4,
            num_furniture_classes=num_furniture_classes,
            max_input_slots=max_input_slots,
            max_output_slots=max_output_slots,
            num_timesteps=diffusion_steps
        )

        # === 3) Schedulers & losses (unchanged) ===
        vocab_size = num_furniture_classes + 2
        self.d3pm = CustomVQDiffusionScheduler(
            num_vec_classes=vocab_size,
            num_train_timesteps=diffusion_steps,
            alpha_cum_start=diffusion_semantic_kwargs['att_1'],
            alpha_cum_end=diffusion_semantic_kwargs['att_T'],
            gamma_cum_start=diffusion_semantic_kwargs['ctt_1'],
            gamma_cum_end=diffusion_semantic_kwargs['ctt_T']
        )
        self.d3pm.set_timesteps(diffusion_steps, device)
        self.ddpm = DDPMScheduler(
            num_train_timesteps=diffusion_steps,
            prediction_type='epsilon',
            beta_start=diffusion_geometric_kwargs['beta_start'],
            beta_end=diffusion_geometric_kwargs['beta_end'],
            beta_schedule=diffusion_geometric_kwargs['schedule_type']
        )
        self.ddpm.set_timesteps(diffusion_steps, device)
        self.loss = MixedLoss(
            d3pm=self.d3pm,
            ddpm=self.ddpm,
            batch_size=batch_size,
            max_input_slots=max_input_slots,
            max_output_slots=max_output_slots,
            use_gradnorm=False,
            device=device
        )

    def build_context(self,
                      vertices,  # [B, 3, N_points]
                      furniture_points_in,  # [B, m, 3, P]
                      furniture_classes_in,  # [B, m]
                      input_mask  # [B, m]
                      ) -> torch.Tensor:
        B, m, _, P = furniture_points_in.shape

        # 1) Room encoding
        room_feats = self.encoder(vertices)
        #    room_feats:   [B, S_room, hidden_dim]

        # 2) Furniture encoding (unchanged)
        pts_flat = furniture_points_in.view(B * m, 3, P)
        shape_emb = self.furniture_proj(self.furniture_encoder(pts_flat, return_sequence=False))
        shape_emb = shape_emb.view(B, m, -1)
        sem_emb = self.semantic_embedding(furniture_classes_in)
        furn_feats = torch.cat([sem_emb, shape_emb], dim=-1)  # [B, m, hidden_dim]

        # 3) Transformer stacks
        furn_ctx = self.furn_stack(
            furn_feats,
            context=room_feats,
            key_padding_mask=~input_mask.bool()
        )  # [B, m, hidden_dim]

        # 4) Final context and its xyz
        full_ctx = torch.cat([room_feats, furn_ctx], dim=1)  # [B, S_room + m, hidden_dim]

        return full_ctx

    def forward(
            self,
            vertices: torch.Tensor,  # [B, 3, N_points]
            furniture_points_in: torch.Tensor,  # [B, m, 3, 1024]
            furniture_classes_in: torch.Tensor,  # [B, m]
            gt_output_sem: torch.Tensor,  # [B, n]
            gt_output_geo: torch.Tensor,  # [B, n, 6]
            gt_output_to_input: torch.Tensor,  # [B, n]
            input_mask: torch.Tensor,  # [B, m]
            output_mask: torch.Tensor  # [B, n]
    ):
        """
        1) build full_context,
        2) slot_attention → slot_context,
        3) run diffusion decoder with slot_context.
        """
        B = vertices.shape[0]
        # 1) random diffusion timestep
        t = torch.randint(0, self.diffusion_steps, (1,), device=self.device)

        # 2) build full context as before
        full_context = self.build_context(
            vertices,  # [B, 3, N]
            furniture_points_in,  # [B, m, 2, 3]
            furniture_classes_in,  # [B, m]
            input_mask
        )  # → [B, S_ctx + 2m, hidden_dim]

        # 3) Discrete diffusion on semantics (unchanged)…
        C = self.num_furniture_classes
        log_one_hot_out_sem = index_to_log_onehot(gt_output_sem, C + 1)  # [B, C+1, n]
        x_t_sem = self.d3pm.add_noise(log_one_hot_out_sem, t)  # [B, n]

        # 4) Continuous diffusion on geometry (unchanged)…
        noise_geo = torch.randn_like(gt_output_geo)  # [B, n, 6]
        x_t_geo = self.ddpm.add_noise(gt_output_geo, noise_geo, t)  # [B, n, 6]

        # 5) Run denoiser using slot_context instead of full_context:
        sem_logits, eps_geo_hat, ptr_logits = self.denoiser(
            x_t_sem,  # [B, n]
            x_t_geo,  # [B, n, 6]
            t.repeat(x_t_sem.shape[0]),  # [B]
            full_context,
            ~(torch.cat(
                [torch.ones((B, (full_context.shape[1] - input_mask.shape[1])), device=self.device), input_mask],
                dim=-1).bool()),
            ~output_mask.bool()
        )
        # sem_logits:   [B, n, C+1]
        # eps_geo_hat:  [B, n, 6]
        # ptr_logits:   [B, n, m]

        # 6) Compute losses exactly as before (MixedLoss takes furniture_aabb_in, etc.)
        loss_dict = self.loss(
            input_mask=input_mask,
            output_mask=output_mask,
            log_one_hot_out_sem=log_one_hot_out_sem.permute(0, 2, 1),
            x_t_sem=x_t_sem,
            noise_geo=noise_geo,
            denoise_out_sem_logits=sem_logits,
            denoise_out_geometric=eps_geo_hat,
            ptr_logits=ptr_logits,
            gt_output_to_input=gt_output_to_input,
            timestep=t
        )

        return loss_dict

    @torch.no_grad()
    def gen_samples(
            self,
            condition: torch.Tensor,
            input_mask: torch.Tensor,
            batch_size: int,
            device: str
    ):
        """
        Inference: compress `condition` → slot_context, then run reverse diffusion
        using slot_context at each step.
        """
        B = batch_size
        n = self.max_output_slots

        # 1) Initialize x_t_sem, x_t_geo
        x_t_sem = torch.full((B, n), self.num_furniture_classes + 1, dtype=torch.int64, device=device)
        x_t_geo = torch.randn((B, n, 6), device=device)

        # 3) reverse‐diffuse from t = T−1 to 0
        for timestep in reversed(self.ddpm.timesteps):
            t_vec = torch.full((B,), timestep, dtype=torch.int64, device=device)
            sem_logits_t, eps_geo_hat, ptr_logits_t = self.denoiser(
                x_t_sem,  # [B, n]
                x_t_geo,  # [B, n, 6]
                t_vec,  # [B]
                condition,
                input_mask,
                torch.zeros((B, n), device=device, dtype=torch.bool)
            )

            # a) discrete semantic step (unchanged)…
            logp = self.d3pm.log_pred_from_denoise_out(sem_logits_t)  # [B, C, n]
            x_t_sem = self.d3pm.step(logp, timestep, x_t_sem).prev_sample

            # b) continuous geometry step (unchanged)…
            x_t_geo = self.ddpm.step(eps_geo_hat, timestep, x_t_geo).prev_sample

        # 4) t=0 outputs
        x0_sem = x_t_sem  # [B, n]
        x0_geo = x_t_geo  # [B, n, 6]
        # One final pass for pointer logits
        _, _, ptr_logits = self.denoiser(
            x0_sem,
            x0_geo,
            torch.zeros((B,), device=device, dtype=torch.int64),
            condition,
            input_mask,
            torch.zeros((B, n), device=device, dtype=torch.bool)
        )
        return x0_sem, x0_geo, ptr_logits.argmax(dim=-1)

    @torch.no_grad()
    def sample(self, room_points: torch.Tensor, furniture_points: torch.Tensor, furniture_classes: torch.Tensor,
               input_mask: torch.Tensor, device: str, batch_size: int):
        _, geom, input_idx = self.gen_samples(
            condition=self.build_context(room_points, furniture_points, furniture_classes, input_mask),
            input_mask=input_mask,
            batch_size=batch_size,
            device=device
        )
        position, orientation = torch.split(geom, [3, 3], dim=2)

        return {"pred_input_idx": input_idx.cpu().numpy(), "pred_position": position.cpu().numpy(),
                "pred_orientation": orientation.cpu().numpy()}
