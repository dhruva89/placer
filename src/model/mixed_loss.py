import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import multinomial_kl, log_categorical
from gradnorm_pytorch import GradNormLossWeighter

class MixedLoss(nn.Module):
    """
    MixedLoss without Hungarian matching. Assumes:
      - Input slots are pre-sorted (and padded) so that index i always corresponds
        to the i-th real furniture piece, or a PAD slot if i >= number_of_pieces.
      - Output slots are pre-sorted in the same way (and padded) so that index i
        corresponds to the i-th ground-truth object, or PAD if i >= N_out_gt.
    This lets us compute all losses slot-by-slot (no bipartite matching).
    """

    def __init__(
            self,
            d3pm,  # CustomVQDiffusionScheduler on semantics
            ddpm,  # DDPM scheduler on geometry
            batch_size: int,
            max_input_slots: int,
            max_output_slots: int,
            pad_weight: float = 0.08,
            use_gradnorm: bool = True,
            device: str = "cuda:0",
    ):
        super().__init__()
        self.d3pm = d3pm
        self.ddpm = ddpm
        self.batch_size = batch_size
        self.max_input_slots = max_input_slots
        self.max_output_slots = max_output_slots
        self.pad_weight = pad_weight
        self.use_gradnorm = use_gradnorm
        self.device = device

    def forward(
            self,
            input_mask,  # [B, m_in]       (bool or 0/1)
            output_mask,  # [B, n]          (bool or 0/1)
            log_one_hot_out_sem,  # [B, n, C]
            x_t_sem,  # [B, n]          (LongTensor of noise indices)
            noise_geo,  # [B, n, 6]
            denoise_out_sem_logits,  # [B, n, C]
            denoise_out_geometric,  # [B, n, 6]
            ptr_logits,  # [B, n, M]       (M = m_in + 1)
            gt_output_to_input,  # [B, n]          (LongTensor in [0 .. m_in])
            timestep  # scalar Tensor
    ):
        """
        Vectorized version of MixedLoss.forward, with the pointer head also learning to predict
        the “pad index” (m_in) when an output slot is empty.

        Assumes:
          - input_mask[b, i] = 1 if input‐slot i is real, 0 otherwise
          - output_mask[b, j] = 1 if output‐slot j is real (i.e. a GT object), 0 if it’s PAD
        """

        B, n, C = denoise_out_sem_logits.shape
        device = denoise_out_sem_logits.device
        m_in = self.max_input_slots
        M = m_in + 1  # pointer dimension (0..m_in); index m_in = “pad/new”

        # ——————————————————————————————————————————————————————————————————————
        # 1) Compute per-slot semantic log‐probs (batched):
        #    full_log_recon: [B, C, n]
        #    log_pred_prob:  [B, C, n]
        full_log_recon = self.d3pm.log_pred_from_denoise_out(denoise_out_sem_logits)
        log_pred_prob = self.d3pm.q_posterior(full_log_recon, x_t_sem, timestep)

        # 2) Compute semantic KL/NLL per‐slot, all B at once:
        #    log_one_hot_out_sem: [B, n, C] → [B, C, n]
        log_x_start = log_one_hot_out_sem.permute(0, 2, 1)  # [B, C, n]
        loss_sem_all = self._compute_kl(log_x_start, x_t_sem, timestep, log_pred_prob)  # [B, n]

        # 3) Compute auxiliary (multinomial KL) per‐slot, all B at once:
        finite_tgt = log_one_hot_out_sem.permute(0, 2, 1)  # [B, C, n]
        num_embed = self.d3pm.num_embed
        finite_tgt_reduced = finite_tgt[:, : (num_embed - 1), :]  # [B, C-1, n]
        finite_rec = full_log_recon  # [B, C, n]
        aux_tensor = multinomial_kl(finite_tgt_reduced, finite_rec)  # [B, n]

        # Zero‐out at t=0
        if timestep.item() == 0:
            aux_tensor.zero_()
            loss_sem_all.zero_()

        # 4) Position & orientation errors per‐slot, all B at once:
        gt_pos = noise_geo[:, :, :3]  # [B, n, 3]
        gt_ang = noise_geo[:, :, 3:6]  # [B, n, 3]
        pred_pos = denoise_out_geometric[:, :, :3]  # [B, n, 3]
        pred_ang = denoise_out_geometric[:, :, 3:6]  # [B, n, 3]

        # 4a) Position MSE per slot:
        pos_err_all = (pred_pos - gt_pos).pow(2).sum(dim=2)  # [B, n]

        # 4b) Orientation error (sin‐cos distance) per slot:
        sin_pred = torch.sin(pred_ang)  # [B, n, 3]
        cos_pred = torch.cos(pred_ang)  # [B, n, 3]
        sin_gt = torch.sin(gt_ang)  # [B, n, 3]
        cos_gt = torch.cos(gt_ang)  # [B, n, 3]
        d_sin = (sin_pred - sin_gt).pow(2)  # [B, n, 3]
        d_cos = (cos_pred - cos_gt).pow(2)  # [B, n, 3]
        ori_err_all = (d_sin + d_cos).sum(dim=2)  # [B, n]

        if timestep.item() == 0:
            pos_err_all.zero_()
            ori_err_all.zero_()

        # ——————————————————————————————————————————————————————————————————————
        # 5) Pointer cross‐entropy, but now train pad‐slots → target = m_in
        #
        # 5a) Mask out invalid pointers: any i in [N_in[b]..m_in-1] is invalid for example b
        N_in = input_mask.sum(dim=1)  # [B], #real input slots per batch‐element
        idx = torch.arange(M, device=device).view(1, 1, M)  # [1,1,M]
        N_expand = N_in.view(B, 1, 1)  # [B,1,1]
        invalid_ptr = ((idx < (M - 1)) & (idx >= N_expand))  # [B,1,M]
        invalid_ptr = invalid_ptr.expand(-1, n, -1)  # [B,n,M]
        ptr_logits_masked = ptr_logits.masked_fill(invalid_ptr, float("-9e15"))  # [B,n,M]

        # 5b) Build targets so that padded‐output slots (output_mask == 0) → target = m_in
        targets = gt_output_to_input.clone()  # [B, n], each in [0..m_in]
        pad_index = m_in
        targets[~output_mask.bool()] = pad_index

        ptr_logits_flat = ptr_logits_masked.view(B * n, M)  # [B·n, M]
        targets_flat = targets.view(B * n)  # [B·n]
        loss_ptr_flat = F.cross_entropy(
            ptr_logits_flat,
            targets_flat,
            reduction="none"
        )  # [B·n]
        loss_ptr_all = loss_ptr_flat.view(B, n)  # [B, n]

        # 6) Build weight masks for semantic+aux and geometry (unchanged),
        #    but for pointer we include both real & pad slots with different weights:
        is_real = output_mask.bool()  # [B, n]
        is_pad = ~is_real  # [B, n]

        # 6a) For semantic & aux, we still down‐weight PAD slots to pad_weight,
        #     and zero‐out entire example if it has no real slots:
        weight_sem = torch.ones_like(is_real, dtype=torch.float32, device=device)  # [B, n]
        weight_sem[is_pad] = self.pad_weight
        has_real_out = is_real.any(dim=1)  # [B]
        weight_sem = weight_sem * has_real_out.view(B, 1).float()

        # 6b) For geometry losses, only real slots count (unchanged):
        #     pos_err_all * is_real, ori_err_all * is_real

        # 6c) For pointer, we include both real & pad slots:
        weight_ptr = torch.ones_like(is_real, dtype=torch.float32, device=device)  # [B, n]
        weight_ptr[is_pad] = self.pad_weight

        # ——————————————————————————————————————————————————————————————————————
        # 7) Sum‐and‐aggregate exactly as in the loop:

        # 7a) Semantic loss
        sem_sum = (loss_sem_all * weight_sem).sum()  # scalar
        tot_sem = weight_sem.sum()  # scalar

        # 7b) Auxiliary loss
        aux_sum = (aux_tensor * weight_sem).sum()  # scalar
        tot_aux = tot_sem  # same denom as semantic

        # 7c) Position & orientation: only where is_real
        pos_sum = (pos_err_all * is_real.float()).sum()  # scalar
        tot_pos = is_real.sum()  # scalar
        ori_sum = (ori_err_all * is_real.float()).sum()  # scalar
        tot_ori = tot_pos  # same denom

        # 7d) Pointer: sum over ALL slots (real + pad) with their weights
        ptr_sum = (loss_ptr_all * weight_ptr).sum()  # scalar
        tot_ptr = weight_ptr.sum()  # scalar

        # ——————————————————————————————————————————————————————————————————————
        # 8) Compute final per‐loss terms (with eps guard):
        eps = 1e-8
        sem_loss = sem_sum / (tot_sem + eps) if (tot_sem.item() > 0) else torch.tensor(0.0, device=device)
        aux_loss = aux_sum / (tot_aux + eps) if (tot_aux.item() > 0) else torch.tensor(0.0, device=device)
        pos_loss = pos_sum / (tot_pos + eps) if (tot_pos.item() > 0) else torch.tensor(0.0, device=device)
        ori_loss = ori_sum / (tot_ori + eps) if (tot_ori.item() > 0) else torch.tensor(0.0, device=device)
        ptr_loss = ptr_sum / (tot_ptr + eps) if (tot_ptr.item() > 0) else torch.tensor(0.0, device=device)

        # 9) Apply fixed weights and return a dict:
        losses = {
            "loss.semantic": sem_loss,
            "loss.semantic_aux": aux_loss,
            "loss.geometric_position": pos_loss,
            "loss.geometric_orientation": ori_loss,
            "loss.pointer": ptr_loss
        }
        return losses

    def _compute_kl(self, log_x_start, noise_idx, t_scalar, log_pred_prob):
        """
        Compute the per-slot D3PM KL or NLL:
          - log_x_start:   [1, n, C] (log one-hot for GT at t=0)
          - noise_idx:     [1, n]   (indices ∈ [0..C-1 or PAD=C] at time t)
          - t_scalar:      single integer timestep
          - log_pred_prob: [1, C, n] (model's predicted log qθ(x₀ | xₜ, t))
        Returns:  [1, n] KL (or NLL if t=0).
        """
        log_q = self.d3pm.q_posterior(log_x_start, noise_idx, t_scalar)  # [1, C, n]
        kl = multinomial_kl(log_q, log_pred_prob)  # [1, n]
        nll = -log_categorical(log_x_start, log_pred_prob[:, :-1, :])  # [1, n]
        if t_scalar.item() == 0:
            return nll
        return kl
