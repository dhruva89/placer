import torch
from diffusers import VQDiffusionScheduler
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_vq_diffusion import gumbel_noised, index_to_log_onehot

from src.utils import log_onehot_to_index, LOG_ZERO
import torch.nn.functional as F


class CustomVQDiffusionScheduler(VQDiffusionScheduler):

    @register_to_config
    def __init__(self, num_vec_classes: int, num_train_timesteps: int = 100, alpha_cum_start: float = 0.99999,
                 alpha_cum_end: float = 0.000009, gamma_cum_start: float = 0.000009, gamma_cum_end: float = 0.99999):
        super().__init__(num_vec_classes, num_train_timesteps, alpha_cum_start, alpha_cum_end, gamma_cum_start,
                         gamma_cum_end)
        self.mask_class = self.num_embed - 1
        self.empty_class = self.num_embed - 2  # index for the new empty class
        self.num_train_timesteps = num_train_timesteps

    def log_sample_categorical(self, logits, num_classes):
        with_gumbel_noise = gumbel_noised(logits, None)
        sample = with_gumbel_noise.argmax(dim=1)
        log_sample = index_to_log_onehot(sample, num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        # log_x_start from MixedDiffusion is [B, n, C] – convert to [B, C, n]
        log_EV_qxt_x0 = self.apply_cumulative_transitions(
            log_x_start, t  # ← correct orientation
        )
        log_sample = self.log_sample_categorical(log_EV_qxt_x0, self.num_embed)
        return log_sample  # stays [B, C, n]

    @staticmethod
    def log_pred_from_denoise_out(denoise_out):
        """
        convert output of denoising network to log probability over classes and [mask]
        """
        out = denoise_out.permute((0, 2, 1))
        B, _, N = out.shape

        log_pred = F.log_softmax(out.float(), dim=1).float()
        log_pred = torch.clamp(log_pred, LOG_ZERO, 0)
        return log_pred

    def add_noise(
            self,
            one_hot_original_samples: torch.Tensor,
            timesteps: torch.Tensor,
    ):
        log_xt = self.q_sample(log_x_start=one_hot_original_samples, t=timesteps)
        x_t = log_onehot_to_index(log_xt)

        return x_t
