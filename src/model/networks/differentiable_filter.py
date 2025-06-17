import torch
import torch.nn as nn

import torch.nn.functional as F


class DifferentiableFilter(nn.Module):
    def __init__(self, max_furniture_pieces, threshold_init=0.6, tau_filter=0.1):
        super().__init__()
        # Learnable threshold parameter
        self.threshold = nn.Parameter(torch.tensor(threshold_init, dtype=torch.float32))
        self.small_value = nn.Parameter(torch.tensor(-10.0, dtype=torch.float32))
        self.large_value = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        self.tau_filter = max(tau_filter, 0.05)  # Prevent tau from being too small
        self.max_furniture_pieces = max_furniture_pieces

    def forward(self, tensor_a, tensor_b, scores):
        """
        Args:
            tensor_a: Tensor of shape (B, T)
            tensor_b: Tensor of shape (B, T, C)
            scores: Tensor of shape (B, T) used for filtering.

        Returns:
            filtered_a: Tensor_a filtered with values approaching -1 where the mask is near false.
            filtered_b: Tensor_b filtered using the soft mask.
        """

        # Ensure threshold stays within a reasonable range
        self.threshold.data = torch.clamp(self.threshold, 0.3, 0.9)

        furniture_biased = torch.full_like(tensor_a, self.small_value.item())
        furniture_biased[..., self.max_furniture_pieces] = self.large_value

        soft_mask = torch.sigmoid((scores - self.threshold) / self.tau_filter)
        soft_mask_3d = soft_mask.unsqueeze(-1)

        # Stabilized filtering
        filtered_a = tensor_a * soft_mask_3d + furniture_biased * (1.0 - soft_mask_3d)
        filtered_b = tensor_b * soft_mask_3d

        return filtered_a, filtered_b
