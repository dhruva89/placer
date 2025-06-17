import torch.nn as nn


class InputSemanticClassEmbedding(nn.Module):
    def __init__(self, num_furniture_classes=36, embedding_dim=512):
        super(InputSemanticClassEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_furniture_classes + 1, embedding_dim, padding_idx=num_furniture_classes)

    def forward(self, whitelist_tensor):
        # Expect whitelist_tensor to be of type torch.long and shape [B, L]
        # Output shape: [B, L, embedding_dim]
        return self.embedding(whitelist_tensor)
