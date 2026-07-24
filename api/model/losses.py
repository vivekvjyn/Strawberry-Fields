import torch
import torch.nn as nn
import torch.nn.functional as F


class NxLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, keys):
        query = F.normalize(query, dim=1)

        if keys.dim() == 3:
            B, N, D = keys.shape
            keys = keys.view(B * N, D)

        keys = F.normalize(keys, dim=1)
        logits = torch.mm(query, keys.t()) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)

        return F.cross_entropy(logits, labels)


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        logits = torch.mm(features, features.t()) / self.temperature

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=mask.device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        return -mean_log_prob.mean()
