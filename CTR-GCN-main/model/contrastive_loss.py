import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(
            features,
            features.T
        ) / self.temperature

        logits_max, _ = torch.max(
            similarity,
            dim=1,
            keepdim=True
        )

        logits = similarity - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(
            exp_logits.sum(1, keepdim=True)
        )

        mean_log_prob_pos = (
            mask * log_prob
        ).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()

        return loss