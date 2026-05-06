import torch
import torch.nn as nn


class TemporalAttention(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv1d(
            channels,
            channels // 4,
            kernel_size=1
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            channels // 4,
            channels,
            kernel_size=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        """
        x shape:
        (N, C, T, V)
        """

        # Average over joints
        attn = x.mean(dim=3)

        # (N, C, T)

        attn = self.conv1(attn)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        # (N, C, T, 1)
        attn = attn.unsqueeze(-1)
        # Apply attention
        out = x * attn
        return out, attn