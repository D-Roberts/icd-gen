"""
A modified DiT with DyT for learning energies: the EnerDiT.
"""

import torch
import torch.nn as nn


# As in Transformers without normalization
# https://github.com/jiachenzhu/DyT/blob/main/dynamic_tanh.py
class DyTanh(nn.Module):
    """"""

    def __init__(self, normalized_shape, channels_last=True, alpha_init_value=0.5):
        super().__init__()
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


# Ad-hoc test
X_train = torch.randn(4, 10, 3, requires_grad=True)  # (B, H, W,C)
dyt = DyTanh(normalized_shape=(4, 10, 3), channels_last=True, alpha_init_value=0.1)
dyt_out = dyt(X_train)

dyt_out.retain_grad()

dummy_loss = dyt_out.sum()
dummy_loss.backward()

# Let's see its grad
print(dyt_out.grad)
print(dyt_out)

"""
will start with the EnerDiT archi and then build buildinblocks.

This will not be a state of the art scale.

I will then have to find ways to train it faster and with 
less compute / one GPU.
"""
