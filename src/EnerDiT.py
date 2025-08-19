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
# X_train = torch.randn(4, 10, 10, 3, requires_grad=True)  # (B,H, W ,C)
# dyt = DyTanh(normalized_shape=(4, 10, 10, 3), channels_last=True, alpha_init_value=0.1)
# dyt_out = dyt(X_train)

# dyt_out.retain_grad()

# dummy_loss = dyt_out.sum()
# dummy_loss.backward()

# Let's see its grad
# print(dyt_out.grad)
# print(dyt_out.shape) # (4, 10, 10, 3)

# Embed


# now the sequence has double width due to concat clean and noisy
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, dim_in):
        super().__init__()

        # aim to embed the clean and noisy together for dim reduction
        # alternatively I could add the position embed directly in the
        # 200 dim

        # set bias to zero
        self.projection = torch.nn.Linear(dim_in, embed_dim, bias=False)

    def forward(self, x):
        x = self.projection(x)
        # (batch_size, embed_dim, num_patches)
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # TODO@DR check for correctness again
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # print("torch.sin(position * div_term)", torch.sin(position * div_term).shape)
        # print("pe[:, 0::2]", pe[:, 0::2].shape)
        # print("pe[:, 1::2]", pe[:, 1::2].shape)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # print("pe shape", pe.shape)
        self.register_buffer("pe", pe.unsqueeze(0))  # Add a batch dimension

    def forward(self, x):
        # x is the sequence of patch embeddings, shape (batch_size, seq_len, d_model)
        # We need to add positional embeddings to each element in the sequence
        # for the moment only create the pos embeddings

        return self.pe[:, : x.size(1)]


"""
will start with the EnerDiT archi and then build buildinblocks.

This will not be a state of the art scale.

I will then have to find ways to train it faster and with 
less compute / one GPU.

Start very simple and build up.
"""


class EnerDiT(nn.Module):
    def __init__(self, batch, input_dim, output_dim, channels):
        super(EnerDiT, self).__init__()

        self.DyT = DyTanh((batch, input_dim, output_dim, channels))

        self.linear = nn.Linear(input_dim * output_dim * channels, output_dim)

    def forward(self, x):
        b_s, in_d, out_d, c = x.shape
        x = self.DyT(x)
        x.retain_grad()  # need a hook
        # print(x.view(b_s, -1).shape)

        return self.linear(x.view(b_s, -1))


# AdHoc testing
X_train = torch.randn(4, 10, 10, 3, requires_grad=True)  # (B, ,C)
model = EnerDiT(4, 10, 10, 3)
model_out = model(X_train)
model_out.retain_grad()


dummy_loss = model_out.sum()
dummy_loss.backward()
print(model_out.grad)
