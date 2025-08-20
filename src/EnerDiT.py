"""
A modified DiT with DyT for learning energies: the EnerDiT.
"""

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


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
        print("In tanh x shape", x.shape)
        print("In tanh weight shape ", self.weight.shape)

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


# TODO@DR: recall that I had an issue with the DiT/SiT Sin Pos Embed-
# double check what that was about. Also recall not in HF.
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

TODO@DR: determine how to encode the query patch: with the context or not
"""


class EnerDiTFinal(nn.Module):
    def __init__(self):
        super(EnerDiTFinal, self).__init__()
        pass

    def forward(self, x, y):
        """return energy from score and query patch, which is last patch
        x here will be the output of the previous layer and
        y would be the embedded noisy query (already extracted from sequence)
        """

        return 0.5 * torch.inner(x, y)


class ScoreFinalLayer(nn.Module):
    """
    this is before the Energy final final
    """

    def __init__(self, d_model, input_dim, output_dim, channels, context_len):
        super().__init__()

        self.dyt_final = DyTanh((d_model, context_len))

        # TODO@DR: not sure yet about the modulation.
        # TODO@DR: also what am I really predicting here? The socre, the energy or the
        # the clean image?
        self.final_dit_layer = nn.Linear(
            d_model, input_dim * output_dim * channels, bias=True
        )

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.dyt_final(x)
        print("shape of x as it comes out of dyt in final layer ", x.shape)

        x = self.final_dit_layer(torch.permute(x, (0, 2, 1)))
        return x


# Taking inspiration from https://github.com/facebookresearch/DiT/blob/main/models.py
class EnerDiT(nn.Module):
    """
    assume at this point that the patches are cut up and fused already.
    """

    def __init__(
        self,
        batch=4,
        context_len=5,
        d_model=32,
        input_dim=10,
        output_dim=10,
        channels=3,
        num_heads=1,
    ):
        super(EnerDiT, self).__init__()

        self.DyT = DyTanh((batch, input_dim * output_dim * channels, context_len))

        # TODO@DR: note that the DiT pos embeddings are slightly different albeit
        # still sincos; might want to come back to this, it might matter

        # Can't use the Patch embedder from timm bc my patches already come
        # in patchified and fused.

        self.embedpatch = PatchEmbedding(d_model, input_dim * output_dim * channels)
        #
        # self.embedpatch = PatchEmbed(input_dim, input_dim, channels, d_model, bias=True)
        # context_len is num of patches
        self.pos_embed = SinusoidalPositionalEmbedding(d_model, context_len)

        # TODO@DR: see about the time embedder

        # then comes the set of N EnerDiT blocks TODO@DR
        self.final_score_layer = ScoreFinalLayer(
            d_model, input_dim, output_dim, channels, context_len
        )

        # this should return the energy; use the last token
        self.final_enerdit_layer = EnerDiTFinal()

    def pre_init(self):
        """will init weights here and w whatever else I
        want to init
        """
        pass

    def forward(self, x):
        b_s, in_d, out_d, c, context_len = x.shape

        x_for_dyt = x.view(b_s, in_d * out_d * c, context_len)

        # print(x_for_dyt.shape)

        x = self.DyT(x_for_dyt)

        # x.retain_grad()  # need a hook
        # print(x.view(b_s, -1).shape)
        # . flaten for embed
        x = x.view(b_s, in_d * out_d * c, -1)
        # print("x shape after Dyt and reshape", x.shape)

        # # permute so that (b, context_len, dim)
        # permuted = torch.permute(x, (0, 2, 1))
        permuted = torch.permute(x, (0, 2, 1))
        # print("permuted shape", permuted.shape)
        patch_embed = self.embedpatch(permuted)

        # reshape for patch_embed
        # x = x.view(b_s, in_d, out_d, c, context_len)
        # TODO@DR: this won't work because I have H different from W
        # x = torch.permute(x, (0, 3, 1, 2, 4))
        # print("x shape", x.shape)

        # patch_embed = self.embedpatch(x)

        # TODO@DR this isn't working well here on shapes fix
        # patch_embed = self.embedpatch(x.view(b_s, c, 250, -1))
        # print("patch_embed pos ", patch_embed.shape)

        pos_embed = self.pos_embed(
            patch_embed
        )  # [1, 10, 32] will add same order each batch
        # print(f"pos embed shape********* {pos_embed.shape}")

        embedded = patch_embed + pos_embed

        # add enerdit blocks

        # add final (score out layer)
        score = self.final_score_layer(embedded)
        # add enerditfinal
        print("score shape ", score.shape)

        # this is for all patches
        # y = x[:,:,:,-1].view()
        energy = self.final_enerdit_layer(score, x.view(b_s, context_len, -1))

        # only for last patch the noisy query
        return energy[:, :, -1]


# AdHoc testing
X_train = torch.randn(4, 10, 10, 3, 5, requires_grad=True)  # (B, ,C, context_len)
model = EnerDiT()
energy = model(X_train)
energy.retain_grad()


dummy_loss = energy.sum()
dummy_loss.backward()
# print(energy.grad)


class TimeLoss(nn.Module):
    def __init__(self):
        super(TimeLoss, self).__init__()

    def forward(self, preds, label):
        pass


class SpaceLoss(nn.Module):
    def __init__(self):
        super(SpaceLoss, self).__init__()
        pass

    def forward(self, preds, labels):
        pass


class Trainer:
    """
    To train the EnerDiT
    """

    def __init__(
        self,
    ):
        pass

    def train(
        self,
    ):
        pass

    def eval(
        self,
    ):
        pass
