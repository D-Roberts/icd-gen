"""
Archi step by step debug process.
"""

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torch.nn.functional as F


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
        # print("In tanh x shape", x.shape)
        # print("In tanh weight shape ", self.weight.shape)

        # TODO@DR: put the weight and bias back later but need to match the shapes

        # if self.channels_last:
        #     x = x * self.weight + self.bias
        # else:
        #     x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


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
    def __init__(self, dim_in, d_model):
        super().__init__()

        # aim to embed the clean and noisy together for dim reduction
        # alternatively I could add the position embed directly in the
        # 200 dim

        # set bias to zero
        self.projection = torch.nn.Linear(dim_in, d_model, bias=False)

    def forward(self, x):
        # print(f"layer in patch embed {self.projection}")
        # print(f"debug patch embed shapes {x.shape}")
        x = self.projection(x)
        # as expected (B, seq_len, d_model=embeddim)
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
EnerdiT archi and buildinblocks.

This will not be a state of the art scale bc of low on time.

I will then have to find ways to train it faster and with 
less compute / one GPU bc of low on time to deadline.

"""


class EnerdiTFinal(nn.Module):
    def __init__(self):
        super(EnerdiTFinal, self).__init__()
        pass

    def forward(self, sh, th, y, cf1):
        """return energy for each token in context as well as
        query token (last) TODO@DR reason about the zero padding

        use the corrected s in U = 0.5<s, y>

        cf1 is a correction factor. SHould be the ratio of
        eps y and eps t which should give sensitivity of
        the true score net to y and t. Let's assume small numbers

        and take cf1 as hyperparameter

        there was a cf2 correction factor as well but let's
        ignore it for now

        return : energy for each token in context and the query token
        """

        # TODO @DR: Should cf1 be analytically estimated / approx
        # a hyperparam or learned in here jointly?

        sc = sh - th * cf1

        # TODO@DR: check that energy is calc on the right dims
        energy = 0.5 * torch.sum(sc * y, dim=(-1))
        return energy


class FinalLayer(nn.Module):
    """
    this is before the two heads
    """

    def __init__(self, d_model, input_dim, context_len):
        super().__init__()

        self.dyt_final = DyTanh((d_model, context_len))

        # TODO@DR: not sure yet about the modulation.
        # TODO: @DR a potential differentiable one.

        # TODO@DR: COnsider adding a Silu here also

        self.final_dit_layer = nn.Linear(d_model, input_dim, bias=False)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.dyt_final(x)
        # print("shape of x as it comes out of dyt in final layer ", x.shape)

        x = self.final_dit_layer(torch.permute(x, (0, 2, 1)))
        return x


# TODO@DR: I'll just do them separatelly and will see what if
# anything different in each


class TimeHead(nn.Module):
    """
    with a silu and a dyt as in pre-norm but no resid connections
    """

    def __init__(self, d_model, input_dim, context_len):
        super().__init__()

        self.dyt_time = DyTanh((d_model, context_len))
        self.silu = nn.SiLU()
        self.time_head = nn.Linear(d_model, input_dim, bias=True)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.dyt_time(x)
        # print("shape of x as it comes out of dyt in final layer ", x.shape)
        x = self.silu(x)
        x = self.time_head(torch.permute(x, (0, 2, 1)))
        return x


class SpaceHead(nn.Module):
    """
    with a silu and a dyt as in pre-norm but no resid connections
    """

    def __init__(self, d_model, input_dim, context_len):
        super().__init__()

        self.dyt_space = DyTanh((d_model, context_len))
        self.silu = nn.SiLU()
        self.space_head = nn.Linear(d_model, input_dim, bias=True)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.dyt_space(x)
        # print("shape of x as it comes out of dyt in final layer ", x.shape)
        x = self.silu(x)
        x = self.space_head(torch.permute(x, (0, 2, 1)))
        return x


class EnerdiTBlock(nn.Module):
    """
    no layernorm or adaptvie ones for now.
    dyt
    """

    def __init__(self, d_model, context_len, num_heads=1, mlp_ratio=4.0):  # as in DiT
        super().__init__()

        # TODO@DR: will have to check on all the logic of where dyt gets applyied
        self.dyt1 = DyTanh((d_model, context_len))
        self.attn = Attention(
            d_model, num_heads, qkv_bias=True
        )  # TODO@DR there are some kwargs here will have to check them out
        self.dyt2 = DyTanh((d_model, context_len))

        mlp_hidden_dim = int(0.4 * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        # TODO@DR: will have to checkout the timm implementations as I typically
        # work with HF

    def forward(self, x):
        # not modulating for now
        x = torch.permute(x, (0, 2, 1))
        # print(f"shape of x in enerditblock after reshape", x.shape) # B, d_model, context_len
        x = self.dyt1(x)
        x = torch.permute(x, (0, 2, 1))  # for attn
        x = x + self.attn(x)
        x = torch.permute(x, (0, 2, 1))  # for dyt TODO@DR: refactor DyT layer
        x = self.dyt2(x)
        x = torch.permute(x, (0, 2, 1))  # for mlp
        x = x + self.mlp(x)
        return x


# Taking inspiration from https://github.com/facebookresearch/DiT/blob/main/models.py
class EnerdiT(nn.Module):
    """
    assume at this point that the patches are cut up and fused already.
    """

    def __init__(
        self,
        batch,
        context_len,
        d_model,
        # input_dim=10,
        # output_dim=10,
        # channels=3,
        input_dim,  # the way datagen is setup now - comes in one flattened
        cf1_init_value=0.1,
        num_heads=1,
        depth=1,
        mlp_ratio=4,
    ):
        super(EnerdiT, self).__init__()

        # self.DyT = DyTanh((batch, input_dim, context_len))

        # TODO@DR: note that the DiT pos embeddings are slightly different albeit
        # still sincos; might want to come back to this, it might matter

        # Can't use the Patch embedder from timm bc my patches already come
        # in patchified and fused.

        self.patch_embed = PatchEmbedding(input_dim, d_model)

        # print(f"patch embed layer in EnerdiT {self.patch_embed}")
        #
        # self.embedpatch = PatchEmbed(input_dim, input_dim, channels, d_model, bias=True)
        # context_len is num of patches
        self.pos_embed = SinusoidalPositionalEmbedding(d_model, context_len)

        ######################################Before this - inputs embedding

        # TODO@DR: see about the time embedder

        # then comes the list of N EnerdiT blocks

        # TODO@DR consider a Silu and DyT here maybe - silu were working so well - check that diff

        # self.blocks = nn.ModuleList(
        #     [EnerdiTBlock(d_model, num_heads, mlp_ratio) for _ in range(depth)]
        # )

        # # correction factor space time scores
        self.corf = nn.Parameter(torch.ones(1) * cf1_init_value)
        self.final_enerdit_layer = EnerdiTFinal()

        # self.prehead_linear = nn.Linear(d_model, d_model, bias=False)

        self.time_head = TimeHead(d_model, input_dim, context_len)
        self.space_head = SpaceHead(d_model, input_dim, context_len)

    def forward(self, x):
        print("what shape comes the batch into Enerdit ", x.shape)
        b_s, in_d, context_len = x.shape
        # in_d is patch_dim which is like c*w * h of each patch and here * 2 because of fused

        # x_for_dyt = torch.permute(x, (0, 2, 1))

        # print(x_for_dyt.shape)

        # x = self.DyT(x_for_dyt)

        # x.retain_grad()  # need a hook
        # print("x shape after Dyt and reshape", x.shape)
        # (b, context, dim)

        # # permute so that (b, context_len, dim)
        # permuted = torch.permute(x, (0, 2, 1))
        # print("permuted shape", permuted.shape)

        patch_embed = self.patch_embed(x)

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

        # TODO@DR - so far
        x = patch_embed + pos_embed

        # add enerdit blocks
        # for block in self.blocks:
        #     x = block(x)

        # add final (score out layer)
        # TODO@DR: there should be another linear here with DyT and silu

        # x = self.prehead_linear(x)

        space_score = self.space_head(x)
        time_score = self.time_head(x)

        # add enerditfinal
        # print("score shape ", score.shape)

        # print(f"what is out of block shape", score.shape) #(b, context_len, in_dim)
        # print(f"what is the query going into energy layer ", x_for_dyt.shape)

        # sh, th, y, cf1 - learn it
        # y is the noised
        energy = self.final_enerdit_layer(space_score, time_score, x, self.corf)
        # print(f"what is {self.corf}")

        # TODO@DR - reason through context next toward loss

        # so this will now output the energy for each token with last
        # one being for query; RIght now train code is setup to
        # predict the clean image so do that for a first train run

        return energy, space_score, time_score


# AdHoc testing
# X_train = torch.randn(4, 100, 3, 5, requires_grad=True)  # (B, ,C, context_len)
# model = EnerdiT()
# energy = model(X_train)
# energy.retain_grad()


# dummy_loss = energy.sum()
# dummy_loss.backward()
# print(energy.grad)
