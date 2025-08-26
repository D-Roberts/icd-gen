"""
Archi step by step debug process.
"""

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torch.nn.functional as F


# Similar to Transformers without normalization
class DyTanh(nn.Module):
    """
    dev dyt

    expected normalized shape would be (seq_len, input_dim)

    elementwise layer; so shape out is like shape in.
    """

    def __init__(self, shape_in, alpha_init_value=0.5):
        super().__init__()
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(shape_in))
        # print("tanh weight shape", self.weight.shape)
        self.bias = nn.Parameter(torch.zeros(shape_in))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        # print("In tanh x shape", x.shape)
        x = x * self.weight + self.bias
        return x


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


# this is the more standard sin pos embed; diff a bit from dit
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (seq_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # double the d_model size

        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # start index at 0 with step size 2
        # print("pe[:, 0::2] and pe", pe[:, 0::2].shape, pe)
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # start index at 1 with step size 2
        self.register_buffer("pe", pe.unsqueeze(0))  # Add a batch dimension

    def forward(self, x):
        # x is the sequence of patch embeddings, shape (batch_size, seq_len, d_model)
        # We need to add positional embeddings to each element in the sequence

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

        and take cf1 as another learned param

        there was a cf2 correction factor as well but let's
        ignore it for now

        return : energy for each token in context and the query token
        """

        # TODO@DR: will have to see about shapes and signs
        # sc = sh - th * cf1

        # TODO@DR: check that energy is calc on the right dims
        # energy = 0.5 * torch.sum(sc * y, dim=(-1))

        # let's see first with only space head
        energy = 0.5 * torch.sum(sh * y, dim=(-1))

        return energy


class PreHead(nn.Module):
    """
    this is before the two heads
    """

    def __init__(self, context_len, d_model):
        super().__init__()

        self.dyt_final = DyTanh((context_len, d_model))
        self.silu = nn.SiLU()

        # TODO@DR: not sure yet about the modulation.
        # TODO: @DR a potential differentiable one.
        # TODO@DR also the conditioning on context - might try different
        # ways like a cross-attn layer added in at differnt layers

        self.prehead_layer = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # print("x shape ans it comes into dyt_final", x.shape)
        x = self.dyt_final(x)
        # print("shape of x as it comes out of dyt in final layer ", x.shape)
        x = self.silu(x)
        x = self.prehead_layer(x)
        return x


# TODO@DR: I'll just do them separatelly and will see what if
# anything different in each


class TimeHead(nn.Module):
    """
    with a silu and a dyt as in pre-norm but no resid connections
    as of right now
    """

    def __init__(self, d_model, input_dim, context_len):
        super().__init__()

        self.dyt_time = DyTanh((context_len, d_model))
        self.silu = nn.SiLU()
        self.time_head = nn.Linear(d_model, input_dim, bias=True)

    def forward(self, x):
        x = self.dyt_time(x)
        # print("shape of x as it comes out of dyt in final layer ", x.shape)
        x = self.silu(x)
        x = self.time_head(x)

        return x


class SpaceHead(nn.Module):
    """
    with a silu and a dyt as in pre-norm but no resid connections
    """

    def __init__(self, d_model, input_dim, context_len):
        super().__init__()

        self.dyt_space = DyTanh(shape_in=(context_len, d_model))
        self.silu = nn.SiLU()
        self.space_head = nn.Linear(d_model, input_dim, bias=True)

    def forward(self, x):
        # print(f"shape in head {x.shape}") # [3, 8, 4] is (B, seq_len, d_model)
        x = self.dyt_space(x)
        x = self.silu(x)
        x = self.space_head(x)
        return x


class EnerdiTBlock(nn.Module):
    """
    no layernorm or adaptvie ones for now.
    dyt
    """

    def __init__(self, d_model, context_len, num_heads=1, mlp_ratio=4.0):  # as in DiT
        super().__init__()

        # TODO@DR: will have to check on all the logic of where dyt gets applyied
        self.dyt1 = DyTanh((context_len, d_model))

        self.attn = Attention(
            d_model, num_heads, qkv_bias=True
        )  # TODO@DR there are some kwargs here will have to check them out

        self.dyt2 = DyTanh((context_len, d_model))

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
        x = self.dyt1(x)
        x = x + self.attn(x)
        x = self.dyt2(x)
        x = x + self.mlp(x)
        return x


# Taking inspiration from https://github.com/facebookresearch/DiT/blob/main/models.py
class EnerdiT(nn.Module):
    """
    assume at this point that the patches are cut up and fused already.
    """

    def __init__(
        self,
        context_len,
        d_model,
        # input_dim=10, # right now the patches come in flattened from datagen
        # output_dim=10,
        # channels=3,
        input_dim,  # the way datagen is setup now - comes in one flattened
        cf1_init_value=0.1,
        num_heads=1,
        depth=1,
        mlp_ratio=4,
    ):
        super(EnerdiT, self).__init__()

        self.DyT = DyTanh((context_len, input_dim))

        # Can't use the Patch embedder from timm bc my patches already come
        # in patchified and fused.
        self.patch_embed = PatchEmbedding(input_dim, d_model)
        self.pos_embed = SinusoidalPositionalEmbedding(d_model, context_len)

        ######################################Before this - inputs embedding

        # TODO@DR: see about the time embedder

        # TODO@DR consider a Silu and DyT here maybe as well; though right now the block starts
        # with dyt

        # then comes the list of N EnerdiT blocks
        self.blocks = nn.ModuleList(
            [EnerdiTBlock(d_model, num_heads, mlp_ratio) for _ in range(depth)]
        )

        # # correction factor space time scores
        self.corf = nn.Parameter(torch.ones(1) * cf1_init_value)
        self.final_enerdit_layer = EnerdiTFinal()

        self.prehead_linear = PreHead(context_len, d_model)

        # a kind of unembed
        self.time_head = TimeHead(d_model, input_dim, context_len)
        self.space_head = SpaceHead(d_model, input_dim, context_len)

        # self.pre_init() #TODO@DR see later about custom init - right now is hurting

    def pre_init(self):
        """will init weights here however way I want and whatever else I
        want to init

        for now the linears and the final but not final init
        """

        # TODO@DR: this is in spirit of DiT but I am not convinced I'll stay
        # with this
        def lin_init(module):
            if isinstance(module, nn.Linear):
                # TODO@DR: reconsider later: not learning well with this below at this time in dev
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(lin_init)

        # zero out out layer on the heads TODO@DR: think this again

        nn.init.constant_(self.space_head.space_head.bias, 0)
        nn.init.constant_(self.time_head.time_head.bias, 0)

        # TODO@DR: reconsider later: not learning well with this below at this time
        # nn.init.constant_(self.space_head.space_head.weight, 0)
        # nn.init.constant_(self.time_head.time_head.weight, 0)
        # print(f"am I doing this? {self.time_head.time_head.weight}")

    def forward(self, x):
        # print("what shape comes the batch into Enerdit ", x.shape)
        b_s, in_d, context_len = x.shape
        # in_d is patch_dim which is like c*w * h of each patch and here * 2 because of fused

        # reshape for embedding and dyt
        # B, seq_len, fused patch dim
        x_for_dyt = torch.permute(x, (0, 2, 1))
        x = self.DyT(x_for_dyt)

        # print("x shape after Dyt and reshape", x.shape)
        # (b, context, dim)

        patch_embed = self.patch_embed(x)
        pos_embed = self.pos_embed(
            patch_embed
        )  # [1, seq_len, d_model] will add same order each instance in batch

        x = patch_embed + pos_embed

        ##################Input normalization and embedding area over

        # add enerdit blocks
        for block in self.blocks:
            x = block(x)

        x = self.prehead_linear(x)
        space_score = self.space_head(x)
        time_score = self.time_head(x)  # the time head is the same shape now
        # print(f"th is now {time_score.shape}")

        # TODO@DR: what shapes should the scores be? Well this will
        # be determined in the loss###########################

        # sh, th, y, cf1 - learn it
        # y is the noised in theory but here is called x, the noised query and context tokens
        energy = self.final_enerdit_layer(space_score, time_score, x_for_dyt, self.corf)
        # print(f"what is {self.corf}")

        # TODO@DR - reason through context next toward loss

        return energy, space_score, time_score


# AdHoc testing
# X_train = torch.randn(4, 100, 3, 5, requires_grad=True)  # (B, ,C, context_len)
# model = EnerdiT()
# energy = model(X_train)
# energy.retain_grad()


# dummy_loss = energy.sum()
# dummy_loss.backward()
# print(energy.grad)
