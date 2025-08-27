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
        # alternatively I could add the space embed directly in the
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
# I decided to call them Space Embeddings. Seem to me more apt
# Since Pos was from language really
class SpaceEmbedding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        se = torch.zeros(max_len, d_model)  # (seq_len, d_model)
        spatial = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # double the d_model size

        se[:, 0::2] = torch.sin(spatial * div_term)  # start index at 0 with step size 2
        # print("pe[:, 0::2] and pe", pe[:, 0::2].shape, pe)
        se[:, 1::2] = torch.cos(spatial * div_term)  # start index at 1 with step size 2
        self.register_buffer("se", se.unsqueeze(0))  # Add a batch dimension

    def forward(self, x):
        # x is the sequence of patch embeddings, shape (batch_size, seq_len, d_model)
        # We need to add space embeddings to each element in the sequence

        return self.se[:, : x.size(1)]


# inspired from https://github.com/facebookresearch/DiT/blob/main/models.py#L26
class TimeEmbedding(nn.Module):
    """We just have to have them.

    logt will be drawn from a U(logtmin, logtmax), will tmin = 10**-9 and tmax = 10**3

    each token in sequence will have an associated t. embed to same d_model and add to the
    pathc and space embeddings.

    embeddings are sin cos

    t is a seq len vector in this formulation with a context prompt.
    (as of right now)

    """

    def __init__(
        self,
        d_model,
        frequency_embedding_size=256,
        mint=1 / (10**3),
        maxt=1 / (10 ** (-9)),
    ):
        super().__init__()
        self.time_embedder = nn.Sequential(
            nn.Linear(frequency_embedding_size, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.mint = mint
        self.maxt = maxt

    @staticmethod
    def time_embedding(t, dim, mint, maxt):
        half = dim // 2
        freqs = torch.exp(
            -math.log(maxt - mint)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(
            device=t.device
        )  # the device will need to be given
        args = t[:, None].float() * freqs[None]  # for each t in the t tensor
        # print(args)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding  # this will be shape (len of t, 256)

    def forward(self, t):
        t_freq = self.time_embedding(
            t, self.frequency_embedding_size, self.mint, self.maxt
        )
        time_embed = self.time_embedder(t_freq)
        return time_embed


"""
EnerdiT archi and buildinblocks.

This will not be a state of the art scale bc of low on time.

I will then have to find ways to train it /make it work faster and with 
less compute / one GPU bc of low on time to deadline.

"""


class EnerdiTFinal(nn.Module):
    def __init__(self):
        super(EnerdiTFinal, self).__init__()
        pass

    def forward(self, sh, th, noisy, cf1):
        """

        U = 0.5( <s, y> - correction) TODO@DR reconsider math for this
        correction

        correction = cf1 * time score

        cf1 learned

        return : energy for query token
        """
        # only use the non-zero portion of the noisy query

        noisy = torch.permute(noisy, (0, 2, 1))
        sh = torch.permute(sh, (0, 2, 1))
        # print(f"in energy calculation noisy shape {noisy.shape}")
        # print(f"in energy calculation space score shape {sh.shape}")

        bs, d, seq_len = noisy.shape
        d = d // 2

        # TODO@DR reason again if the first part is the non-zero one
        query = noisy[:, :d, -1]
        sp_pred = sh[:, :, -1]  # the space head should already have d // 2

        # print(f"sp_pred in energy {sp_pred.shape}")
        # print(f"th shape in energy {th.shape}") #(B, seqlen)
        # should be scalar according to theory
        th_pred = th[:, -1]
        # print(f"th_pred shape in energy {th_pred.shape}") #(B,)

        # This is not right because th_pred is a scalar
        # sc = sh - th * cf1

        # TODO@DR: I am adding the correction after inner prod
        # so if this works empirically need to rework theory math
        # for this approximation

        correction = cf1 * th_pred
        # TODO@DR: will have to see about signs; query is the noisy
        energy = 0.5 * (torch.sum(sp_pred * query, dim=(-1)) - correction)

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

        # print("shape of x as it comes out of time head ", x.shape)
        # (B, seq_len, patch dim)
        # add an average pooling layer in time head since it is supposed
        # to be a U partial wrt t, a scalar
        return x.mean(-1)


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
        self.space_embed = SpaceEmbedding(d_model, context_len)
        self.time_embed = TimeEmbedding(
            d_model,
            frequency_embedding_size=256,
            mint=1 / (10**3),
            maxt=1 / (10 ** (-9)),
        )

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

        # a kind of unembed; aim to use only on the non-zero part of
        # the query and clean label
        self.time_head = TimeHead(d_model, input_dim // 2, context_len)
        self.space_head = SpaceHead(d_model, input_dim // 2, context_len)

        # self.pre_init() #TODO@DR see later about custom init - right now is hurting

    def pre_init(self):
        """will init weights here however way I want and whatever else I
        want to init

        for now the linears and the final but not final init
        """

        # TODO@DR: this is in spirit of DiT but I am not convinced I'll stay
        # with this
        # def lin_init(module):
        #     if isinstance(module, nn.Linear):
        #         # TODO@DR: reconsider later: not learning well with this below at this time in dev
        #         nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)

        # self.apply(lin_init)

        # zero out out layer on the heads TODO@DR: think this again

        # nn.init.constant_(self.space_head.space_head.bias, 0)
        # nn.init.constant_(self.time_head.time_head.bias, 0)

        # TODO@DR: reconsider later: not learning well with this below at this time
        # nn.init.constant_(self.space_head.space_head.weight, 0)
        # nn.init.constant_(self.time_head.time_head.weight, 0)
        # print(f"am I doing this? {self.time_head.time_head.weight}")

    def forward(self, x, t):
        # print("what shape comes the batch into Enerdit ", x.shape)
        b_s, in_d, context_len = x.shape

        assert t.shape[0] == x.shape[-1]
        # in_d is patch_dim which is like c*w * h of each patch and here * 2 because of fused

        # reshape for embedding and dyt
        # B, seq_len, fused patch dim
        x_for_dyt = torch.permute(x, (0, 2, 1))
        x = self.DyT(x_for_dyt)

        # print("x shape after Dyt and reshape", x.shape)
        # (b, context, dim)

        patch_embed = self.patch_embed(x)
        space_embed = self.space_embed(
            patch_embed
        )  # [1, seq_len, d_model] will add same order each instance in batch
        time_embed = self.time_embed(t)

        x = patch_embed + space_embed + time_embed

        ##################Input normalization and embedding area over

        # add enerdit blocks
        for block in self.blocks:
            x = block(x)

        x = self.prehead_linear(x)
        space_score = self.space_head(x)
        time_score = self.time_head(x)  # the time head is the same shape now
        # so take mean of it in energy since it should be a scalar

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
