"""
A pure transformer architecture for learning energy based densities
in context -  the EnerdiT.
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


class EnerdiTFinal(nn.Module):
    def __init__(self):
        super(EnerdiTFinal, self).__init__()
        pass

    def forward(self, x, y):
        """return energy from score and query patch, which is last patch
        x here will be the output of the previous layer and
        y would be the noisy query (already extracted from sequence)


        """

        return 0.5 * (x * y).sum(dim=-1)


class ScoreFinalLayer(nn.Module):
    """
    this is before the Energy final final
    """

    def __init__(self, d_model, input_dim, context_len):
        super().__init__()

        self.dyt_final = DyTanh((d_model, context_len))

        # TODO@DR: not sure yet about the modulation.
        # TODO: @DR a potential differentiable one.

        # TODO@DR: also what am I really predicting here? The socre, the energy or the
        # the clean image?
        self.final_dit_layer = nn.Linear(d_model, input_dim, bias=True)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.dyt_final(x)
        # print("shape of x as it comes out of dyt in final layer ", x.shape)

        x = self.final_dit_layer(torch.permute(x, (0, 2, 1)))
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
        num_heads=1,
        depth=1,
        mlp_ratio=4,
    ):
        super(EnerdiT, self).__init__()

        self.DyT = DyTanh((batch, input_dim, context_len))

        # TODO@DR: note that the DiT pos embeddings are slightly different albeit
        # still sincos; might want to come back to this, it might matter

        # Can't use the Patch embedder from timm bc my patches already come
        # in patchified and fused.

        self.embedpatch = PatchEmbedding(d_model, input_dim)
        #
        # self.embedpatch = PatchEmbed(input_dim, input_dim, channels, d_model, bias=True)
        # context_len is num of patches
        self.pos_embed = SinusoidalPositionalEmbedding(d_model, context_len)

        # TODO@DR: see about the time embedder

        # then comes the list of N EnerDiT blocks

        self.blocks = nn.ModuleList(
            [EnerdiTBlock(d_model, num_heads, mlp_ratio) for _ in range(depth)]
        )

        # TODO@DR: Time and space head for now identical but probably
        # should not be
        self.space_head = ScoreFinalLayer(d_model, input_dim, context_len)
        self.time_head = ScoreFinalLayer(d_model, input_dim, context_len)

        # this should return the energy; use the last token
        self.final_enerdit_layer = EnerdiTFinal()
        self.pre_init()

    def pre_init(self):
        """will init weights here and w whatever else I
        want to init
        """
        pass

    def forward(self, x):
        print("what shape comes the batch into Enerdit ", x.shape)
        b_s, in_d, context_len = x.shape

        x_for_dyt = torch.permute(x, (0, 2, 1))

        # print(x_for_dyt.shape)

        x = self.DyT(x_for_dyt)

        # x.retain_grad()  # need a hook
        # print(x.view(b_s, -1).shape)
        # . flaten for embed

        # print("x shape after Dyt and reshape", x.shape)

        # # permute so that (b, context_len, dim)
        # permuted = torch.permute(x, (0, 2, 1))
        # permuted = torch.permute(x, (0, 2, 1))
        # print("permuted shape", permuted.shape)
        patch_embed = self.embedpatch(x)

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
        for block in self.blocks:
            x = block(x)

        # add final (score out layer)
        space_score = self.space_head(x)
        time_score = self.time_head(x)

        # add enerditfinal
        # print("score shape ", score.shape)

        # this is for all patches
        # y = x[:,:,:,-1].view()

        # print(f"what is out of block shape", score.shape) #(b, context_len, in_dim)
        # print(f"what is the query going into energy layer ", x_for_dyt.shape)

        energy = self.final_enerdit_layer(space_score, x_for_dyt)

        # TODO@DR - reason through context next toward loss

        # so this will now output the energy for each token with last
        # one being for query; RIght now train code is setup to
        # predict the clean image so do that for a first train run

        # print("shapes of score and x out of enerdit now ", score.shape, x.shape)
        # it wants context last
        return energy, space_score, time_score
        # return torch.permute(score, (0, 2,1)), x #shape of x is (b, context, d_model)


# AdHoc testing
# X_train = torch.randn(4, 100, 3, 5, requires_grad=True)  # (B, ,C, context_len)
# model = EnerdiT()
# energy = model(X_train)
# energy.retain_grad()


# dummy_loss = energy.sum()
# dummy_loss.backward()
# print(energy.grad)

# TODO@DR: Reframe it as energy match and not score match (especially if I use
# a time head and a space head)
