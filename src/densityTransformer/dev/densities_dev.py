"""
Archi step by step debug process.
"""

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Patchencode, Attention, Mlp
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
        self.bias = nn.Parameter(torch.zeros(shape_in))
        self.shape_in = shape_in

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        # print("In tanh x shape", x.shape)
        # print(f"trouble weight in tanh {self.shape_in}")
        # elementwise
        x = x * self.weight + self.bias
        return x


# encode - Not used in the simple datagen
# now the sequence has double width due to concat clean and noisy
class Patchencodeding(nn.Module):
    def __init__(self, dim_in, d_model):
        super().__init__()

        # aim to encode the clean and noisy together for dim reduction
        # alternatively I could add the space encode directly in the
        # 200 dim

        # set bias to zero
        self.projection = torch.nn.Linear(dim_in, d_model, bias=False)

    def forward(self, x):
        # print(f"layer in patch encode {self.projection}")
        # print(f"debug patch encode shapes {x.shape}")
        x = self.projection(x)
        # as expected (B, seq_len, d_model=encodedim)
        return x


# this is the more standard sin pos encode; diff a bit from dit
# I decided to call them Space encodedings. Seem to me more apt
# Since Pos was from language really
class Spaceencodeding(nn.Module):
    def __init__(self, d_model, max_len=512):
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
        # x is the sequence of patch encodedings, shape (batch_size, seq_len, d_model)
        # We need to add space encodedings to each element in the sequence

        return self.se[:, : x.size(1)]


class Contextencodeding(nn.Module):
    def __init__(self, d_model, vocab_size=2):
        """
        encode if 0=clean or 1=noisy akin to question answering bert style encodedings
        for in context tokens

        aim to have noisy first clean second
        will be added to the fused noisy clean context to tell the model what's what
        """
        super().__init__()
        self.encodeding_layer = nn.encodeding(vocab_size, d_model)

    def forward(self, x):
        return self.encodeding_layer(x)


# inspired from https://github.com/facebookresearch/DiT/blob/main/models.py#L26
class Timeencodeding(nn.Module):
    """We just have to have them.

    logt will be drawn from a U(logtmin, logtmax), will tmin = 10**-9 and tmax = 10**3

    each token in sequence will have an associated t. encode to same d_model and add to the
    pathc and space encodedings.

    encodedings are sin cos

    as of right now I have same t accross the sequence
    I encode it with this
    I tile it for seq len - identical for each sequence token
    I concat on the encodeding dimension to the x input in DensityTranformer block

    The t will be distinct for each instance in the batch
    """

    def __init__(
        self,
        d_model,
        frequency_encodeding_size=256,
        mint=0,
        maxt=10000,
    ):
        super().__init__()
        self.time_encodeder = nn.Sequential(
            nn.Linear(frequency_encodeding_size, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.frequency_encodeding_size = frequency_encodeding_size
        self.mint = mint
        self.maxt = maxt

    @staticmethod
    def time_encodeding(t, dim, mint, maxt):
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
        encodeding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            encodeding = torch.cat(
                [encodeding, torch.zeros_like(encodeding[:, :1])], dim=-1
            )
        return encodeding  # this will be shape (len of t, 256)

    def forward(self, t):
        t_freq = self.time_encodeding(
            t, self.frequency_encodeding_size, self.mint, self.maxt
        )
        time_encode = self.time_encodeder(t_freq)
        return time_encode


"""
DensityTranformer archi and buildinblocks.

Running low on time.

"""


class DensityTranformerFinal(nn.Module):
    def __init__(
        self,
    ):
        super(DensityTranformerFinal, self).__init__()

    def forward(self, sh, th, noisy):
        """

        U = 0.5( <s, y> - correction) TODO@DR reconsider math for this
        correction

        correction = cf1 * time score

        cf1 hyperpar but consider learning it later; though no direct supervision
        as of now available since U layer doesn't feed through the optimization
        feedback loop

        set it to cf1 = 0.1

        return : energy for query token
        """
        # only use the non-zero portion of the noisy query

        # print(
        #     f"in energy calculation noisy shape is with simple and timm patchify {noisy.shape}"
        # )
        # (20, 1024)- not patchified
        # noisy = torch.permute(noisy, (0, 2, 1))

        # print(f"in energy calculation space score shape with simple {sh.shape}")
        # after unpatchify (b, d)
        # in simple now not the case anymore
        # sh = torch.permute(sh, (0, 2, 1))

        # in simple, only batch and full image dim
        b, d = noisy.shape
        # d = d // 2

        # query = noisy[:, :d, -1]
        # query = noisy  # in simple this is the query

        # sp_pred = sh[:, :, -1]  # the Space Final should already have d // 2
        # sp_pred = sh  # this needs to come unpatchified with the time encode
        # and whatever other context removed
        # just add an average pooling after time encoded removed

        # print(f"sp_pred in energy {sp_pred.shape}")
        # print(f"th shape in energy {th.shape}")  # (B, seqlen) with context otw (b,)
        # should be scalar according to theory
        # th_pred = th[:, -1]

        # th_pred = th

        # In simple, time score also must come unpatchified and one pred only
        # since no context - so detach time encoded token and average pool
        # IN Time Final layer

        # print(f"th_pred shape in energy {th_pred.shape}") #(B,)

        # This is not right because th_pred is a scalar
        # sc = sh - th * cf1

        # TODO@DR: I am adding the correction after inner prod
        # so if this works empirically need to rework theory math
        # for this approximation

        cf1 = 0.1
        correction = cf1 * th.squeeze()  # comes in as (B, 1)

        # print(f"in energy sp_pred {sp_pred}")
        # print(f"in energy query {query}")
        # print(f"in energy th_pred {th_pred}")

        # TODO@DR: will have to see about signs; query is the noisy
        sp_based = 0.5 * (torch.sum(sh * noisy, dim=(-1)))
        energy = sp_based - correction

        # print(f"\nin energy energy {energy} and shape {energy.shape} ") # B
        # print(f"\nsh in ener {sh} and shape {sh.shape}") # (B, d)
        # print(f"\nsp_based in ener {sp_based} and shape {sp_based.shape}") # B,
        # # some really large nums in sp_based
        # print(f"\nsqueezed th_based in ener {th.squeeze()} and shape {th.squeeze().shape}") # B
        # # small numbers in th
        # print(f"\nquery in ener {noisy} and shape {noisy.shape}") # B, d

        # correction is not being learned since U is not fed
        # through the learning loop directly
        # leave it as a hyper par right now

        # TODO@DR: consider a normalization term

        return energy


class PreHead(nn.Module):
    """
    this is before the two heads; not used right now.
    """

    def __init__(self, context_len, d_model):
        super().__init__()

        # context + 1 because of the concatenated time encode
        self.dyt_final = DyTanh((context_len + 2, d_model))
        self.silu = nn.SiLU()

        # TODO@DR: not sure yet about the modulation.
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

    x comes in (B, d) after unpatch in simple task see more detail in
    Space Final
    """

    def __init__(self, d_model, input_dim, context_len):
        super().__init__()

        self.dyt_time = DyTanh((context_len * d_model))
        self.silu = nn.SiLU()
        self.time_final = nn.Linear(
            context_len * d_model, 1, bias=True
        )  # make it one directly -as is the supervision and partial U of t

    def forward(self, x):
        # print("shape of x as it comes in Time Final ", x.shape)
        x = self.dyt_time(x)
        # print("shape of x as it comes out of dyt in final layer ", x.shape)
        x = self.silu(x)
        x = self.time_final(x)

        # print("shape of x as it comes out of Time Final ", x.shape)
        return x


class SpaceHead(nn.Module):
    """
    with a silu and a dyt as in pre-norm but no resid connections

    For simple task - need to detach time encode and unpatchify

    for simple img level no context task x here is already (B, d)
    """

    def __init__(self, d_model, input_dim, context_len):
        super().__init__()

        self.dyt_space = DyTanh(
            shape_in=(context_len * d_model)
        )  # bc it was unpatchified
        self.silu = nn.SiLU()
        self.space_final = nn.Linear(
            context_len * d_model, input_dim, bias=True
        )  # project to d dim

    def forward(self, x):
        # print(f"shape in head {x.shape}")  # [3, 8, 4] is (B, seq_len, d_model)
        # in simple task: torch.Size([20, 257, 32])
        # detach time encode before
        x = self.dyt_space(x)
        x = self.silu(x)
        x = self.space_final(x)
        return x


class DensityTranformerBlock(nn.Module):
    """
    no layernorm or adaptvie ones for now.
    dyt
    """

    def __init__(self, d_model, context_len, num_heads=1, mlp_ratio=4.0):  # as in DiT
        super().__init__()

        # context_len+2 bc I conditioned with the time encodes - 2kinds now
        # and the two context tokens (noisy and clean)
        self.dyt1 = DyTanh((context_len + 4, d_model))

        self.attn = Attention(
            d_model, num_heads, qkv_bias=True
        )  # identity and not norms on internal qkv

        self.dyt2 = DyTanh((context_len + 4, d_model))

        mlp_hidden_dim = int(0.4 * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, time_encode, time_encode1, encodeded_context):
        # print(f"shape of time emebde w simple {time_encode.shape}")
        # print(f"shape of x near time emebde w simple {x.shape}")

        # Condition on the time encodeding in each block (only 1 block used at this time)
        # print(f"what is concat {encodeded_context.shape}")
        x = torch.cat([x, time_encode, time_encode1, encodeded_context], dim=1)
        x = self.dyt1(x)
        x = x + self.attn(x)
        x = self.dyt2(x)
        x = x + self.mlp(x)
        return x


# Taking inspiration from https://github.com/facebookresearch/DiT/blob/main/models.py
class DensityTranformer(nn.Module):
    """
    assume at this point that the patches are cut up and fused already if for in-context.
    in simple - timm patching used.
    """

    def __init__(
        self,
        d_model,
        input_dim,  # the way datagen is setup now - comes in one flattened
        cf1_init_value=0.1,
        num_heads=1,
        depth=1,
        mlp_ratio=4,
        patch_dim=2,
        context_len=None,
    ):
        super(DensityTranformer, self).__init__()

        # 1 is for input channesl which is grayscale 1; patch_dim is one dim
        # so area is p*p; input_dim is likewise h
        self.h = int(math.sqrt(input_dim))  # for us image comes flattened
        self.patch_encode = Patchencode(self.h, patch_dim, 1, d_model, bias=True)

        # IN simple synthetic image level case, context_len is num of patches
        if context_len is None:
            self.context_len = input_dim // (patch_dim * patch_dim)  # 1024 // 4=256

        else:
            self.context_len = context_len

        self.space_encode = Spaceencodeding(d_model, self.context_len)
        self.time_encode = Timeencodeding(
            d_model,
            frequency_encodeding_size=256,
            mint=0,
            maxt=10000,
        )
        self.time_encode1 = Timeencodeding(
            d_model,
            frequency_encodeding_size=256,
            mint=0,
            maxt=10000,
        )

        # on concat noisy first clean after
        self.patch_encode_context = Patchencodeding(input_dim, d_model)
        self.context_seq_encode = Contextencodeding(d_model=d_model, vocab_size=2)

        ######################################Before this - inputs encodeding
        # context len now is number of patches from Patch making ViT style

        # then comes the list of N DensityTranformer blocks
        self.blocks = nn.ModuleList(
            [
                DensityTranformerBlock(d_model, self.context_len, num_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )

        # # correction factor space time scores

        self.final_DensityTranformer_layer = DensityTranformerFinal()

        # self.prehead_linear = PreHead(self.context_len, d_model,)

        # print(f"is context len the right thing {context_len}")

        # a kind of unencode; aim to use only on the non-zero part of
        # the noisy query and clean label

        # in simple - at full image with no context and unfused
        # model predicts the full image but in patchified form
        # context_len here is just the number of patches ViT style
        # so I think input dim should be patch dim here
        # since I aimed for patch dim of 2 should have patch size of 4

        self.time_final = TimeHead(d_model, input_dim, self.context_len)
        self.space_final = SpaceHead(d_model, input_dim, self.context_len)

        # self.pre_init() #TODO@DR see later about custom init - right now is hurting

    # def pre_init(self):
    #     """will init weights here however way I want and whatever else I
    #     want to init

    #     for now the linears and the final but not final init
    #     """

    #     # TODO@DR: this is in spirit of DiT but I am not convinced I'll stay
    #     # with this
    #     def lin_init(module):
    #         if isinstance(module, nn.Linear):
    #             # TODO@DR: reconsider later: not learning well with this below at this time in dev
    #             nn.init.xavier_uniform_(module.weight)
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0)

    #     self.apply(lin_init)

    #     # zero out out layer on the heads TODO@DR: think this again

    #     nn.init.constant_(self.space_final.space_final.bias, 0)
    #     nn.init.constant_(self.time_final.time_final.bias, 0)

    #     nn.init.constant_(self.space_final.space_final.weight, 0)
    #     nn.init.constant_(self.time_final.time_final.weight, 0)

    def unpatchify(self, x):
        """
        x: (B, num_patches, patch_dim)

        imgs (B, D) in simple (B, 1024)
        """
        b, num_p, pdim = x.shape
        x = x.reshape(shape=(b, num_p * pdim))
        return x

    def forward(self, x, noisy_context, clean_context, t, device):
        """x is the noisy query"""
        # print("what shape comes the batch into DensityTranformer ", x.shape)
        # b_s, in_d, context_len = x.shape
        b_s, in_d = x.shape  # for simple only b and d dim

        x_for_U = x

        assert t.shape[0] == x.shape[0]

        # For simple - need to patchify; it comes in (b, d)

        C = 1  # grayscale
        x = x.view(
            (b_s, C, self.h, self.h)
        )  # timm patch layer wants image in hxw format
        patch_encode = self.patch_encode(x)

        # print(f"patch_encode now after timm in simple {patch_encode.shape}")

        space_encode = self.space_encode(
            patch_encode
        )  # [1, seq_len, d_model] will add same order each instance in batch

        # print(f"shape of space encodeding {space_encode.shape}")

        time_encode = self.time_encode(t)
        # print(f"shape of time encodeding {time_encode.shape}")

        x = patch_encode + space_encode
        # for Simple need to squeeze out the 1 dim
        x = x.squeeze()

        # Tile identical t encodedings for each context token.
        # Right now they get concat to the input of each DensityTranformer block

        time_encode = time_encode.unsqueeze(1)
        # time_encode = torch.tile(time_encode, (1, 256, 1)) # on simple
        # concatenate 1 to the sequence; porbably should do this always
        # and not to each patch token

        # Add another time encodeding for sqrt(t/d) scale
        new_t = torch.sqrt(t) / self.h
        time_encode1 = self.time_encode1(new_t)
        time_encode1 = time_encode1.unsqueeze(1)

        ###############Put context in

        context = torch.stack((noisy_context, clean_context), dim=1)
        # print(f"shape of context before encode {context.shape}")

        encode_context_patch = self.patch_encode_context(context)
        # print(f"shape of encode_context_patch  {encode_context_patch.shape}")

        # have a context of only one noisy and one clean
        indices = torch.LongTensor([0, 1])
        encode_context_seq = self.context_seq_encode(indices.to(device))
        # print(f"encode_context_seq {encode_context_seq}")
        # print(f"shape of encode_context_seq  {encode_context_seq.shape}")
        encodeded_context = encode_context_patch + encode_context_seq
        # print(f"shape of encode_context_seq  {encodeded_context.shape}")

        ##################Input encodeding area over

        # add DensityTranformer blocks
        for block in self.blocks:
            x = block(x, time_encode, time_encode1, encodeded_context)

        # I need to detach timeencode now before heads
        # But I need the prehead layer to reshape first

        # print(f"check out x after block {x.shape}") #[5, 257, 32]) (b, context_len+1, d_model) or 258 of 2 kinds
        # x = self.prehead_linear(x) #I'll take this out
        # print(f"check out x after prehead layer {x.shape}")#[5, 257, 32])

        x = x[:, :-4, :]  # this is where I detach the time encode both kinds
        # print(f"check out x after time encode detached {x.shape}") #[5, 256, 32]) ok
        # take all conditionining off inlcuding context

        # And unpatchify here before heads
        x = self.unpatchify(x)
        # print(f"shape of x before heads {x.shape}") # (b, 256*32)

        space_score = self.space_final(x)
        time_score = self.time_final(x)

        # sh, th, y, cf1 - learn it
        # y is the noised in theory but here is called x, the noised query and context tokens

        # Energy is not fed unless used as regularizer
        energy = self.final_DensityTranformer_layer(space_score, time_score, x_for_U)

        return energy, space_score, time_score


# AdHoc testing
# X_train = torch.randn(4, 100, 3, 5, requires_grad=True)  # (B, ,C, context_len)
# model = DensityTranformer()
# energy = model(X_train)
# energy.retain_grad()


# dummy_loss = energy.sum()
# dummy_loss.backward()
# print(energy.grad)
