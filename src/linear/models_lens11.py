import numpy as np

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm

import math
from transformers import GPT2Model, ViTModel
import warnings

warnings.filterwarnings("ignore")


def weight_matrix(dim_in, dim_out, mode="default"):
    """
    Can use to initialize weight matrices in nn layers
        e.g. self.W_v = weight_matrix(h=ndim, w=ndim, mode="default")

    Throughout, we multiply on the right (e.g. y = W @ x) for consistency with the math notation.
        Thus, dim_in is the number of columns, and dim_out is the number of rows. (i.e. w, h in PyTorch notation)

    For info on default init from torch method nn.Linear, see here:
      https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    W_tensor = torch.empty(dim_out, dim_in)
    if mode == "default":
        low = -1.0 / np.sqrt(dim_in)
        high = 1.0 / np.sqrt(dim_in)
        torch.nn.init.uniform_(W_tensor, a=low, b=high)
    elif mode == "kaiming":
        torch.nn.init.kaiming_uniform_(W_tensor)
    elif mode == "normal":
        torch.nn.init.normal_(W_tensor, mean=0, std=0.02)
    else:
        raise ValueError("Unsupported `mode`")
    return torch.nn.Parameter(W_tensor)


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


# this is the more standard sin pos embed; diff a bit from dit
# I decided to call them Space Embeddings. Seem to me more apt
# Since Pos was from language really
class SpaceEmbedding(nn.Module):
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
        # x is the sequence of patch embeddings, shape (batch_size, seq_len, d_model)
        # We need to add space embeddings to each element in the sequence

        return self.se[:, : x.size(1)]


class TransformerModelEmb(nn.Module):
    """
    Simplified attention only 1 layer and softmax;

    If choose to use a frozen pretrained transformer kernel,
    gpt2 embeddings for instance have size 768 so
    d_model must be 768.
    """

    def __init__(
        self,
        context_length,
        dim_input,
        d_model=32,
        add_frozen_kernel=False,
        backbone="ViT",
        n_layer=1,
        n_head=1,
    ):
        super().__init__()

        self.add_frozen_kernel = add_frozen_kernel

        if self.add_frozen_kernel:
            d_model = 768
            if backbone == "GPT2":
                self.bb = "GPT2"
                self._backbone = GPT2Model.from_pretrained("gpt2")
                for param in self._backbone.base_model.parameters():
                    param.requires_grad = False
            elif backbone == "ViT":
                self.bb = "ViT"
                vitm = ViTModel.from_pretrained("google/vit-base-patch16-224")
                # vit comes with its own position embeddings
                self._backbone = vitm.encoder
                for param in self._backbone.parameters():
                    param.requires_grad = False

        # these
        self.W_KQ = weight_matrix(d_model, d_model, mode="default")
        self.W_PV = weight_matrix(d_model, d_model, mode="default")
        self.rho = 1.0

        self.embedpatch = PatchEmbedding(d_model, dim_input)
        self.embedpos = SpaceEmbedding(d_model, context_length)
        self.unembed = torch.nn.Linear(d_model, dim_input, bias=False)

    def forward(self, xs):
        """
        xs is a sequence of fused noisy and clean patches representing a prompt.
            the query will be the last noisy patch and its label will be
            the clean one.

        return: DR: last layer full output and the argument to the softmax
                    to be able to analyze representations later.
        """
        # print(xs.shape) #[80, 200, 10] fused patch dim is 200
        batchsz, n_dim, n_tokens = xs.size()

        permuted = torch.permute(xs, (0, 2, 1))
        patch_embed = self.embedpatch(permuted)
        # print(f"embed shape********* {patch_embed.shape}")

        pos_embed = self.embedpos(
            patch_embed
        )  # [1, 10, 32] will add same order each batch
        # print(f"pos embed shape********* {pos_embed.shape}")

        embedded = patch_embed + pos_embed
        # print(f"after embeddings shape ........{embedded.shape}") # they have (20, 10, 32)

        # Choose to add a frozen pretrained backbone, as a kernel projector

        if self.add_frozen_kernel:
            if self.bb == "GPT2":
                embedded = self._backbone(inputs_embeds=embedded).last_hidden_state

            if self.bb == "ViT":
                embedded = self._backbone(embedded).last_hidden_state

        embedded = torch.permute(embedded, (0, 2, 1))
        # the rest of this expects shape unpermuted
        W_KQ = self.W_KQ
        W_PV = self.W_PV

        # xs_skip_last = xs[:, :, :-1]
        xs_skip_last = embedded[:, :, :-1]

        # because last is the query
        # print(f"xs_skip_last {xs_skip_last.shape}") # (20, 9, 32)

        # now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ embedded / self.rho
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = W_PV @ xs_skip_last @ softmax_attn_arg

        # print(f"shape of f_attn {f_attn.shape}")  # (batch, d_model, seqlen)
        # ([20, 32, 10]) including the query

        # the target comes in 200dim
        # so unembed here to 200 dim though I could probably keep only the 100dim
        out_full = self.unembed(torch.transpose(f_attn, 2, 1))

        # print(f"shape unemmbedded output {out_full.shape}")

        return torch.transpose(out_full, 2, 1), attn_arg


class TransformerModel2H(nn.Module):
    """
    somftamx simplified 1 layer only 2 heads.
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=2):
        super().__init__()

        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")

        self.rho = 1.0
        self.num_heads = 2
        self.head_dim = dim_input // 2  # this is dk
        # diminput is like d_model or embed dim in classic mha

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, seq_len = xs.size()
        xs_skip_last = xs[:, :, :-1]

        # . for full context; keep it around for analysis
        xs_reshaped = xs.view(batchsz, self.num_heads, self.head_dim, seq_len)
        # xs_last_reshaped = xs[:, :, [-1]].view(
        #     batchsz, self.num_heads, self.head_dim, 1
        # )  # only the last token in seq
        # print("view shape of xs_last_reshaped", xs_last_reshaped.shape)

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        # print(f"shape of W_KQ {W_KQ.shape}") #8x8
        # so dim_in x dim_in or dmodelxdmodel

        xskq = torch.transpose(xs_skip_last, 1, 2) @ W_KQ  # transp to put dimodel last
        # print("view shape of xskq", xskq.shape) # B, seq-1, dim

        xskq_split = self.split_heads(xskq)
        # print("view shape of xskq_split", xskq_split.shape) #B, num_heads, seq_len-1, dimhead

        # print("xs_reshaped.transp ", torch.permute(xs_reshaped, (0, 1, 3, 2)).shape) #don't seem to need reshape here
        # print("xs_reshaped non transp ", xs_reshaped.shape)

        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = (
            torch.matmul(xskq_split, xs_reshaped) / self.rho
        )  # ([B, 2, seq_len-1, seq_len]) # I am reshaping xs directly; could be xs_reshaped_last
        # print("shape of attn_arg ", attn_arg.shape)

        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        # print("shape of softmax_attn_arg ", softmax_attn_arg.shape)
        # B, numheads, seq-1, seq

        wpvx = W_PV @ xs_skip_last

        # print("shape of this wpvx ", wpvx.shape) #B, dim, seq-1
        split_wpvq = wpvx.view(
            batchsz, self.num_heads, self.head_dim, seq_len - 1
        )  # when skipped last

        f_attn = split_wpvq @ softmax_attn_arg  # no residual connection here
        # print("shape of f_attn before contigu reshape", f_attn.shape)
        # B, num_head, dim_head, seqlen

        f_attn_comb = f_attn.contiguous().view(
            batchsz, n_dim, seq_len  # could be 1 instead of seq_len if only last
        )
        # print("shape of f_attn after contigu reshape", f_attn_comb.shape)
        # B, dim, seq_len
        return f_attn_comb[:, :, -1]  # only for last token so (B, dim)


# as in icd
class TransformerModelV1(nn.Module):
    """
    Simplest model:
    - no positional encoding is used
    - `linear self-attention` (no softmax wrapper used)

    Notes
     - dim_input - the dimension of input tokens
     - dim_attn  - the dimension of the residual stream (attention head + MLP input and output)
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = context_length  # scaling used in Bartlett 2023

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        rho = n_tokens

        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / rho
        f_attn = xs + W_PV @ xs @ attn_arg

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


# as in icd
class TransformerModelV1noresOmitLast(TransformerModelV1):
    """
    See docstring TransformerModelV1 : TODO@DR these are just going to be two simple baselines -
                                             keep only one copy of V1 and V2
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(
            context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head
        )

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()
        # print(f' xs.size() { xs.size()}')

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        rho = n_tokens - 1
        # print(f' W_KQ { W_KQ.shape}')
        # print(f' W_PV { W_PV.shape}')

        xs_skip_last = xs[:, :, :-1]
        # print(f' xs_skip_last { xs_skip_last.shape}')

        projection_estimate = xs_skip_last @ torch.transpose(xs_skip_last, 1, 2) / rho
        # print(f' projection_estimate { projection_estimate.shape}')

        f_attn_approx = W_PV @ projection_estimate @ W_KQ @ xs[:, :, [-1]]
        # out = f_attn_approx[
        #     :, :, -1
        # ]  # take dim_n output result at last token, for all batches

        # all
        return f_attn_approx, projection_estimate


class TransformerModel2L(nn.Module):
    """
    [DR] 2-layer Simplest model - mimicking the theory section:
    - no positional encoding is used
    - with softmax otw it would just be a linear layer with the matmul
    and addition of the extra weight matrices

    - tie weights as in theory

    - no res connection or normalizations

    - dim_input - the dimension of input tokens

    TODO@DR: this is simplified and does not include EL as in theory.

    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=2, n_head=1):
        super().__init__()

        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = 1.0

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]

        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        # layer 1
        xs_skip = xs[:, :, :-1]
        # s = xs[:, :, [-1]]

        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        # attn_arg = (
        #     torch.transpose(xs, 1, 2) @ W_KQ @ xs / self.rho
        # )  # this should be now (499, 500)

        attn_arg = torch.transpose(xs_skip, 1, 2) @ W_KQ @ xs / self.rho
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)  # 499, 500

        f_attn = W_PV @ xs_skip @ softmax_attn_arg  # now this should be (16, 500)

        # print("what was xs shape and what is f_attn shape if use xs_skip transp and xs", f_attn.shape)
        # but for two layers I cannot use s only because what would f_attn_skip be for next layer.
        # # # add another layer- no residual connection; no linear in between; use the same matrices

        # # now skip for layer 2
        f_attn_skip = f_attn[:, :, :-1]

        attn_arg1 = (
            torch.transpose(f_attn_skip, 1, 2) @ W_KQ @ f_attn / self.rho
        )  # keep full f_attn second to get last token
        softmax_attn_arg1 = torch.softmax(attn_arg1, dim=1)

        f_attn1 = W_PV @ f_attn_skip @ softmax_attn_arg1
        # print("what is the shape of skip version
        # f_attn1 (but full second f_attn)", f_attn1.shape)
        # #[80, 16, 500] - carve out last token

        # out = f_attn1[:, :, -1]
        return f_attn1, f_attn


MODEL_CLASS_FROM_STR = {
    "TransformerModel2L": {"class": TransformerModel2L, "alias": "T2L"},
    "TransformerModelEmb": {"class": TransformerModelEmb, "alias": "TV2"},
    "TransformerModel2H": {"class": TransformerModel2H, "alias": "T2H"},
    "TransformerModelV1": {"class": TransformerModelV1noresOmitLast, "alias": "TV1"},
}
# define companion dict mapping alias to class string
MODEL_CLASS_ALIAS_TO_STR = {v["alias"]: k for k, v in MODEL_CLASS_FROM_STR.items()}
