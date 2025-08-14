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


# TODO@DR - follow this example for alternative to the Pos Embed and inspect chagnes
"""
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
"""


# TODO@DR: If I put a SiT in - recall that I had an issue with their Sin Pos Embed-
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


class TransformerModelV2(nn.Module):
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
        self.embedpos = SinusoidalPositionalEmbedding(d_model, context_length)
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

        # embed
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


MODEL_CLASS_FROM_STR = {
    "TransformerModelV2": {"class": TransformerModelV2, "alias": "TV2"},
}
# define companion dict mapping alias to class string
MODEL_CLASS_ALIAS_TO_STR = {v["alias"]: k for k, v in MODEL_CLASS_FROM_STR.items()}
