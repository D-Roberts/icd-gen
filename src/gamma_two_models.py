import numpy as np

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, ViTModel
from tqdm import tqdm

import math
import warnings

warnings.filterwarnings("ignore")


# from icd ICML'25
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
    def __init__(self, d_model, max_len=512):
        super().__init__()
        se = torch.zeros(max_len, d_model)  # (seq_len, d_model)
        spatial = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # double the d_model size

        se[:, 0::2] = torch.sin(spatial * div_term)  # start index at 0 with step size 2
        # print("se[:, 0::2] and pe", se[:, 0::2].shape, se)
        se[:, 1::2] = torch.cos(spatial * div_term)  # start index at 1 with step size 2
        self.register_buffer("se", se.unsqueeze(0))  # Add a batch dimension

    def forward(self, x):
        # x is the sequence of patch embeddings, shape (batch_size, seq_len, d_model)
        # We need to add space embeddings to each element in the sequence

        return self.se[:, : x.size(1)]


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

        self.embedpatch = PatchEmbedding(dim_input, d_model)
        self.embedpos = SpaceEmbedding(d_model, context_length)
        self.unembed = torch.nn.Linear(d_model, dim_input, bias=True)

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
        print(f"patch embed shape********* {patch_embed.shape}")

        pos_embed = self.embedpos(
            patch_embed
        )  # [1, 10, 32] will add same order each batch
        print(f"pos embed shape********* {pos_embed.shape}")

        embedded = patch_embed + pos_embed
        print(
            f"after embeddings shape ........{embedded.shape}"
        )  # they have (20, 10, 32)

        # Choose to add a frozen pretrained backbone, as a kernel projector

        if self.add_frozen_kernel:
            if self.bb == "GPT2":
                embedded = self._backbone(inputs_embeds=embedded).last_hidden_state

            if self.bb == "ViT":
                embedded = self._backbone(embedded).last_hidden_state

        embedded = torch.permute(embedded, (0, 2, 1))
        # the rest of this expects shape unpermuted
        W_KQ = self.W_KQ  # dmod, dmod
        W_PV = self.W_PV
        print(f"recall shape of W_KQ {W_KQ.shape}")

        # patch seq == context len should be last dim
        # xs_skip_last = xs[:, :, :-1]
        xs_skip_last = embedded[:, :, :-1]

        # because last is the query
        # print(f"xs_skip_last {xs_skip_last.shape}") # (20, 9, 32)

        # now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ embedded / self.rho
        print(f"recall shape of softm _arg {attn_arg.shape}")  # B, D-1, D
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = W_PV @ xs_skip_last @ softmax_attn_arg
        print(f"recall shape of f_attn {f_attn.shape}")  # B, d_mod, D

        # print(f"shape of f_attn {f_attn.shape}")  # (batch, d_model, seqlen)
        # ([20, 32, 10]) including the query

        # the target comes in 200dim
        # so unembed here to 200 dim though I could probably keep only the 100dim
        out_full = self.unembed(torch.transpose(f_attn, 2, 1))

        # print(f"shape unemmbedded output {out_full.shape}")

        return torch.transpose(out_full, 2, 1), attn_arg


#######################Jelassi style (modified for denoise rather than
# classification task) to learn spatial pos

# simplified model ViT Spatial Transformer with special Space Attention


# # for denoise task
class SigmaN(nn.Module):
    def __init__(self, alpha, p):
        super(SigmaN, self).__init__()
        self.alpha = alpha
        self.p = p

    def forward(self, H):
        sig_term1 = H**self.p
        # print(f"sig term 1 shape {sig_term1.shape} and val {sig_term1}")
        sig_term2 = self.alpha * H
        # print(f"sig term 2 shape {sig_term2.shape} and val {sig_term2}")
        # without sum [5, 10] which is (B, D)
        return sig_term1 + sig_term2


class Attention(nn.Module):
    def __init__(self, Q, Wv):
        super(Attention, self).__init__()
        self.Q = Q
        self.Wv = Wv
        self.sm = nn.Softmax(dim=1)

    def forward(self, X):
        print(f"X shape {X.shape}")  # [5, 10, 100] B, D, d
        Q = self.Q  # d d
        print(f"Q shape {Q.shape}")
        Wv = self.Wv  # B, d, D
        attn = self.sm(Q)
        print(f"attn shape in attn layer {attn.shape}")  # d d

        # TODO@DR: The logic of this model in the -incontext context
        # must be changed / verified
        v_X = torch.permute(X, (0, 2, 1)) @ Wv.T
        print(f"torch.permute(X, (0, 2,1)) {torch.permute(X, (0, 2,1)).shape}")
        print(f" Wv.T shape { Wv.T.shape}")
        print(f"v_X shape {v_X.shape} in attn layer")

        out = v_X @ attn.T
        return out


class SpatialTransformer(nn.Module):
    def __init__(self, alpha, p, sigma_Q, D, d):  # d is d_model too here
        super(SpatialTransformer, self).__init__()

        # I think Q is A in paper
        Q_0 = torch.eye(d) * sigma_Q + torch.randn(d, d) * 0.001
        v_0 = torch.randn(d, D) * 0.001

        Q = torch.nn.Parameter(Q_0)
        Wv = torch.nn.Parameter(v_0)

        self.Q = Q
        self.Wv = Wv
        self.attention = Attention(Q, Wv)
        self.sigma = SigmaN(alpha, p)

    def forward(self, X):
        # print(f"X shape in net {X.shape}, w shape {w.shape}, v shape {self.v.shape}")
        H = self.attention(X)
        print(f"H shape in net {H.shape}")
        out = self.sigma(H)
        print(f"out shape in net {out.shape}")
        return out, H
