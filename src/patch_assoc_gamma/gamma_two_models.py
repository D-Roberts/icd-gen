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
        # print("se[:, 0::2] and pe", se[:, 0::2].shape, se)
        se[:, 1::2] = torch.cos(spatial * div_term)  # start index at 1 with step size 2
        self.register_buffer("se", se.unsqueeze(0))  # Add a batch dimension

    def forward(self, x):
        # x is the sequence of patch encodedings, shape (batch_size, seq_len, d_model)
        # We need to add space encodedings to each element in the sequence

        return self.se[:, : x.size(1)]


class TransformerModelV2(nn.Module):
    """
    Simplified attention only 1 layer and softmax;

    If choose to use a frozen pretrained transformer kernel,
    gpt2 encodedings for instance have size 768 so
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
                # vit comes with its own position encodedings
                self._backbone = vitm.encoder
                for param in self._backbone.parameters():
                    param.requires_grad = False

        # these
        self.W_KQ = weight_matrix(d_model, d_model, mode="default")
        self.W_PV = weight_matrix(d_model, d_model, mode="default")
        self.rho = 1.0

        self.encodepatch = Patchencodeding(dim_input, d_model)
        self.encodepos = Spaceencodeding(d_model, context_length)
        self.unencode = torch.nn.Linear(d_model, dim_input, bias=True)

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
        patch_encode = self.encodepatch(permuted)
        # print(f"patch encode shape********* {patch_encode.shape}")

        pos_encode = self.encodepos(
            patch_encode
        )  # [1, 10, 32] will add same order each batch
        # print(f"pos encode shape********* {pos_encode.shape}")

        encodeded = patch_encode + pos_encode
        # print(
        #     f"after encodedings shape ........{encodeded.shape}"
        # )  # they have (20, 10, 32)

        # Choose to add a frozen pretrained backbone, as a kernel projector

        if self.add_frozen_kernel:
            if self.bb == "GPT2":
                encodeded = self._backbone(inputs_encodes=encodeded).last_hidden_state

            if self.bb == "ViT":
                encodeded = self._backbone(encodeded).last_hidden_state

        encodeded = torch.permute(encodeded, (0, 2, 1))
        # the rest of this expects shape unpermuted
        W_KQ = self.W_KQ  # dmod, dmod
        W_PV = self.W_PV
        # print(f"recall shape of W_KQ {W_KQ.shape}")

        # patch seq == context len should be last dim
        # xs_skip_last = xs[:, :, :-1]
        xs_skip_last = encodeded[:, :, :-1]

        # because last is the query
        # print(f"xs_skip_last {xs_skip_last.shape}") # (20, 9, 32)

        # now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ encodeded / self.rho
        # print(f"recall shape of softm _arg {attn_arg.shape}")  # B, D-1, D
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = W_PV @ xs_skip_last @ softmax_attn_arg
        # print(f"recall shape of f_attn {f_attn.shape}")  # B, d_mod, D

        # print(f"shape of f_attn {f_attn.shape}")  # (batch, d_model, seqlen)
        # ([20, 32, 10]) including the query

        # the target comes in 200dim
        # so unencode here to 200 dim though I could probably keep only the 100dim
        out_full = self.unencode(torch.transpose(f_attn, 2, 1))

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


# as in Def3.2Simplification3.1
# as in Def3.2Simplification3.1
class Attention321(nn.Module):
    def __init__(self, Q, v):
        super(Attention321, self).__init__()
        self.Q = Q
        self.v = v
        d = v.shape[-1]
        D = v.shape[-2]
        self.sm = nn.Softmax(dim=-1)
        self.out_project = nn.Linear(D, d, bias=False)

    def forward(self, X):
        # print(f"X shape {X.shape}")  # (B, D, d)
        Q = self.Q  # [10, 10]
        # print(f"Q shape {Q.shape}")
        v = self.v  # d=100 shape
        attn = self.sm(Q)  # DxD
        # print(f"attn shape {attn.shape}")  # DxD ok
        # v_X = X @ v # (B, D)
        # print(f"v_X.shape {v_X.shape}")
        # out = v_X.mm(attn.T)

        # print(f"X {X.shape} and {v.T.shape} and {attn.T.shape}")
        permuteX = torch.permute(X, (0, 2, 1))
        alternative = permuteX @ v.T @ attn.T  # is B, D, d @ d, D @ D. D
        # print(f"alternative shape {alternative.shape}")
        # print(f"out.shape {out.shape}") # B, D # like a regression task which is what this is without the final softmax

        # but because in denoise task output must be same shape as input
        # project one more time to get B,

        # might freeze it
        out = self.out_project(alternative)
        return out


class SpatialTransformer(nn.Module):
    def __init__(self, alpha, p, sigma_Q, D, d):
        super(SpatialTransformer, self).__init__()

        Q_0 = torch.eye(D) * sigma_Q + torch.randn(D, D) * 0.001
        v_0 = torch.randn(D, d) * 0.001

        Q = torch.nn.Parameter(Q_0)
        v = torch.nn.Parameter(v_0)  # shape of w which is of dim d

        self.Q = Q
        self.v = v
        self.attention = Attention321(Q, v)
        self.sigma = SigmaN(alpha, p)

    def forward(self, X):
        # print(f"X shape in net {X.shape}, w shape {w.shape}, v shape {self.v.shape}")
        H = self.attention(X)
        # print(f"shape of H {H.shape}")
        out = self.sigma(H)
        return torch.transpose(out, 2, 1), H
