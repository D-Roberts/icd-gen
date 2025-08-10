import numpy as np

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm

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


class TransformerModelV2(nn.Module):
    """
    Simplified attention only 1 layer and softmax;

    Misnomer - this is not a transformer.
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()

        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = 1.0
        self.project_out = nn.Linear(dim_input, dim_input // 2, bias=False)

    def forward(self, xs):
        """
        xs is a sequence of fused noisy and clean patches representing a prompt.
            the query will be the last noisy patch and its label will be
            the clean one.

        return: DR: last layer full output and the argument to the softmax
                    to be able to analyze representations later.
        """
        print(xs.shape)
        xs = torch.permute(xs, (0, 2, 1))
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        xs_skip_last = xs[:, :, :-1]

        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ xs / self.rho
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = W_PV @ xs_skip_last @ softmax_attn_arg

        # out = f_attn[
        #     :, :, -1
        # ]  # take dim_n output result at last token, for all batches

        # @DR: return all to be able to plot and inspect ranks and other
        # properties of representations

        print(f"out from model ***********{f_attn.shape}")
        f_attn_reperm = torch.permute(f_attn, (0, 2, 1))
        return self.project_out(f_attn_reperm), attn_arg


MODEL_CLASS_FROM_STR = {
    "TransformerModelV2": {"class": TransformerModelV2, "alias": "TV2"},
}
# define companion dict mapping alias to class string
MODEL_CLASS_ALIAS_TO_STR = {v["alias"]: k for k, v in MODEL_CLASS_FROM_STR.items()}
