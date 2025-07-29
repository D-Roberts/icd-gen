import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_io import load_runinfo_from_rundir
from data_tools import data_train_test_split_linear
from settings import DIR_MODELS
from torch_device import device_select


class ReturnLastToken(nn.Module):
    """
    Baseline model -- return final token
    """

    def __init__(self):
        super().__init__()

    def forward(self, xs):
        outs = xs[:, :, -1]  # return the last token
        return outs


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


class TransformerModelV1nores(TransformerModelV1):
    """
    See docstring TransformerModelV1
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

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        rho = n_tokens

        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / rho
        f_attn = (
            W_PV @ xs @ attn_arg
        )  # the residual stream term "+ xs" has been removed

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV1noresForceDiag(nn.Module):
    """
    See docstring TransformerModelV1
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        # self.W_KQ = weight_matrix(dim_input, dim_input, mode='normal')
        # self.W_PV = weight_matrix(dim_input, dim_input, mode='normal')

        self.W_KQ = torch.nn.Parameter(torch.tensor(0.1))
        self.W_PV = torch.nn.Parameter(torch.tensor(0.1))

        # self.W_KQ = torch.nn.Parameter(0.1 * torch.eye(dim_input))
        # self.W_PV = torch.nn.Parameter(0.1 * torch.eye(dim_input))
        self.rho = context_length  # scaling used in Bartlett 2023

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ * torch.eye(
            n_dim
        )  # self.W_KQ is a 1-parameter scalar --> make n x n diag arr
        W_PV = self.W_PV * torch.eye(
            n_dim
        )  # self.W_PV is a 1-parameter scalar --> make n x n diag arr

        rho = n_tokens

        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / rho
        f_attn = (
            W_PV @ xs @ attn_arg
        )  # the residual stream term "+ xs" has been removed

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape=16, channels_last=True, alpha_init_value=0.5):
        super().__init__()
        # self.normalized_shape = normalized_shape #TODO@DR this is greatly simplified
        self.alpha_init_value = alpha_init_value
        # self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        # self.weight = nn.Parameter(torch.ones(normalized_shape))
        # self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        # if self.channels_last:
        #     x = x * self.weight + self.bias
        # else:
        #     x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class TransformerModelV1noresOmitLast(TransformerModelV1):
    """
    See docstring TransformerModelV1
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

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        rho = n_tokens - 1

        xs_skip_last = xs[:, :, :-1]

        projection_estimate = xs_skip_last @ torch.transpose(xs_skip_last, 1, 2) / rho

        f_attn_approx = W_PV @ projection_estimate @ W_KQ @ xs[:, :, [-1]]
        out = f_attn_approx[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelV1noresForceDiagAndOmitLast(nn.Module):
    """
    See docstring TransformerModelV1
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        self.W_KQ = torch.nn.Parameter(torch.tensor(0.1))
        self.W_PV = torch.nn.Parameter(torch.tensor(0.1))
        self.rho = context_length

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ * torch.eye(
            n_dim
        )  # self.W_KQ is a 1-parameter scalar --> make n x n diag arr
        W_PV = self.W_PV * torch.eye(
            n_dim
        )  # self.W_PV is a 1-parameter scalar --> make n x n diag arr

        rho = n_tokens - 1

        xs_skip_last = xs[:, :, :-1]

        # attn_arg = torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ xs_skip_last / rho
        projection_estimate = xs_skip_last @ torch.transpose(xs_skip_last, 1, 2) / rho

        f_attn_approx = W_PV @ projection_estimate @ W_KQ @ xs
        out = f_attn_approx[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelV11(nn.Module):
    """
    Simplest model 2 heads
    - no positional encoding is used
    - same as V1 but now softmax in place of `linear` self-attention
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=2):
        super().__init__()
        assert n_layer == 1  # TODO implement...
        assert n_head == 2  # TODO implement...
        assert (
            dim_attn is None
        )  # TODO implement... for now we take dim_attn == dim_input
        # TODO in multilayer version, add AttnHead class beneath AttnLayer class? forward pass is just loop over nlayer

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")

        self.rho = 1.0
        self.num_heads = 2
        self.head_dim = dim_input // 2

        self.output = nn.Linear(context_length, context_length, bias=None)

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
        batchsz, n_dim, n_tokens = xs.size()

        xs_reshaped = xs.view(batchsz, self.num_heads, self.head_dim, n_tokens)

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        xskq = torch.transpose(xs, 1, 2) @ W_KQ

        # print("view shape of xskq", xskq.shape)

        xskq_split = self.split_heads(xskq)
        # print("view shape of xskq_split", xskq_split.shape)

        # print("xs_reshaped.transp ", torch.permute(xs_reshaped, (0, 1, 3, 2)).shape) #don't seem to need reshape here
        # print("xs_reshaped non transp ", xs_reshaped.shape)

        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = (
            torch.matmul(xskq_split, xs_reshaped) / self.rho
        )  # ([1, 2, 500, 500]) # I am reshaping xs directly
        # print("shape of attn_arg ", attn_arg.shape)

        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        # print("shape of softmax_attn_arg ", softmax_attn_arg.shape)

        wpvx = W_PV @ xs

        # print("shape of this wpvx ", wpvx.shape)
        split_wpvq = wpvx.view(batchsz, self.num_heads, self.head_dim, n_tokens)

        f_attn = split_wpvq @ softmax_attn_arg  # no residual connection here

        f_attn_comb = f_attn.contiguous().view(batchsz, n_dim, n_tokens)
        output = self.output(f_attn_comb)

        out = output[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV11SkipLast(nn.Module):
    """
    Simplest model 2 heads - like 11 but on omit last which learns so much better in V2
    - no positional encoding is used
    - same as V1 but now softmax in place of `linear` self-attention
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=2):
        super().__init__()
        assert n_layer == 1  # TODO implement...
        assert n_head == 2  # TODO implement...
        assert (
            dim_attn is None
        )  # TODO implement... for now we take dim_attn == dim_input
        # TODO in multilayer version, add AttnHead class beneath AttnLayer class? forward pass is just loop over nlayer

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")

        self.rho = 1.0
        self.num_heads = 2
        self.head_dim = dim_input // 2

        self.output = nn.Linear(context_length, context_length, bias=None)

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
        batchsz, n_dim, n_tokens = xs.size()

        xs_skip_last = xs[:, :, :-1]

        xs_reshaped = xs.view(batchsz, self.num_heads, self.head_dim, n_tokens)
        # xs_last_reshaped = xs[:, :, [-1]].view(batchsz, self.num_heads, self.head_dim, 1) #only the last
        print("view shape of xs_last_reshaped", xs_reshaped.shape)

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        xskq = torch.transpose(xs_skip_last, 1, 2) @ W_KQ

        # print("view shape of xskq", xskq.shape)

        xskq_split = self.split_heads(xskq)
        # print("view shape of xskq_split", xskq_split.shape)

        # print("xs_reshaped.transp ", torch.permute(xs_reshaped, (0, 1, 3, 2)).shape) #don't seem to need reshape here
        # print("xs_reshaped non transp ", xs_reshaped.shape)

        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = (
            torch.matmul(xskq_split, xs_reshaped) / self.rho
        )  # ([1, 2, 500, 500]) # I am reshaping xs directly
        # print("shape of attn_arg ", attn_arg.shape)

        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        # print("shape of softmax_attn_arg ", softmax_attn_arg.shape)

        wpvx = W_PV @ xs_skip_last

        # print("shape of this wpvx ", wpvx.shape)
        split_wpvq = wpvx.view(
            batchsz, self.num_heads, self.head_dim, n_tokens - 1
        )  # when skipped last

        f_attn = split_wpvq @ softmax_attn_arg  # no residual connection here
        # print("shape of f_attn", f_attn.shape)

        f_attn_comb = f_attn.contiguous().view(batchsz, n_dim, n_tokens)

        output = self.output(f_attn_comb)

        out = output[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV11SkipLastOneOnly(nn.Module):
    """
    Simplest model 2 heads - like 11 but on omit last which learns so much better in V2
    - no positional encoding is used
    - same as V1 but now softmax in place of `linear` self-attention
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=2):
        super().__init__()
        assert n_layer == 1  # TODO implement...
        assert n_head == 2  # TODO implement...
        assert (
            dim_attn is None
        )  # TODO implement... for now we take dim_attn == dim_input
        # TODO in multilayer version, add AttnHead class beneath AttnLayer class? forward pass is just loop over nlayer

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")

        self.rho = 1.0
        self.num_heads = 2
        self.head_dim = dim_input // 2

        self.output = nn.Linear(1, 1, bias=None)  # when only the last token maintained

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
        batchsz, n_dim, n_tokens = xs.size()

        xs_skip_last = xs[:, :, :-1]

        # xs_reshaped = xs.view(batchsz, self.num_heads, self.head_dim, n_tokens)
        xs_last_reshaped = xs[:, :, [-1]].view(
            batchsz, self.num_heads, self.head_dim, 1
        )  # only the last
        print("view shape of xs_last_reshaped", xs_last_reshaped.shape)

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        xskq = torch.transpose(xs_skip_last, 1, 2) @ W_KQ

        # print("view shape of xskq", xskq.shape)

        xskq_split = self.split_heads(xskq)
        # print("view shape of xskq_split", xskq_split.shape)

        # print("xs_reshaped.transp ", torch.permute(xs_reshaped, (0, 1, 3, 2)).shape) #don't seem to need reshape here
        # print("xs_reshaped non transp ", xs_reshaped.shape)

        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = (
            torch.matmul(xskq_split, xs_last_reshaped) / self.rho
        )  # ([1, 2, 500, 500]) # I am reshaping xs directly
        print("shape of attn_arg ", attn_arg.shape)

        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        print("shape of softmax_attn_arg ", softmax_attn_arg.shape)

        wpvx = W_PV @ xs_skip_last

        # print("shape of this wpvx ", wpvx.shape)
        split_wpvq = wpvx.view(
            batchsz, self.num_heads, self.head_dim, n_tokens - 1
        )  # when skipped last

        f_attn = split_wpvq @ softmax_attn_arg  # no residual connection here
        print("shape of f_attn", f_attn.shape)

        f_attn_comb = f_attn.contiguous().view(
            batchsz, n_dim, 1
        )  # can't be num tokens here

        # output = self.output(f_attn_comb)

        # out = output[:, :, -1]  # take dim_n output result at last token, for all batches
        return f_attn_comb[:, :, -1]


class TransformerModelV2(nn.Module):
    """
    Simplest model:
    - no positional encoding is used
    - same as V1 but now softmax in place of `linear` self-attention
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1  # TODO implement...
        assert n_head == 1  # TODO implement...
        assert (
            dim_attn is None
        )  # TODO implement... for now we take dim_attn == dim_input
        # TODO in multilayer version, add AttnHead class beneath AttnLayer class? forward pass is just loop over nlayer

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = 1.0

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / self.rho
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = xs + W_PV @ xs @ softmax_attn_arg

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV2Tied2(nn.Module):
    """
    Simplest model: with 2 layer

    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=2, n_head=1):
        super().__init__()
        assert n_layer == 2  # TODO implement...
        assert n_head == 1  # TODO implement...
        assert (
            dim_attn is None
        )  # TODO implement... for now we take dim_attn == dim_input
        # TODO in multilayer version, add AttnHead class beneath AttnLayer class? forward pass is just loop over nlayer

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = 1.0

        self.dyt = DynamicTanh(normalized_shape=dim_input)
        self.linear = nn.Linear(context_length, context_length, bias=None)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        # layer 1
        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / self.rho
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)

        f_attn = W_PV @ xs @ softmax_attn_arg
        # f_attn = W_PV @ xs @ softmax_attn_arg

        # try add in between; residual connections only make sense when multiple layers really
        # original transf encoder layer: is multihead; add+norm; feed foreard; add + norm; then stack n of these layers

        # so step by step
        # f_attn = self.dyt(xs + f_attn)
        f_attn = xs + f_attn  # better with no dyt here

        # then add + norm

        # # add another layer- no residual connection; no linear in between; use the same matrices

        attn_arg1 = torch.transpose(f_attn, 1, 2) @ W_KQ @ f_attn / self.rho
        softmax_attn_arg1 = torch.softmax(attn_arg1, dim=1)

        f_attn1 = W_PV @ f_attn @ softmax_attn_arg1

        # add residual to f_attn1 as well
        f_attn1 = self.dyt(f_attn1 + f_attn)

        # output = self.linear(f_attn1)
        # out = output[:, :, -1]  # take dim_n output result at last token, for all batches
        out = f_attn1[:, :, -1]
        return out


class TransformerModelV2Tied2_Skip(nn.Module):
    """
    Simplest model: with 2 layer - tied weights- use query sep

    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=2, n_head=1):
        super().__init__()
        assert n_layer == 2  # TODO implement...
        assert n_head == 1  # TODO implement...
        assert (
            dim_attn is None
        )  # TODO implement... for now we take dim_attn == dim_input
        # TODO in multilayer version, add AttnHead class beneath AttnLayer class? forward pass is just loop over nlayer

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = 1.0

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        # layer 1
        # xs_skip = xs[:, :, :-1]
        # s = xs[:, :, [-1]]              #The skipping doesn't work on more than 1 layer / except at the last layer

        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = (
            torch.transpose(xs, 1, 2) @ W_KQ @ xs / self.rho
        )  # this should be now (499, 500)
        # attn_arg = torch.transpose(xs_skip, 1, 2) @ W_KQ @ s / self.rho
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)  # 499, 500

        f_attn = (
            W_PV @ xs @ softmax_attn_arg
        )  # now this should be (16, 500) #yes this works, recovers diags

        # print("what was xs shape and what is f_attn shape if use xs_skip transp and xs", f_attn.shape)

        # but for two layers I cannot use s only because what would f_attn_skip be for next layer.

        # # # add another layer- no residual connection; no linear in between; use the same matrices

        # # now skip for layer 2
        # f_attn_skip = f_attn[:, :, :-1]

        attn_arg1 = (
            torch.transpose(f_attn, 1, 2) @ W_KQ @ f_attn / self.rho
        )  # keep full f_attn second to get last token
        softmax_attn_arg1 = torch.softmax(attn_arg1, dim=1)

        f_attn1 = W_PV @ f_attn @ softmax_attn_arg1
        # print("what is the shape of skip version f_attn1 (but full second f_attn)", f_attn1.shape) #[80, 16, 500] - carve out last token

        out = f_attn1[:, :, -1]

        return out


class TransformerModelV2L2(nn.Module):
    """
    Simplest model: with 2 layer; not tied weights

    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=2, n_head=1):
        super().__init__()
        assert n_layer == 2  # TODO implement...
        assert n_head == 1  # TODO implement...
        assert (
            dim_attn is None
        )  # TODO implement... for now we take dim_attn == dim_input
        # TODO in multilayer version, add AttnHead class beneath AttnLayer class? forward pass is just loop over nlayer

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.W_KQ1 = weight_matrix(dim_input, dim_input, mode="default")

        self.W_PV1 = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = 1.0

        # self.dyt = DynamicTanh(normalized_shape=dim_input)

        # self.layernorm = nn.LayerNorm(dim_input, eps=1e-12)
        # self.linear = nn.Linear(context_length, context_length, bias=None)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        W_KQ1 = self.W_KQ1
        W_PV1 = self.W_PV1

        # layer 1
        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / self.rho
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)

        f_attn = W_PV @ xs @ softmax_attn_arg
        # f_attn = W_PV @ xs @ softmax_attn_arg

        # try add in between; residual connections only make sense when multiple layers really
        # original transf encoder layer: is multihead; add+norm; feed foreard; add + norm; then stack n of these layers; then either transf-decoder or some
        # final mlp as decoder; and at first embed+pos embed
        # omiting the feedforward (linear drop relu or similar) and pos embed and norm/dyt

        # so step by step
        # f_attn = self.dyt(xs + f_attn)
        # f_attn = xs + f_attn # in 2-layer resid connection helps

        # then add + norm

        # # add another layer- no residual connection; no linear in between; use the same matrices

        attn_arg1 = torch.transpose(f_attn, 1, 2) @ W_KQ1 @ f_attn / self.rho
        softmax_attn_arg1 = torch.softmax(attn_arg1, dim=1)

        f_attn1 = W_PV1 @ f_attn @ softmax_attn_arg1

        # add residual to f_attn1 as well
        # f_attn1 = f_attn1 + f_attn # f_attn has xs resid connect as well÷

        # output = self.linear(f_attn1)
        # out = output[:, :, -1]  # take dim_n output result at last token, for all batches
        # out = self.layernorm(f_attn1[:, :, -1])
        # out = self.dyt(f_attn1[:, :, -1])
        out = f_attn1[:, :, -1]
        return out


class TransformerModelV2L2_Merged(nn.Module):
    """
    Simplest model: with 2 layer; not tied weights but simplified in mat algebra manipulations

    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=2, n_head=1):
        super().__init__()
        assert n_layer == 2  # TODO implement...
        assert n_head == 1  # TODO implement...
        assert (
            dim_attn is None
        )  # TODO implement... for now we take dim_attn == dim_input
        # TODO in multilayer version, add AttnHead class beneath AttnLayer class? forward pass is just loop over nlayer

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PVK1Q1PV = weight_matrix(dim_input, dim_input, mode="default")
        # self.W_KQ1 = weight_matrix(dim_input, dim_input, mode='default')

        self.W_P1V1PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = 1.0

        # self.dyt = DynamicTanh(normalized_shape=dim_input)

        # self.layernorm = nn.LayerNorm(dim_input, eps=1e-12)
        # self.linear = nn.Linear(context_length, context_length, bias=None)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ

        W_PVK1Q1PV = self.W_PVK1Q1PV
        W_P1V1PV = self.W_P1V1PV

        # layer 1
        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / self.rho
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)

        # f_attn = W_PV @ xs @ softmax_attn_arg

        # try add in between; residual connections only make sense when multiple layers really
        # original transf encoder layer: is multihead; add+norm; feed foreard; add + norm; then stack n of these layers; then either transf-decoder or some
        # final mlp as decoder; and at first embed+pos embed
        # omiting the feedforward (linear drop relu or similar) and pos embed and norm/dyt

        # so step by step
        # f_attn = self.dyt(xs + f_attn)
        # f_attn = xs + f_attn # in 2-layer resid connection helps

        # then add + norm

        # # add another layer- no residual connection; no linear in between; use the same matrices

        attn_arg1 = (
            torch.transpose(softmax_attn_arg, 1, 2)
            @ torch.transpose(xs, 1, 2)
            @ W_PVK1Q1PV
            @ xs
            @ softmax_attn_arg
            / self.rho
        )
        softmax_attn_arg1 = torch.softmax(attn_arg1, dim=1)

        f_attn1 = W_P1V1PV @ xs @ softmax_attn_arg @ softmax_attn_arg1

        # add residual to f_attn1 as well
        # f_attn1 = f_attn1 + f_attn # f_attn has xs resid connect as well÷

        # output = self.linear(f_attn1)
        # out = output[:, :, -1]  # take dim_n output result at last token, for all batches
        # out = self.layernorm(f_attn1[:, :, -1])
        # out = self.dyt(f_attn1[:, :, -1])
        out = f_attn1[:, :, -1]
        return out


class TransformerModelV2(nn.Module):
    """
    Simplest model:
    - no positional encoding is used
    - same as V1 but now softmax in place of `linear` self-attention
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1  # TODO implement...
        assert n_head == 1  # TODO implement...
        assert (
            dim_attn is None
        )  # TODO implement... for now we take dim_attn == dim_input
        # TODO in multilayer version, add AttnHead class beneath AttnLayer class? forward pass is just loop over nlayer

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = 1.0

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / self.rho
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = xs + W_PV @ xs @ softmax_attn_arg

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV2nores(TransformerModelV2):
    """
    See docstring TransformerModelV2
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

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        # faster to just use final token as the query, not whole context (we throw it away later)
        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs[:, :, [-1]] / self.rho

        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = (
            W_PV @ xs @ softmax_attn_arg
        )  # the residual stream term "+ xs" has been removed

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV2noresOmitLast(TransformerModelV2):
    """
    See docstring TransformerModelV2
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(
            context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head
        )

        self.dyt = DynamicTanh(normalized_shape=dim_input)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        # rho = n_tokens

        xs_skip_last = xs[:, :, :-1]
        attn_arg = (
            torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ xs[:, :, [-1]] / self.rho
        )

        # p7 Bartlett: "Softmax applied column-wise" (dim = data dim, not token dim)
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = (
            W_PV @ xs_skip_last @ softmax_attn_arg
        )  # the residual stream term "+ xs" has been removed

        # print("shape of f_attn", f_attn.shape)
        # print("shape of xs_skip_last", xs_skip_last.shape)

        # add residual connection TODO@DR modified this

        # print("Shape of f_attn ", f_attn.shape)
        # f_attn = self.dyt(f_attn + xs[:, :, [-1]])
        # f_attn = f_attn + xs[:, :, [-1]]

        # out = f_attn[:, :, -1]  # take dim_n output result at last token, for all batches; this is now torch.Size([80, 16, 1])
        # print("Shape of out ", out.shape)
        return torch.squeeze(f_attn)  # yes - this is the same as they do


class TransformerModelV2L2Skip(TransformerModelV2):
    """
    See docstring TransformerModelV2
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(
            context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head
        )

        self.dyt = DynamicTanh(normalized_shape=dim_input)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        # rho = n_tokens

        xs_skip_last = xs[:, :, :-1]
        attn_arg = (
            torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ xs[:, :, [-1]] / self.rho
        )

        # p7 Bartlett: "Softmax applied column-wise" (dim = data dim, not token dim)
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = (
            W_PV @ xs_skip_last @ softmax_attn_arg
        )  # the residual stream term "+ xs" has been removed

        # print("shape of f_attn", f_attn.shape)
        # print("shape of xs_skip_last", xs_skip_last.shape)

        # add residual connection TODO@DR modified this

        # print("Shape of f_attn ", f_attn.shape)
        # f_attn = self.dyt(f_attn + xs[:, :, [-1]])
        # f_attn = f_attn + xs[:, :, [-1]]

        # out = f_attn[:, :, -1]  # take dim_n output result at last token, for all batches; this is now torch.Size([80, 16, 1])
        # print("Shape of out ", out.shape)
        return torch.squeeze(f_attn)  # yes - this is the same as they do


class TransformerModelV3(nn.Module):
    """
    [DR] 2-layer Simplest model:
    - no positional encoding is used
    - with softmax otw it would just be a linear layer with the matmul and addition of the extra weight matrices

    Notes
     - dim_input - the dimension of input tokens
     - dim_attn  - the dimension of the residual stream (attention head + MLP input and output)
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=2, n_head=1):
        super().__init__()
        assert n_layer == 2
        assert n_head == 1
        assert (
            dim_attn is None
        )  # still take as dim input; keep it as simple as possible

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = 1.0  # scaling used in Bartlett 2023

        self.W_KQ1 = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV1 = weight_matrix(dim_input, dim_input, mode="default")

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        W_KQ1 = self.W_KQ1
        W_PV1 = self.W_PV1

        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / self.rho  # scaling as in V2
        # print("attn arg shape", attn_arg.shape)  # ([80, 500, 500]) context len
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = xs + W_PV @ xs @ softmax_attn_arg  # add; we would norm here too

        # print("f_attn shape", f_attn.shape)

        # 2-nd layer - identical add manually here for simplicity TODO: check in vit what is fed to next layer
        attn_arg1 = torch.transpose(f_attn, 1, 2) @ W_KQ1 @ f_attn / self.rho
        softmax_attn_arg1 = torch.softmax(attn_arg1)
        f_attn1 = f_attn + W_PV1 @ f_attn @ softmax_attn_arg1

        out = f_attn1[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV6noresOmitLast(TransformerModelV2):
    """
    See docstring TransformerModelV2
    with simple tanh
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(
            context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head
        )

        self.dyt = DynamicTanh(normalized_shape=dim_input)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        # rho = n_tokens

        xs_skip_last = xs[:, :, :-1]
        attn_arg = (
            torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ xs[:, :, [-1]] / self.rho
        )

        # p7 Bartlett: "Softmax applied column-wise" (dim = data dim, not token dim)
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = (
            W_PV @ xs_skip_last @ softmax_attn_arg
        )  # the residual stream term "+ xs" has been removed

        f_attn = self.dyt(f_attn)

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelV7noresOmitLast(TransformerModelV2):
    """
    See docstring TransformerModelV2
    with simple tanh
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(
            context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head
        )

        self.dyt = DynamicTanh(normalized_shape=dim_input)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()
        xs = self.dyt(xs)

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        # rho = n_tokens

        xs_skip_last = xs[:, :, :-1]
        attn_arg = (
            torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ xs[:, :, [-1]] / self.rho
        )

        # p7 Bartlett: "Softmax applied column-wise" (dim = data dim, not token dim)
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = (
            W_PV @ xs_skip_last @ softmax_attn_arg
        )  # the residual stream term "+ xs" has been removed

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelV8noresOmitLast(TransformerModelV1):
    """
    See docstring TransformerModelV1 and V6
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(
            context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head
        )

        self.dyt = DynamicTanh(normalized_shape=dim_input)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        rho = n_tokens - 1

        xs_skip_last = xs[:, :, :-1]

        projection_estimate = xs_skip_last @ torch.transpose(xs_skip_last, 1, 2) / rho

        f_attn_approx = W_PV @ projection_estimate @ W_KQ @ xs[:, :, [-1]]

        f_attn_approx = self.dyt(f_attn_approx)

        out = f_attn_approx[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelV9noresOmitLast(TransformerModelV1):
    """
    See docstring TransformerModelV1 and V8
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(
            context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head
        )

        self.dyt = DynamicTanh(normalized_shape=dim_input)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()
        xs = self.dyt(xs)

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        rho = n_tokens - 1

        xs_skip_last = xs[:, :, :-1]

        projection_estimate = xs_skip_last @ torch.transpose(xs_skip_last, 1, 2) / rho

        f_attn_approx = W_PV @ projection_estimate @ W_KQ @ xs[:, :, [-1]]

        f_attn_approx = self.dyt(f_attn_approx)

        out = f_attn_approx[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelV10(nn.Module):
    """
    softmax 1 layer 2 heads; each head has weight matrices of dim_n / 2 (dim n should be multiple of 2)
    concat
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=2):
        super().__init__()
        assert n_layer == 1
        assert n_head == 2  # TODOthis is manual for now
        assert dim_attn is None

        # attention matrices (split by to 2 heads.)
        self.num_heads = 2
        self.head_dim = dim_input // 2
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = 1.0
        # add an output layer after concat
        self.linear = nn.Linear(
            context_length - 1, context_length - 1, bias=None
        )  # the context_len includes the token to be corrupted

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        # rho = n_tokens

        xs_skip_last = xs[:, :, :-1]

        # make an approximation
        # first part
        k_heads = xs_skip_last.view(
            xs_skip_last.shape[0], xs_skip_last.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        kwkq = k_heads @ W_KQ
        q_last = xs[:, :, [-1]]
        q = q_last.view(
            q_last.shape[0], q_last.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        attn_arg = kwkq @ q / self.rho

        # p7 Bartlett: "Softmax applied column-wise" (dim = data dim, not token dim)
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = (
            W_PV @ k_heads @ softmax_attn_arg
        )  # the residual stream term "+ xs" has been removed

        print("f_attn shape was", f_attn.shape)
        output = self.linear(f_attn)
        print("so now output after lin out layer is ", output.shape)
        out = output[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV10noresOmitLast(TransformerModelV2):
    """
    2 heads ; softmax; one final linear layer no bias
    with simple tanh
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(
            context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head
        )

        # attention matrices (manually by head)
        self.W_KQ = weight_matrix(dim_input // 2, dim_input // 2, mode="default")
        self.W_PV = weight_matrix(dim_input // 2, dim_input // 2, mode="default")
        self.rho = 1.0  # scaling by num tokens in forward rho = n_tokens - 1

        self.W_KQ1 = weight_matrix(dim_input // 2, dim_input // 2, mode="default")
        self.W_PV1 = weight_matrix(dim_input // 2, dim_input // 2, mode="default")

        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        W_KQ1 = self.W_KQ1
        W_PV1 = self.W_PV1

        xs_skip_last = xs[:, :, :-1]
        attn_arg = (
            torch.transpose(xs_skip_last, 1, 2)
            @ torch.concatenate((W_KQ, W_KQ1), axis=0)
            / self.rho
        )  # this thing where only last token so dim is 1 and not cont_elnt - 1
        attn_arg1 = (
            torch.transpose(xs_skip_last, 1, 2) @ W_KQ1 @ xs[:, :, [-1]] / self.rho
        )
        print("attn_arg shape was", attn_arg.shape)
        print("attn_arg shape was", attn_arg1.shape)

        # p7 Bartlett: "Softmax applied column-wise" (dim = data dim, not token dim)
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        softmax_attn_arg1 = torch.softmax(attn_arg1, dim=1)

        f_attn = (
            W_PV @ xs_skip_last @ softmax_attn_arg
        )  # the residual stream term "+ xs" has been removed
        f_attn1 = W_PV1 @ xs_skip_last @ softmax_attn_arg1

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        print("f_attn shape was", f_attn.shape)
        output = self.linear(torch.concatenate(f_attn, f_attn1, axis=1))

        # print("so now output after lin out layer is ", output.shape)
        out = output[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV3nores(TransformerModelV3):
    """
    See docstring TransformerModelV3
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=2, n_head=1):
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
        W_KQ = self.W_KQ
        W_PV = self.W_PV

        W_KQ1 = self.W_KQ1
        W_PV1 = self.W_PV1

        attn_arg = (
            torch.transpose(xs, 1, 2) @ W_KQ @ xs[:, :, [-1]] / self.rho
        )  # take last as they do in V2nores
        # print("attn arg shape", attn_arg.shape)  # ([80, 500, 500]) context len
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = W_PV @ xs @ softmax_attn_arg  # no add

        # print("f_attn shape", f_attn.shape)

        # 2-nd layer - identical add manually here for simplicity TODO: check in vit what is fed to next layer
        attn_arg1 = torch.transpose(f_attn, 1, 2) @ W_KQ1 @ f_attn / self.rho
        softmax_attn_arg1 = torch.softmax(attn_arg1, dim=1)
        f_attn1 = W_PV1 @ f_attn @ softmax_attn_arg1  # no add

        out = f_attn1[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV3noresOmitLast(TransformerModelV3):
    """
    See docstring TransformerModelV3
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=2, n_head=1):
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
        W_KQ = self.W_KQ
        W_PV = self.W_PV

        W_KQ1 = self.W_KQ1
        W_PV1 = self.W_PV1

        xs_skip_last = xs[:, :, :-1]
        attn_arg = (
            torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ xs[:, :, [-1]] / self.rho
        )  # take last as they do in V2nores
        # print("attn arg shape", attn_arg.shape)  # ([80, 500, 500]) context len
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = W_PV @ xs_skip_last @ softmax_attn_arg  # no add

        # print("f_attn shape", f_attn.shape)

        # 2-nd layer - identical add manually here for simplicity TODO: check in vit what is fed to next layer
        attn_arg1 = (
            torch.transpose(f_attn, 1, 2) @ W_KQ1 @ f_attn / self.rho
        )  # TODO: I don't think the skip last affects this portion
        softmax_attn_arg1 = torch.softmax(attn_arg1, dim=1)
        f_attn1 = W_PV1 @ f_attn @ softmax_attn_arg1  # no add

        # print("\n W_KQ \n", W_KQ)
        # print("\n W_KQ1 \n", W_KQ1)

        out = f_attn1[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV4noresOmitLast(nn.Module):
    """
    [TODO DR] 2-layer Simplest model: - this is not right - have to do math on paper
    - no positional encoding is used
    - with softmax otw it would just be a linear layer with the matmul and addition of the extra weight matrices

    Notes
     - dim_input - the dimension of input tokens
     - dim_attn  - the dimension of the residual stream (attention head + MLP input and output)
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=2, n_head=1):
        super().__init__()
        assert n_layer == 2
        assert n_head == 1
        assert (
            dim_attn is None
        )  # still take as dim input; keep it as simple as possible

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = 1.0  # scaling by num tokens in forward rho = n_tokens - 1

        self.W_KQ1 = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV1 = weight_matrix(dim_input, dim_input, mode="default")

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()
        W_KQ = self.W_KQ
        W_PV = self.W_PV

        W_KQ1 = self.W_KQ1
        W_PV1 = self.W_PV1

        self.rho = n_tokens - 1

        xs_skip_last = xs[:, :, :-1]

        # attn_arg = torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ xs_skip_last / rho
        projection_estimate = (
            xs_skip_last @ torch.transpose(xs_skip_last, 1, 2) / self.rho
        )

        f_attn_approx = W_PV @ projection_estimate @ W_KQ @ xs

        proj2 = f_attn_approx @ torch.transpose(f_attn_approx, 1, 2)

        # synthetic second linear projection  to simulate 2 linear layers
        f_attn_approx1 = W_PV1 @ proj2 @ W_KQ1 @ projection_estimate
        out = f_attn_approx1[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelQKVnores(nn.Module):
    """
    Simplest model:
    - no positional encoding is used
    - same as V1 but now softmax in place of `linear` self-attention
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        self.W_Q = weight_matrix(dim_input, dim_input, mode="default")
        self.W_K = weight_matrix(dim_input, dim_input, mode="default")
        self.W_V = weight_matrix(dim_input, dim_input, mode="default")

        self.rho = 1.0

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        ####batchsz, n_dim, n_tokens = xs.size()

        Q = self.W_Q @ xs[:, :, [-1]]
        K = self.W_K @ xs
        V = self.W_V @ xs

        # QK_d = (Q @ K.T) / self.rho
        KQ_d = (
            torch.transpose(K, 1, 2) @ Q / self.rho
        )  # this is tensor-argument of softmax attention
        prob = torch.softmax(KQ_d, dim=1)
        attention = V @ prob

        out = attention[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches

        return out


def load_model_from_rundir(dir_run, epoch_int=None):
    """
    Step 0: assume rundir has structure used in io_dict above
    Step 1: read dim_n and context length from runinfo.txt
    Step 2: load model at particular epoch from model_checkpoints (if epoch unspecified, load model_end.pth)
    """
    # load runinfo settings
    runinfo_dict = load_runinfo_from_rundir(dir_run)
    dim_n = runinfo_dict["dim_n"]
    context_length = runinfo_dict["context_len"]
    nn_model_str = runinfo_dict["model"]
    epochs = runinfo_dict["epochs"]

    if epoch_int is not None:
        model_fpath = (
            dir_run
            + os.sep
            + "model_checkpoints"
            + os.sep
            + "model_e%d.pth" % epoch_int
        )
    else:
        model_fpath = (
            dir_run + os.sep + "model_checkpoints" + os.sep + "model_final.pth"
        )

    # nn_model_str is a string like 'TransformerModelV1' or its alias 'TV1' (short form)
    if nn_model_str in MODEL_CLASS_ALIAS_TO_STR.keys():
        nn_model_str = MODEL_CLASS_ALIAS_TO_STR[nn_model_str]
        nn_class = MODEL_CLASS_FROM_STR[nn_model_str]["class"]
    else:
        nn_class = MODEL_CLASS_FROM_STR[nn_model_str]["class"]

    net = nn_class(
        context_length, dim_n
    )  # TODO this currently assumes all models have same two inputs
    print("loading model at:", model_fpath, "...")
    net.load_state_dict(torch.load(model_fpath))
    net.eval()

    return net


def load_modeltype_from_fname(fname, dir_models=DIR_MODELS, model_params=None):
    """
    Trained model checkpoint files are assumed to follow a certain naming convention, and are placed in DIR_MODELS
        e.g. models\basicnetmult_chkpt_e240_L100_n128.pth

    If model_params is None, then the model params will be inferred from the filename itself
    """
    print("\nLoading model checkpoint from file...")
    fpath = dir_models + os.sep + fname
    print("...", fpath)

    if model_params is None:
        print("model_params is None; inferring class init from filename...")
        model_type = fname.split("_")[0]
        context_length = int(fname.split("_L")[1].split("_")[0])
        dim_input = int(fname.split("_n")[1].split("_")[0])
    else:
        model_type = model_params["nn_model"]
        context_length = int(model_params["context_length"])
        dim_input = int(model_params["dim_input"])

    # nn_model_str is a string like 'TransformerModelV1' or its alias 'TV1' (short form)
    if model_type in MODEL_CLASS_ALIAS_TO_STR.keys():
        nn_model_str = MODEL_CLASS_ALIAS_TO_STR[model_type]
        nn_class = MODEL_CLASS_FROM_STR[nn_model_str]["class"]
    else:
        nn_class = MODEL_CLASS_FROM_STR[model_type]["class"]

    print(
        "class:",
        model_type,
        "\n\tcontext_length=%d, dim_input=%d" % (context_length, dim_input),
    )
    net = nn_class(context_length, dim_input)
    print("loading weights from fpath:", fpath)
    net.load_state_dict(
        torch.load(fpath, weights_only=True)
    )  # avoid FutureWarning in latest Torch ~2.5

    return net, model_type, context_length, dim_input


def count_parameters(model):
    """
    Use: print parameters of torch nn.Module in nice manner
    From: https://stackoverflow.com/questions/67546610/pretty-print-list-without-module-in-an-ascii-table

    Table is just a list of lists
    """

    def pretty_print(table, ch1="-", ch2="|", ch3="+"):
        if len(table) == 0:
            return
        max_lengths = [
            max(column)
            for column in zip(*[[len(cell) for cell in row] for row in table])
        ]
        for row in table:
            print(ch3.join(["", *[ch1 * l for l in max_lengths], ""]))
            print(
                ch2.join(
                    [
                        "",
                        *[
                            ("{:<" + str(l) + "}").format(c)
                            for l, c in zip(max_lengths, row)
                        ],
                        "",
                    ]
                )
            )
        print(ch3.join(["", *[ch1 * l for l in max_lengths], ""]))

    total_params = 0
    table = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.append([name, str(params)])
        total_params += params
    print("(Modules | Parameters)")
    pretty_print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


MODEL_CLASS_FROM_STR = {
    "TransformerModelV1": {"class": TransformerModelV1, "alias": "TV1"},
    "TransformerModelV1nores": {"class": TransformerModelV1nores, "alias": "TV1nr"},
    "TransformerModelV1noresForceDiag": {
        "class": TransformerModelV1noresForceDiag,
        "alias": "TV1nrFD",
    },
    "TransformerModelV1noresOmitLast": {
        "class": TransformerModelV1noresOmitLast,
        "alias": "TV1nrOL",
    },
    "TransformerModelV1noresForceDiagAndOmitLast": {
        "class": TransformerModelV1noresForceDiagAndOmitLast,
        "alias": "TV1nrFDOL",
    },
    "TransformerModelV2": {"class": TransformerModelV2, "alias": "TV2"},
    "TransformerModelV2nores": {"class": TransformerModelV2nores, "alias": "TV2nr"},
    "TransformerModelV2noresOmitLast": {
        "class": TransformerModelV2noresOmitLast,
        "alias": "TV2nrOL",
    },
    "TransformerModelV3noresOmitLast": {
        "class": TransformerModelV3noresOmitLast,
        "alias": "TV3nrOL",
    },
    "TransformerModelQKVnores": {"class": TransformerModelQKVnores, "alias": "TQKVnr"},
    "TransformerModelV4noresOmitLast": {
        "class": TransformerModelV4noresOmitLast,
        "alias": "TV4nrOL",
    },
    "TransformerModelV6noresOmitLast": {
        "class": TransformerModelV6noresOmitLast,
        "alias": "TV6nrOL",
    },
    "TransformerModelV7noresOmitLast": {
        "class": TransformerModelV7noresOmitLast,
        "alias": "TV7nrOL",
    },
    "TransformerModelV8noresOmitLast": {
        "class": TransformerModelV8noresOmitLast,
        "alias": "TV8nrOL",
    },
    "TransformerModelV9noresOmitLast": {
        "class": TransformerModelV9noresOmitLast,
        "alias": "TV9nrOL",
    },
    "TransformerModelV11": {"class": TransformerModelV11, "alias": "TV11nr"},
    "TransformerModelV11SkipLast": {
        "class": TransformerModelV11SkipLast,
        "alias": "TV11nrOL",
    },
    "TransformerModelV11SkipLastOneOnly": {
        "class": TransformerModelV11SkipLastOneOnly,
        "alias": "TV11nrOLOO",
    },
    "TransformerModelV2Tied2": {"class": TransformerModelV2Tied2, "alias": "TV2T2"},
    "TransformerModelV2L2": {"class": TransformerModelV2L2, "alias": "TV2L2"},
    "TransformerModelV2L2_Merged": {
        "class": TransformerModelV2L2_Merged,
        "alias": "TV2L2M",
    },
    "TransformerModelV2Tied2_Skip": {
        "class": TransformerModelV2Tied2_Skip,
        "alias": "TV2T2S",
    },
}
# define companion dict mapping alias to class string
MODEL_CLASS_ALIAS_TO_STR = {v["alias"]: k for k, v in MODEL_CLASS_FROM_STR.items()}


if __name__ == "__main__":
    # sequence: three vectors from R^2
    sample_sequence = np.array([[[1, 1, 1], [2, 3, 4]]]).astype("float32")

    print("Prep sample input for each model:")
    print("=" * 20)
    sample_input = torch.tensor(sample_sequence)  # add a batch dimension to front
    print("sample_input.size()", sample_input.size())
    batchsz, n_dim, n_tokens = sample_input.size()

    # Demonstrate parameter count table and sample outputs
    print("\nModel (sequence): ReturnLastToken()")
    model = ReturnLastToken()
    count_parameters(model)
    print("sample_output:", model(sample_input))

    print(sample_input)
    print(torch.transpose(sample_input, 1, 2))

    print("\nModel (sequence): TransformerModelV1()")
    model = TransformerModelV1(n_tokens, n_dim)
    count_parameters(model)
    print("sample_output:", model(sample_input))

    print("\nModel (sequence): TransformerModelV2()")
    model = TransformerModelV2(n_tokens, n_dim)
    count_parameters(model)
    print("sample_output:", model(sample_input))

    print("\nModel (sequence): TransformerModelV3()")
    model = TransformerModelV2(n_tokens, n_dim)
    count_parameters(model)
    print("sample_output:", model(sample_input))

    print("\nModel (sequence): TransformerModelV2()")
    model = TransformerModelV4(n_tokens, n_dim)
    count_parameters(model)
    print("sample_output:", model(sample_input))

    # ========= scrap
    print("\n========= scrap")
    # emulate forwards pass of linear self-attention (Bartlett 2023, Eq. 3)
    W_PV = torch.randn(2, 2)
    W_KQ = torch.randn(2, 2)
    rho = n_tokens

    print(type(W_PV), W_PV.dtype)
    print(type(W_KQ), W_KQ.dtype)
    print(type(sample_input), sample_input.dtype)

    attn_arg = torch.transpose(sample_input, 1, 2) @ W_KQ @ sample_input / rho
    f_attn = sample_input + W_PV @ sample_input @ attn_arg

    print(f_attn)
    out = f_attn[:, :, -1]  # take dim_n output result at last token, for all batches
    print(out, out.size())
    print(out.flatten(), out.flatten().size())

    print("\nSynthesize train/test data for sequence models")
    print("=" * 20)
    x_train, y_train, x_test, y_test, _, _ = data_train_test_split_linear(
        context_len=100,
        dim_n=128,
        num_W_in_dataset=1000,
        context_examples_per_W=1,
        test_ratio=0.2,
        verbose=True,
        seed=0,
    )
    print("x_train.shape:", x_train.shape)

    print("\nSample input -> output for ReturnLastToken() sequence model")
    print("=" * 20)
    model = ReturnLastToken()

    single_batch = x_train[0, :, :].unsqueeze(0)  # add back trivial batch dim
    print("\tsingle_batch tensor.size:", single_batch.shape)
    out_from_single_batch = model(single_batch)
    print("\tout_from_single_batch tensor.size:", out_from_single_batch.shape)

    full_batch = x_train[:, :, :]
    print("\n\tfull_batch tensor.size:", full_batch.shape)
    out_from_full_batch = model(full_batch)
    print("\tout_from_full_batch tensor.size:", out_from_full_batch.shape)

    print("\nSample input -> output for TransformerModelV1() sequence model")
    print("=" * 20)
    model = TransformerModelV1(100, 128)

    single_batch = x_train[0, :, :].unsqueeze(0)
    print("\tsingle_batch tensor.size:", single_batch.shape)
    out_from_single_batch = model(single_batch)
    print("\tout_from_single_batch tensor.size:", out_from_single_batch.shape)

    full_batch = x_train[:, :, :]
    print("\n\tfull_batch tensor.size:", full_batch.shape)
    out_from_full_batch = model(full_batch)
    print("\tout_from_full_batch tensor.size:", out_from_full_batch.shape)

    print("\nSample input -> output for TransformerModelV2() sequence model")
    print("=" * 20)
    model = TransformerModelV2(100, 128)

    single_batch = x_train[0, :, :].unsqueeze(0)
    print("\tsingle_batch tensor.size:", single_batch.shape)
    out_from_single_batch = model(single_batch)
    print("\tout_from_single_batch tensor.size:", out_from_single_batch.shape)

    full_batch = x_train[:, :, :]
    print("\n\tfull_batch tensor.size:", full_batch.shape)
    out_from_full_batch = model(full_batch)
    print("\tout_from_full_batch tensor.size:", out_from_full_batch.shape)

    print("\nSample input -> output for TransformerModelV2() sequence model")
    print("=" * 20)
    model = TransformerModelV3(100, 128)

    single_batch = x_train[0, :, :].unsqueeze(0)
    print("\tsingle_batch tensor.size:", single_batch.shape)
    out_from_single_batch = model(single_batch)
    print("\tout_from_single_batch tensor.size:", out_from_single_batch.shape)

    full_batch = x_train[:, :, :]
    print("\n\tfull_batch tensor.size:", full_batch.shape)
    out_from_full_batch = model(full_batch)
    print("\tout_from_full_batch tensor.size:", out_from_full_batch.shape)

    print("\nSample input -> output for TransformerModelV2() sequence model")
    print("=" * 20)
    model = TransformerModelV4(100, 128)

    single_batch = x_train[0, :, :].unsqueeze(0)
    print("\tsingle_batch tensor.size:", single_batch.shape)
    out_from_single_batch = model(single_batch)
    print("\tout_from_single_batch tensor.size:", out_from_single_batch.shape)

    full_batch = x_train[:, :, :]
    print("\n\tfull_batch tensor.size:", full_batch.shape)
    out_from_full_batch = model(full_batch)
    print("\tout_from_full_batch tensor.size:", out_from_full_batch.shape)
