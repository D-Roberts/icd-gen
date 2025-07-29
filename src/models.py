import numpy as np

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm

import warnings


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


def build_model(conf):
    if conf["family"] == "gpt2":
        model = TransformerModel(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=conf["n_embd"],
            n_layer=conf["n_layer"],
            n_head=conf["n_head"],
        )
    else:
        raise NotImplementedError

    return model


    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models

# this is from icl stfd code to work with the HuggingFace TODO@DR Note that if I want to strip the model of more things,
# there are more in the HF config. But at a certain point i'll have to take the clone down and hack into it again
# same with diffusers
class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)

        # print("GPT2 decoder model parameters:")
        # m = self._backbone
        # for name, param in m.state_dict().items():
        #     print(f"Layer: {name}, Shape: {param.shape}")

        self._read_out = nn.Linear(n_embd, 1)


    def forward(self, xs, inds=None):
        if inds is None:
            inds = torch.arange(xs.shape[1])
        else:
            inds = torch.tensor(inds)
        
        embeds = self._read_in(xs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs TODO@DR: change this for the input shape in the icd currently


# TODO@DR: This will be exactly the baseline from ic-denoiser paper
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



class TransformerModelV2noresOmitLast(TransformerModelV2):
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



# TODO@DR add here my others

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


MODEL_CLASS_FROM_STR = {
    "TransformerModelV1": {"class": TransformerModelV1, "alias": "TV1"},
    "TransformerModelV1noresOmitLast": {
        "class": TransformerModelV1noresOmitLast,
        "alias": "TV1nrOL",
    },
    "TransformerModelV1noresForceDiagAndOmitLast": {
        "alias": "TV1nrFDOL",
    },
    "TransformerModelV2": {"class": TransformerModelV2, "alias": "TV2"},
    "TransformerModelV2noresOmitLast": {
        "class": TransformerModelV2noresOmitLast,
        "alias": "TV2nrOL",
    },
    "TransformerModelV3noresOmitLast": {
        "class": TransformerModelV3noresOmitLast,
        "alias": "TV3nrOL",
    },
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




