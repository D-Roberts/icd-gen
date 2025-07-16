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


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


# from tspiras icl


class NeuralNetwork(nn.Module):
    def __init__(self, in_size=50, hidden_size=1000, out_size=1):
        super(NeuralNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ParallelNetworks(nn.Module):
    def __init__(self, num_models, model_class, **model_class_init_args):
        super(ParallelNetworks, self).__init__()
        self.nets = nn.ModuleList(
            [model_class(**model_class_init_args) for i in range(num_models)]
        )

    def forward(self, xs):
        assert xs.shape[0] == len(self.nets)

        for i in range(len(self.nets)):
            out = self.nets[i](xs[i])
            if i == 0:
                outs = torch.zeros(
                    [len(self.nets)] + list(out.shape), device=out.device
                )
            outs[i] = out
        return outs


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

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs


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


class TransformerModelV3(nn.Module):
    """
    [DR] 2-layer Simplest model:
    - no positional encoding is used
    - `linear self-attention` (no softmax wrapper used) in both layers
    - TODO: with softmax otw it would just be a linear layer with the matmul and addition of the extra weight matrices

    Notes
     - dim_input - the dimension of input tokens
     - dim_attn  - the dimension of the residual stream (attention head + MLP input and output)
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=2, n_head=1):
        super().__init__()
        assert n_layer == 2
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = context_length  # scaling used in Bartlett 2023

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
        rho = n_tokens

        W_KQ1 = self.W_KQ1
        W_PV1 = self.W_PV1

        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / rho
        print("attn arg shape", attn_arg.shape)  # ([80, 500, 500]) context len
        f_attn = xs + W_PV @ xs @ attn_arg  # add; we would norm here too
        print("f_attn shape", f_attn.shape)

        # 2-nd layer
        attn_arg1 = torch.transpose(f_attn, 1, 2) @ W_KQ1 @ f_attn / rho
        f_attn1 = f_attn + W_PV1 @ f_attn @ attn_arg1

        out = f_attn1[
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
    "TransformerModelQKVnores": {"class": TransformerModelQKVnores, "alias": "TQKVnr"},
}
# define companion dict mapping alias to class string
MODEL_CLASS_ALIAS_TO_STR = {v["alias"]: k for k, v in MODEL_CLASS_FROM_STR.items()}


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        # xs, ys = xs.cuda(), ys.cuda()
        xs, ys = xs.to(device), ys.to(device)

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            # model.cuda()
            model.to(device)

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
