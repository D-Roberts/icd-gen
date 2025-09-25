import torch
import torch.nn as nn

import math


# now the sequence has double width due to concat clean and noisy
class Patchencodeding(nn.Module):
    def __init__(self, dim_in, d_model):
        super().__init__()

        # aim to encode the clean and noisy together for dim reduction
        # alternatively I could add the position encode directly in the
        # 200 dim

        # set bias to zero
        self.projection = torch.nn.Linear(dim_in, d_model, bias=False)

    def forward(self, x):
        # print(f"layer in patch encode {self.projection}")
        # print(f"debug patch encode shapes {x.shape}")
        x = self.projection(x)
        # (batch_size, encode_dim, num_patches)
        return x


patches = torch.randn(4, 8, 128)  # B, seq_len, fused patch dim
pemb = Patchencodeding(
    128, 32
)  # first dim should be the patch size and second the emb dim

print(f"pemb layer {pemb}")

encodeded = pemb(patches)

print(
    f"encodeded seq shape {encodeded.shape}"
)  # [4, 8, 32]) as expected (B, seq_len, d_model=encodedim)


########################Add Space encode
class Spaceencodeding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        # TODO@DR check for correctness again
        se = torch.zeros(max_len, d_model)  # (seq_len, d_model)
        # print(f"pos emb starts as {pe.shape}")

        spatial = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # print(f"position is {position} of shape {position.shape}")
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # double the d_model size
        # print(f"div is {div_term} of shape {div_term.shape}")
        # print("torch.sin(position * div_term)", torch.sin(position * div_term).shape)
        # print("pe[:, 0::2]", pe[:, 0::2].shape, pe[:, 0::2])
        # print("pe[:, 1::2]", pe[:, 1::2].shape)

        se[:, 0::2] = torch.sin(spatial * div_term)  # start index at 0 with step size 2
        # print("pe[:, 0::2] and pe", pe[:, 0::2].shape, pe)
        se[:, 1::2] = torch.cos(spatial * div_term)  # start index at 1 with step size 2

        # print("test that the two pe with sin and cos are not eq", pe[:, 0::2] == pe[:, 1::2]) # ok
        self.register_buffer("se", se.unsqueeze(0))  # Add a batch dimension

    def forward(self, x):
        # x is the sequence of patch encodedings, shape (batch_size, seq_len, d_model)
        # We need to add positional encodedings to each element in the sequence
        # for the moment only create the pos encodedings

        # print(self.pe.shape)
        # print(self.pe[:, : x.size(1)].shape)

        return self.se[:, : x.size(1)]


# putting an index on the full sequence including the query last token; so max_len in dev is 8
space_encode_layer = Spaceencodeding(d_model=32, max_len=8)

print(f"space encode layer {space_encode_layer}")

# since the position index must match the encodeded patches, apply pos encode directly on the
# encodeded patches
space_encode = space_encode_layer(encodeded)

print(f"space encodedings shape {space_encode.shape}")  # (B, seq_len, d_model)
# so shape is [1, 8, 32]) - it gets broadcasted accross the batch
# it has the same dim encodeding= d_model as the patch encodeding
# looks ok

total_encode = space_encode + encodeded
print(total_encode.shape)
# TODO@DR
# so if I subselect patches randomly for in-context task - I must first add the spatial before the
# # subselection

# print(torch.arange(0, 8, 2).float()) # (0, 2, 4, 6)
# print((-math.log(10000.0) / 8))
# print(torch.arange(0, 8, 2).float() * (-math.log(10000.0) / 8))
# print(torch.exp(torch.arange(0, 8, 2).float() * (-math.log(10000.0) / 8)))

# for name, param in pemb.named_parameters():
#     print(f"param is {param} and name is {name} ")

# no param in space encodedings - OK


class DyTanh(nn.Module):
    """
    dev dyt

    expected shape_in would be (seq_len, input_dim)

    elementwise layer; so shape out is like shape in.
    """

    def __init__(self, shape_in, alpha_init_value=0.5):
        super().__init__()
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(shape_in))
        # print("tanh weight shape", self.weight.shape)
        self.bias = nn.Parameter(torch.zeros(shape_in))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        # print("In tanh x shape", x.shape)
        x = x * self.weight + self.bias
        return x


# encodedings for instance would be 4, 8, 32; here channels would be 32
dyt = DyTanh(shape_in=(8, 32))
print(dyt)

x = dyt(total_encode)
print(f"in shape {total_encode.shape} and out shape {x.shape}")


class TimeEncoding(nn.Module):
    """We just have to have them.

    logt will be drawn from a U(logtmin, logtmax), will tmin = 10**-9 and tmax = 10**3

    each token in sequence will have an associated t. encode to same d_model and add to the
    pathc and space encodedings.

    encodedings are sin cos

    t is a seq len vector in this formulation with a context prompt.
    (as of right now)

    """

    def __init__(
        self, d_model, frequency_encodeding_size=256, mint=10 ** (-9), maxt=10**3
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
        )  # the device will need to be given
        args = t[:, None].float() * freqs[None]  # for each t in the t tensor
        print(args)
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


tencode = TimeEncoding(32, 256, 10 ** (-9), 10**3)
# print(tencode)
# print(
#     tencode.time_encodeding(torch.tensor([1, 3, 4]), 256, 10 ** (-9), 10**3).shape
# )  # 128

t_emb = tencode.forward(torch.tensor([1, 3, 4, 5, 6, 7, 8, 2]))

# print(f"time encodedings for a time tensor {torch.tensor([1, 3, 4])} are {t_emb} of shape {t_emb.shape}")
# # one encodeding for each t; and will have 1 t for each token in the sequence
# print(total_encode.shape)
# print(t_emb.shape)
# print(total_encode + t_emb)
# As of right now time_emb is like space_emb - and it gets added accross the batch so it assumes
# the same t tensor for each sequence in the batch TODO@DR will have to reason about this setup

logt = torch.empty(4).uniform_(math.log(10 ** (-9)), math.log(10**3))
print(f"min logt {torch.min(logt)} and max {torch.max(logt)}")
print(math.log(10 ** (-9)))  # -20
print(math.log(10**3))  # 6.9

t = torch.exp(torch.empty(4).uniform_(math.log(10 ** (-9)), math.log(10**3)))
# t cannot be negative
print(1 / torch.sqrt(torch.min(t)), 1 / torch.sqrt(torch.max(t)))  # these can be large
# z
# torch.randn(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) → Tensor
z = torch.randn(4)
print(z)
zsqrt = torch.sqrt(t) * z

# ------------------
z = torch.randn(3, 4)  # noise for one patch of size 4 in a batch of 3
time_score = torch.randn(3, 1)  # say this is predicted time score
d = 4  # say patch is 2x2

# t for this seq query - the t in noise
t_for_q = t[-1]  # very small here
print(f"t_for_q {t_for_q}")

znormedsq = torch.norm(z, p=2, dim=-1) ** 2
print(f"znormed sq {znormedsq}")

term1 = (t_for_q / d) * time_score
print(f"term1 in time loss {term1}")  # very small bc of t

# # print(f"z normed shape {znormedsq.shape}")
term2 = 0.5 * (1 - znormedsq / d)
print(f"term2 in time loss {term2}")

ltime = (term1 - term2) ** 2
print(f"batch mean time loss {ltime.mean(dim=-1)}")  # looks reasonable

####################the space loss
# d is the patch size; going with the patch rather than the double patch here
# use same z and t

# the Space Final pred score would be a matrix shape of z
sp = torch.randn(3, 4)
# space loss term1 (as in eq 43); non neg and non-zero
term1 = math.sqrt(t_for_q / d) * sp
print(f"term1 in space loss {term1}")

term2 = z / math.sqrt(d)
print(f"term2 in space loss {term2}")
subtract = term1 - term2
print(f"subtract in space loss {subtract}")

# print(f"in space loss subtr shape {subtract.shape}") #3, 64
lspace = (torch.norm(subtract, p=2, dim=1)) ** 2
# take norm over the input dim
print(f"what is lspace {lspace} and over minibatch {lspace.mean()}")
# seems reasonable
