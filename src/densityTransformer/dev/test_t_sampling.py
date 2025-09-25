""""
About time.

"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Uniform
import math

tmin = torch.tensor(10 ** (-9))
tmax = torch.tensor(1000)

t_distrib = Uniform(low=torch.tensor([tmin]), high=torch.tensor([tmax]))

# sample one value at a time
d = torch.tensor(4096)

# print(f"if d=4096 and tmax=1000 than max t/d is {1000/4096}")  # 0.244140625
# print(f"so then if my d is 64, tmax might be {0.244140625*64}")  # 15.625
# for i in range(10):
#     t = t_distrib.sample()
#     print(t)
#     # if d = 4096 as for 64x64 images
#     print(f"t/d if d=64 {t/d}")  # if t is tensor t/d is tensor as well; ok
#     # vals like t/d tensor([0.1449]) if d =4096
#     # vals like 10.5347 if d = 64 like I have now

##################### I think this below is the right t sampling but must
# must adjust tmax I think for my dimensions d

# if logt U(logtmin, logtm)
logtmin = torch.log(tmin)
logtmax = torch.log(tmax)

logt_distrib = Uniform(low=torch.tensor([logtmin]), high=torch.tensor([logtmax]))

for i in range(3):
    logt = logt_distrib.sample()
    print(f"generated logt is {logt}")
    # if d = 4096 as for 64x64 images
    t = torch.exp(logt)  # like 0.15 to 227 etc
    print(f"thent= exp (logt) is {torch.exp(logt)}")

    print(
        f"t/d if d=4096 {t/d}"
    )  # like 0.0007 or 0.1336 0 even very low vals like 0.00005

    print(f"sqrt(t/d) if d=4096 {torch.sqrt(t/d)}")  # like 0.4625, 0.0031 etc

###################################
# what kind of number z/sqrt(d) would be
from torch.distributions import MultivariateNormal

zsampler = MultivariateNormal(torch.zeros(d), torch.eye(d))  # dimension is d
z = zsampler.sample()
# print(z.shape)  # 4096
# print(
#     f"what is z/sqrt(d) like here sqrt(d) is 64 {z/torch.sqrt(d)}"
# )  # -0.0067,  0.0107,  0.0043
# znormsqd = torch.norm(z, p=2) ** 2
# print(f"znormsqrd/d is then {znormsqd / d}")  # 1.013626217842102

#########################
# what kind of numbers are U
# for px in np.arange(0.001, 0.999, 0.05):
#     print(
#         f"U would be {-torch.log(torch.tensor(px))} for prob {px} with log {torch.log(torch.tensor(px))}"
#     )
# right, U is basically negative log likelihood which we make neg log lik to be a sum and pos
# to drive down to 0 by optimization


#########################Get a better sense of the needed time step embedding

x = torch.randn((5, 128)).unsqueeze(1)
# print(x.shape)
y = torch.tile(x, (1, 8, 1))
# print(y.shape)
# print(y[:, 0, :] == y[:, 1, :])


##########################
class TimeEmbedding(nn.Module):
    """We just have to have them.

    logt will be drawn from a U(logtmin, logtmax), will tmin = 10**-9 and tmax = 10**3

    each token in sequence will have an associated t. embed to same d_model and add to the
    pathc and space embeddings.

    embeddings are sin cos

    as of right now I have same t accross the sequence
    I embed it with this
    I tile it for seq len - identical for each sequence token
    I concat on the embedding dimension to the x input in enerdit block

    """

    def __init__(
        self,
        d_model,
        frequency_embedding_size=32,
        mint=1 / (10**3),
        maxt=1 / (10 ** (-9)),
    ):
        super().__init__()
        self.time_embedder = nn.Sequential(
            nn.Linear(frequency_embedding_size, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.mint = mint
        self.maxt = maxt

    @staticmethod
    def time_embedding(t, dim, mint, maxt):
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
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding  # this will be shape (len of t, 256)

    def forward(self, t):
        t_freq = self.time_embedding(
            t, self.frequency_embedding_size, self.mint, self.maxt
        )
        time_embed = self.time_embedder(t_freq)
        return time_embed


tembed = TimeEmbedding(32, 256, 1 / (10 ** (3)), 1 / (10 ** (-9)))
# print(tembed)
# print(
#     tembed.time_embedding(torch.tensor([1, 3, 4]), 256, 10 ** (-9), 10**3).shape
# )  # 128
# for instance if I have
t_as_in_gen_now = torch.exp(
    torch.empty(5).uniform_(math.log(10 ** (-9)), math.log(10**3))
)
t_emb = tembed.forward(t_as_in_gen_now)
print(f"for t {t_as_in_gen_now} the time embed is {t_emb}")

# the other way to gen
logt = logt_distrib.sample(torch.tensor([5]))
print(f"generated logt is {logt}")
# if d = 4096 as for 64x64 images
t = torch.exp(logt)  # like 0.15 to 227 etc
print(f"thent= exp (logt) is {torch.exp(logt)}")

# vs
print(f"t_as_in_gen_now {t_as_in_gen_now}")

#############################
batch_size = 5

for i in range(3):
    t = torch.randint(
        0,
        1000,
        (batch_size,),
    )
    print(
        f"t linear schedule from DiT in one batch step {t}"
    )  # [640, 606, 342,  15, 829]
    # ok - comparable numbers
