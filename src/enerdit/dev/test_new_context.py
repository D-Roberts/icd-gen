"""
About context
"""
import torch
from torch.distributions import Uniform
import torch.nn as nn
from dgen_for_gaussian import train_dataset, train_loader, normalize


def get_batch_samples_w_context(data, data_context):
    # inputs is a batch
    inputs, target = data
    # print(f"the inputs last otken chekc {inputs[2, :, -1]}") # i think as expected
    # normalize to [0, 1]
    inputs, _ = normalize(inputs, target)
    # print(inputs) # looks ok

    # b, pdim, seq_len = inputs.shape # no context in simple
    b, pdim = inputs.shape

    # Sample time steps
    # tmin = torch.tensor(10 ** (-9))
    tmin = torch.tensor(0.01)
    # Change the noise tmax considering how small is d
    tmax = torch.tensor(100)

    logtmin = torch.log(tmin)
    logtmax = torch.log(tmax)

    logt_distrib = Uniform(low=torch.tensor([logtmin]), high=torch.tensor([logtmax]))
    logt = logt_distrib.sample(torch.tensor([b]))
    t = torch.exp(logt).squeeze(-1)  # like 0.15 to 227 etc
    # print(f"generated t is {t} and shape {t.shape}")
    # print(f"the t {t}")

    # get z for this batch from N(0,I)
    z = torch.randn_like(inputs)
    # print(f"z shape {z.shape}")
    sqrttz = torch.zeros_like(z)

    # I am applying the same t noise accross the sequence in one instance
    # and diffeernt t accross the minibatch

    # sqrttz = torch.einsum('bcd,b->bd', z, torch.sqrt(t))
    sqrttz = torch.einsum("bd,b->bd", z, torch.sqrt(t))  # For simple only 2-dim

    # print(f"the noise*sqrtt last token {sqrttz[0,:,-1]}")

    # test that the broadcasting happened as expected
    # print(f" check {sqrttz[0,1,0] / sqrtt[0]} and {z[0, 1, 0]}") #ok

    # Get noisy seq for the batch
    noisy = torch.zeros_like(inputs)
    noisy += inputs
    noisy += sqrttz

    print(
        f"what does noisy look like {torch.mean(noisy), torch.var(noisy), noisy.shape}"
    )

    # Need to add context for each instance as well
    # for right now they are the same but they will be at a different scale
    # to mimic the original unpatched while the input query is patched down
    # to be passed through the model (subject to change)
    clean_context = inputs.clone()
    noisy_context = noisy.clone()

    # print(f"the fused shape {fused.shape}") # ok double patch dim
    # print(f"the fused last otken chekc {fused[2,:,-1]}")
    # print(f"the noisy last otken chekc {noisy[2,:,-1]}") # i think as expected

    # so now have inputs (clean), target, z, noisy only, fused, t

    # return t, z, target, fused
    return t, z, inputs, noisy, noisy_context, clean_context
    # keep it very simple; the context will be added a sequence embedding
    # and it will be unpatched and projected


for i, data in enumerate(train_loader):
    t, z, clean, noisy, noisy_context, clean_context = get_batch_samples_w_context(
        data, data
    )

    if i == 2:
        break
    print(clean_context)
    print(clean)  # same


class ContextEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size=2):
        """
        encode if 0=clean or 1=noisy akin to question answering bert style embeddings
        for in context tokens

        aim to have noisy first clean second
        will be added to the fused noisy clean context to tell the model what's what
        """
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding_layer(x)


ce = ContextEmbedding(3, 2)  # embed dim is 3

res = ce(torch.LongTensor([1, 0]))
print(res, res.shape)  # 2,3 # 2 because 2 embeddings
# print reshape assume noisy and clean concats
reshape = res.view(1, -1)
print(reshape, reshape.shape)
