"""
Goal here is to get the simplest pos embedding of transformers model
that can learn patch associations in context 
so that I can derive a DAM energy for it and 1-step SGD update too.

Seed model is from jelassi's.

Transformer modle M: with input in d X D. Want M to do patch associations on
D for all j in D indeces of patches as more described below. In a further simplified
setting.

If data distribution is simply Gaussian - with no spatial structure then not
learned by one-layer transf. Key point here - transf need spatial structure.
"""

# first understand EXACTLY what they have.

# the datagen
# they've got a classif task of y=1, -1 labels, 1 per instance.
# the partitioning into groups of indeces.

# Model M does patch association if for i in group Sl, its positional encoding mainly
# attends to patches with indices in the same group. We want top k = group cardinality
# to be patches from the same group. And that happens for all groups.

# Nearby pathces should have similar pos encodings. Well I think in group indeces
# should have similar pos encodings.

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
if not torch.backends.mps.is_available():
    print("\nMPS device not found.")
    mps_device = None

if torch.backends.mps.is_available():
    device = torch.device("mps")
    mps_device = torch.device("mps")
    x = torch.ones(1, device=device)
    print("\nCheck M1 chip:", x)
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = "cpu"
print("device selected:", device)


def partition(D, L):
    return torch.randperm(D).view(L, -1)


# D = 10
# d = int(D**2)  # d 100
# n_batch = 5
# N = n_batch
# C = math.log(d) / 2
# # print(C) #2.3 group cardinality

# q = math.log(d) / D
# print(q)  # 0.46
# L = int(D // C) + 1
# print(L)  # 5 L groups

# # feature and norm
# w = torch.randn(d)
# w /= w.square().sum().sqrt()
# print(w.shape)  # 100

# Custom num of patches D and of groups L
torch.manual_seed(0)
D, L = 16, 2
S = partition(D, L)
print(f"partition {S}")  # C=2; L = 5 # yes, random indeces in each group
# of cardinality C = D/L

# patches in groups; patches in same group don't have to be spatially localized
# contiguous. The Spatial transformer finds them. SO in some sense transformers
# perform non local means when they succeed at denoising these groups by
# provably showing recovered spatial structure via these groups.
#
# def make_data(N, D, L, d, S, w):
#     y = torch.randn(N).sign()
#     q = math.log(d) / D
#     sigma = 1 / math.sqrt(d)
#     noise = torch.randn(N, D, d) * sigma
#     X = torch.zeros(N, D, d)
#     # deltas = torch.rand(N, D) - q
#     # deltas = ((deltas.sgn() - 1).sgn() * (torch.rand(N, D) - 0.5).sgn()).sgn()
#     # X = X
#     for i in range(N):
#         l = torch.randint(L, (1,))[0]
#         R = S[l]
#         for j in range(D):
#             if j in R:
#                 X[i][j] = y[i] * w + noise[i][j]
#             else:
#                 prob = 2 * (torch.rand(1) - 0.5)
#                 if prob > 0 and prob < q / 2:
#                     delta = 1
#                 elif prob < 0 and prob > -q / 2:
#                     delta = -1
#                 else:
#                     delta = 0
#                 X[i][j] = delta * w + noise[i][j]
#     return X, y, w, S

N = 3  # num of records/instances
d = 16 * 16  # let's make the pathces 3x3

# make feature and normalize
w = torch.randn(d)
w /= w.square().sum().sqrt()
print(f"w shape {w.shape} and values {w}")
# w shape torch.Size([9]) and values tensor([ 0.2479,  0.0915,  0.1117,  0.2081,  0.1429,  0.6163,  0.4107,  0.5519,
#         -0.0307])


# partition tensor([[0, 1],
#         [3, 2]])
# y tensor([ 1.,  1., -1., -1.,  1.])
# X shape torch.Size([5, 4, 9])
def make_data(N, D, L, d, S, w):
    """D is the num of patches and each patch is dim d; S is the partition of L
    groups.
    """
    y = torch.randn(N).sign()  # this is the label 1, -1
    print(f"y {y}")
    q = math.log(d) / D
    print(f"the cutoff q/2 on prob to set delta multiplier to 1, 0, or -1 {q/2}")

    sigma = 1 / math.sqrt(d)  # this is the std dev for the noise
    noise = torch.randn(N, D, d) * sigma  # noise eps is same shape with X page4
    X = torch.zeros(N, D, d)
    print(f"X shape {X.shape}")
    # deltas = torch.rand(N, D) - q
    # # deltas = ((deltas.sgn() - 1).sgn() * (torch.rand(N, D) - 0.5).sgn()).sgn()
    # # X = X
    # At this point X is only 0
    for i in range(N):  # for each instance in batch or dataset
        l = torch.randint(L, (1,))[0]  # pick a group; in this toy set we have 2

        # so we only pick one group per instance
        # and the indices in that group will get the signal set tied to the
        # assigned label

        # each instance has the same group structure given by partition S
        # but the group with the signal set will vary randomly per instance.

        # groups only to pick from, so l can be either 0 or 1
        print(f"pick group l = {l}")
        R = S[l]
        print(f"we picked group l {l} and the indices in it are R {R}")
        # we have fewer groups than instance so each group can be picked
        # more than once

        for j in range(D):  # for each patch in instance (each instance has D patches)
            # note that the S partition is over the D patches
            if j in R:  # if this patch index is in the picked group
                X[i][j] = (
                    y[i] * w + noise[i][j]
                )  # we have a label assigned to this instance
                # and we create the signal using that label with the feature w
                # w is the same accross the dataset; for indices in the group.
                # so there will possibly be multiple instances i in dataset
                # where the same group was picked and might have the same label y
                # so the same signal will exist.

            else:
                prob = 2 * (torch.rand(1) - 0.5)
                print(
                    f"for the instance {i} and patch {j} which is not in picked group {R} we will use a probability to either set Xij to a +w feature or -wfeature or pure noise irrespective of the label y value - so random set {prob}"
                )
                print(f"delta 1 for prob in 0 to {q/2} and -1 for <0 to {-q/2} rest 0")
                if prob > 0 and prob < q / 2:
                    delta = 1
                elif prob < 0 and prob > -q / 2:
                    delta = -1
                else:
                    delta = 0
                print(f"delta here is {delta}")  # there are a lot more 0 than 1 or
                # -1 in which case the featuer is just noise.
                # self-q is the problem easier or harder with lots of noise?
                X[i][j] = delta * w + noise[i][j]
    return X, y, w, S


# Note that w is the same accross the dataset X.
# So for instances this w will be in one of the sets, but the set/group in which will
# be randomly picked.
# the spatial transformer should learn to do patch association meaning for each
# instance find the topk by cos sim of each patch's estimated v with w and the
# highest should be for the patches in the group with the signal set.

X, y, w, S = make_data(N, D, L, d, S, w)


plt.figure(figsize=(4, 4))
S_matrix = torch.eye(D, D)
for x in S:
    print(f"x in S is {x}")  # this is a group in this case of 2 indices
    # so that it can be plotted in 2D; in 3D they'd be cubes in space
    S_matrix[x[0], x[1]] = 1 / 2
    S_matrix[x[1], x[0]] = 1 / 2
# print(f"S_matrix {S_matrix}")
plt.matshow(S_matrix, cmap="Greys")
plt.axis("off")
plt.savefig("test_partition.png")


# torch.save(X, "data/X.to")
# torch.save(y, "data/y.to")
# torch.save(w, "data/w.to")
# torch.save(S, "data/S.to")


# X = torch.load("data/X.to").requires_grad_(True)
# y = torch.load("data/y.to").long()
# w = torch.load("data/w.to")
# S = torch.load("data/S.to")

X = X.to(device)
y = y.to(device)
w = w.to(device)
# print(X.shape) #torch.Size([50, 10, 100])

plt.figure(figsize=(4, 4))
S_matrix = torch.eye(D, D)
for x in S:
    S_matrix[x[0], x[1]] = 1 / 2
    S_matrix[x[1], x[0]] = 1 / 2
plt.matshow(S_matrix, cmap="Greys")
plt.axis("off")
plt.savefig("test_partition.png")


alpha = 0.03
p = 5


# Now the model - with special Positional Attention Mechanism
# sigma is the activation
class Sigma(nn.Module):
    def __init__(self, alpha, p):
        super(Sigma, self).__init__()
        self.alpha = alpha
        self.p = p

    def forward(self, H):
        sig_term1 = H**self.p
        print(f"sig term 1 shape {sig_term1.shape} and val {sig_term1}")
        sig_term2 = self.alpha * H
        print(f"sig term 2 shape {sig_term2.shape} and val {sig_term2}")
        # without sum [5, 10] which is (B, D)
        return sig_term1.sum(-1) + sig_term2.sum(-1)  # this is classif problem


# for denoise task
class SigmaN(nn.Module):
    def __init__(self, alpha, p):
        super(SigmaN, self).__init__()
        self.alpha = alpha
        self.p = p

    def forward(self, H):
        sig_term1 = H**self.p
        print(f"sig term 1 shape {sig_term1.shape} and val {sig_term1}")
        sig_term2 = self.alpha * H
        print(f"sig term 2 shape {sig_term2.shape} and val {sig_term2}")
        # without sum [5, 10] which is (B, D)
        return sig_term1 + sig_term2  # this is classif problem


sigmaN = SigmaN(alpha, p)
# sigma = Sigma(alpha, p)
# print(sigma)
sigma_Q = math.log(math.log(d))

# init
# multiva norm with some small mean
# Q_0 = torch.eye(D) * sigma_Q + torch.randn(D, D) * 0.001
# v_0 = torch.randn(w.shape) * 0.001

# make these learn
# Q = torch.nn.Parameter(Q_0)

# v = torch.nn.Parameter(v_0)


class Attention(nn.Module):
    def __init__(self, Q, v):
        super(Attention, self).__init__()
        self.Q = Q
        self.v = v
        self.sm = nn.Softmax(dim=-1)

    def forward(self, X):
        print(f"X shape {X.shape}")  # [5, 10, 100]
        Q = self.Q  # [10, 10]
        print(f"Q shape {Q.shape}")
        v = self.v  # d=100 shape
        attn = self.sm(Q)
        print(f"attn shape {attn.shape}")
        v_X = X @ v
        print(f"v_X.shape {v_X.shape}")
        return v_X.mm(attn.T)  # matmul of mxn and nxp


class SpatialTransformer(nn.Module):
    def __init__(self, alpha, p, sigma_Q, D, d):
        super(SpatialTransformer, self).__init__()

        Q_0 = torch.eye(D) * sigma_Q + torch.randn(D, D) * 0.001
        v_0 = torch.randn(d) * 0.001

        Q = torch.nn.Parameter(Q_0)
        v = torch.nn.Parameter(v_0)

        self.Q = Q
        self.v = v
        self.attention = Attention(Q, v)
        self.sigma = SigmaN(alpha, p)

    def forward(self, X):
        print(f"X shape in net {X.shape}, w shape {w.shape}, v shape {self.v.shape}")
        H = self.attention(X)
        print(f"shape of H {H.shape}")
        return self.sigma(H)


# then this is packaged in a Transformer class but basically this is it.
# net = Attention(Q, v)
# print(net)
# print(f"params {net.parameters()}")
# Then it gets put through sigma

# net = net.to(device)


# then get patch association explicit via cosine similarity
def cosine(a, b):
    return (torch.dot(a, b)) / torch.sqrt(torch.dot(a, a) * torch.dot(b, b))


# page 6
net1 = SpatialTransformer(alpha=0.03, p=5, sigma_Q=sigma_Q, D=D, d=d).to(device)
epochs = 1
optimizer = torch.optim.SGD(net1.parameters(), lr=1e-1)  # this is not minibatch level
# is full batch GD


for i in range(epochs):
    partial = net1(X.to(device))
    print(f"partial shape {partial.shape}")
    # torch.Size([5, 10]);  5 is batch; D = 10 (B, D)
    # Y = sigmaN(partial)
    # print(f"now shape Y is {Y.shape}")

    # they do CE with y
    # and cosine of trained v with generated w
    c = cosine(net1.v.data, w)
    print(f"cosine similarity bet learned feature v and gen w {c}")
