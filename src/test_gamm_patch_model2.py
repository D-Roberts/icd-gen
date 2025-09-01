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


D = 10
d = int(D**2)  # d 100
n_batch = 5
N = n_batch
C = math.log(d) / 2
# print(C) #2.3 group cardinality

q = math.log(d) / D
print(q)  # 0.46
L = int(D // C) + 1
print(L)  # 5 L groups

# feature and norm
w = torch.randn(d)
w /= w.square().sum().sqrt()
print(w.shape)  # 100
S = partition(D, L)
print(S)  # C=2; L = 5


def make_data(N, D, L, d, S, w):
    y = torch.randn(N).sign()
    q = math.log(d) / D
    sigma = 1 / math.sqrt(d)
    noise = torch.randn(N, D, d) * sigma
    X = torch.zeros(N, D, d)
    # deltas = torch.rand(N, D) - q
    # deltas = ((deltas.sgn() - 1).sgn() * (torch.rand(N, D) - 0.5).sgn()).sgn()
    # X = X
    for i in range(N):
        l = torch.randint(L, (1,))[0]
        R = S[l]
        for j in range(D):
            if j in R:
                X[i][j] = y[i] * w + noise[i][j]
            else:
                prob = 2 * (torch.rand(1) - 0.5)
                if prob > 0 and prob < q / 2:
                    delta = 1
                elif prob < 0 and prob > -q / 2:
                    delta = -1
                else:
                    delta = 0
                X[i][j] = delta * w + noise[i][j]
    return X, y, w, S


X, y, w, S = make_data(N, D, L, d, S, w)
torch.save(X, "data/X.to")
torch.save(y, "data/y.to")
torch.save(w, "data/w.to")
torch.save(S, "data/S.to")


X = torch.load("data/X.to").requires_grad_(True)
y = torch.load("data/y.to").long()
w = torch.load("data/w.to")
S = torch.load("data/S.to")

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
sigma = Sigma(alpha, p)
print(sigma)
sigma_Q = math.log(math.log(d))

# init
# multiva norm with some small mean
Q_0 = torch.eye(D) * sigma_Q + torch.randn(D, D) * 0.001
v_0 = torch.randn(w.shape) * 0.001

# make these learn
Q = torch.nn.Parameter(Q_0)

v = torch.nn.Parameter(v_0)


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
        v = self.v
        attn = self.sm(Q)
        print(f"attn shape {attn.shape}")
        v_X = X @ v
        return v_X.mm(attn.T)  # matmul of mxn and nxp


# then this is packaged in a Transformer class but basically this is it.
net = Attention(Q, v)
print(net)
# print(f"params {net.parameters()}")
# Then it gets put through sigma

net = net.to(device)


# then get patch association explicit via cosine similarity
def cosine(a, b):
    return (torch.dot(a, b)) / torch.sqrt(torch.dot(a, a) * torch.dot(b, b))


epochs = 1
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)  # this is not minibatch level
# is full batch GD

sigma_Q = math.log(math.log(d))


class SpatialTransformer(nn.Module):
    def __init__(self, alpha, p, sigma_Q, D, d):
        super(SpatialTransformer, self).__init__()

        Q_0 = torch.eye(D) * sigma_Q + torch.randn(D, D) * 0.001
        v_0 = torch.randn(w.shape) * 0.001

        Q = torch.nn.Parameter(Q_0)
        v = torch.nn.Parameter(v_0)

        self.Q = Q
        self.v = v
        self.attention = Attention(Q, v)
        self.sigma = SigmaN(alpha, p)

    def forward(self, X):
        H = self.attention(X)
        return self.sigma(H)


# page 6
net1 = SpatialTransformer(alpha=0.03, p=5, sigma_Q=sigma_Q, D=10, d=100).to(device)

for i in range(epochs):
    partial = net1(X.to(device))
    print(f"partial shape {partial.shape}")  # torch.Size([5, 10]);  5 is batch; D = 10
    # Y = sigmaN(partial)
    # print(f"now shape Y is {Y.shape}")

    # they do CE with y
    # and cosine of trained v with generated w
    c = cosine(v.data, w)
    print(f"cosine similarity bet learned feature v and gen w {c}")
