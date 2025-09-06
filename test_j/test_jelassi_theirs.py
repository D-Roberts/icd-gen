import torch
import torch.nn as nn
import math


def partition(D, L):
    return torch.randperm(D).view(L, -1)


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


class Sigma(nn.Module):
    def __init__(self, alpha, p):
        super(Sigma, self).__init__()
        self.alpha = alpha
        self.p = p

    def forward(self, H):
        return (H**self.p).sum(-1) + (self.alpha * H).sum(-1)


class Attention(nn.Module):
    def __init__(self, Q, v):
        super(Attention, self).__init__()
        self.Q = Q
        self.v = v
        self.sm = nn.Softmax(dim=-1)

    def forward(self, X):
        Q = self.Q
        v = self.v
        attn = self.sm(Q)
        v_X = X @ v
        return v_X.mm(attn.T)


class Transformer(nn.Module):
    def __init__(self, Q, v, sigma):
        super(Transformer, self).__init__()
        self.Q = Q
        self.v = v
        self.attention = Attention(Q, v)
        self.sigma = sigma

    def forward(self, X):
        H = self.attention(X)
        return self.sigma(H)


##################Just for testing#####################
device = "mps"


D = 10
d = int(D**2)
n_batch = 50000  # important number of samples, this will be slow
N = n_batch
C = math.log(d) / 2
q = math.log(d) / D
L = int(D // C) + 1
w = torch.randn(d)
w /= w.square().sum().sqrt()
S = partition(D, L)
N_test = 100

X, y, w, S = make_data(N, D, L, d, S, w)
torch.save(X, "data/X.to")
torch.save(y, "data/y.to")
torch.save(w, "data/w.to")
torch.save(S, "data/S.to")


X = torch.load("data/X.to").requires_grad_(True)
y = torch.load("data/y.to").long()
w = torch.load("data/w.to")
S = torch.load("data/S.to")

X_test, y_test, _, _ = make_data(N_test, D, L, d, S, w)
X = X.to(device)
y = y.to(device)
w = w.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)


alpha = 0.03
p = 5

sigma = Sigma(alpha, p)

sigma_Q = math.log(math.log(d))

Q_0 = torch.eye(D) * sigma_Q + torch.randn(D, D) * 0.001
v_0 = torch.randn(w.shape) * 0.001

Q = torch.nn.Parameter(Q_0)

v = torch.nn.Parameter(v_0)

net = Attention(Q, v)
# net = net.to(device)
coss = []
losss = []
test_error = []

# what about net1
tr = Transformer(Q, v, sigma)
net1 = tr.to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
n_epochs = 300


def cosine(a, b):
    return (torch.dot(a, b)) / torch.sqrt(torch.dot(a, a) * torch.dot(b, b))


for i in range(n_epochs):
    # Y_test = sigma(net(X_test))
    Y_test = net1(X_test)
    test_error.append(((torch.sgn(Y_test) - y_test).abs().sum() / (2 * N_test)) * 100)
    optimizer.zero_grad()
    print(Y_test.shape)
    # Y = sigma(net(X))
    Y = net1(X)

    print(f"Y shape: {Y.shape}")

    loss = torch.log(1 + torch.exp(-Y * y)).mean()
    coss.append(cosine(v.data, w))
    losss.append(loss.data)
    loss.backward()
    optimizer.step()
    Q_store = Q.clone().data
    for j in range(Q.shape[0]):
        Q_store[j, j] = sigma_Q
    Q.data = Q_store
    print(f"loss {loss}")
    # print(Q.grad) # yes it is learning; the classification;; so Y is scalar.
