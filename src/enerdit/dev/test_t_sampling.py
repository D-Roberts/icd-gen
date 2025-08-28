""""
About time.

"""
import numpy as np
import torch
from torch.distributions import Uniform

tmin = torch.tensor(0.001)
tmax = torch.tensor(1000)

t_distrib = Uniform(low=torch.tensor([tmin]), high=torch.tensor([tmax]))

# sample one value at a time
d = torch.tensor(4096)

print(f"if d=4096 and tmax=1000 than max t/d is {1000/4096}")  # 0.244140625
print(f"so then if my d is 64, tmax might be {0.244140625*64}")  # 15.625
for i in range(10):
    t = t_distrib.sample()
    print(t)
    # if d = 4096 as for 64x64 images
    print(f"t/d if d=64 {t/d}")  # if t is tensor t/d is tensor as well; ok
    # vals like t/d tensor([0.1449]) if d =4096
    # vals like 10.5347 if d = 64 like I have now

##################### I think this below is the right t sampling but must
# must adjust tmax I think for my dimensions d

# if logt U(logtmin, logtm)
logtmin = torch.log(tmin)
logtmax = torch.log(tmax)

logt_distrib = Uniform(low=torch.tensor([logtmin]), high=torch.tensor([logtmax]))

for i in range(10):
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
print(z.shape)  # 4096
print(
    f"what is z/sqrt(d) like here sqrt(d) is 64 {z/torch.sqrt(d)}"
)  # -0.0067,  0.0107,  0.0043
znormsqd = torch.norm(z, p=2) ** 2
print(f"znormsqrd/d is then {znormsqd / d}")  # 1.013626217842102

#########################
# what kind of numbers are U
for px in np.arange(0.001, 0.999, 0.05):
    print(
        f"U would be {-torch.log(torch.tensor(px))} for prob {px} with log {torch.log(torch.tensor(px))}"
    )
    # right, U is basically negative log likelihood which we make neg log lik to be a sum and pos
    # to drive down to 0 by optimization


#########################Get a better sense of the needed time step embedding

x = torch.randn((5, 128)).unsqueeze(1)
print(x.shape)
y = torch.tile(x, (1, 8, 1))
print(y.shape)
print(y[:, 0, :] == y[:, 1, :])
