''' likely python code
likely train on greene cluster
is this not available for me to work direclty with the GPUs only with slurm batch jobs submission?
was able to get around that at GTech wonder if can do so here I like to work with my own GPU when (I) develop my models.
https://sites.google.com/nyu.edu/NYU-hpc/hpc-systems/greene/getting-started#h.ywxg9alzindt - I think I can
they have this singularity image builtup with an old torch installation that doesn't allow for newer transformesr etc so
likely needU to build my own images and stuff; phasing out legacy transformers and packages; no other; trying to phase out
'''

"TODO: st**ting withcode"
import sys
import os
import torch
import diffusers
from diffusers import DiffusionPipeline
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns

# torch modules; do these stesp manually so that I recall how since its been so long
# some of this https://github.com/ml-jku/hopfield-layers/blob/master/examples/mnist_bags/mnist_bags_demo.ipynb
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential, Sigmoid
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
# lihgt weight early recalling of DL codes

x = torch.rand(5, 3)
print(x)
print(torch.__version__)

"""
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipe = pipe.to("mps")
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars"
im = pipe(prompt).images[0]
im.show()
# works but very slow on my meager machine; I still don't see exactly how this is going to since this stable diffusion model is huge and my machine is so meager'
"""
#########################################Lets try sth else now.
sns.set() #this is probably not relevant outside ipyn/colabs
sys.path.append(r'./AttentionDeepMIL')

