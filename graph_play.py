"""With the graph geometric"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

# I have several features for each node so each node is a matrix


x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)


# then edges (source and destination nodes)

edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
# note that edge_index is of shape [2, num_edges] and also
# that edges are directed, so (0,1) is different from (1,0)

data = Data(x=x, edge_index=edge_index)
print(data)

# an example: we can convert the point cloud dataset into a graph
# dataset by generating nearest neighbor graph from the point clouds
# via transforms

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(
    root="/tmp/ShapeNet", categories=["Airplane"], pre_transform=T.KNNGraph(k=6)
)  # k could be random

print(dataset[0])

# =T.RandomJitter(0.01) translates each node position by a small
# number

# can't get the dataset though

# ********************Also faiss; likely later not for the step now.**********************
import faiss
import numpy as np

# from train_denoiser import args

# POC TODO@DR: alternative to faiss since issue with python3.12
# for now run with  python3.9 environment

# each row is a patch vector
image_patches = np.random.rand(32, 256).astype(
    "float32"
)  # 10,000 patches of size 16x16 flattened

patch_dim = image_patches.shape[1]
num_neighbors = 4  # number of nearest neighbors to search for

index = faiss.IndexFlatL2(patch_dim)  # L2 distance index
index.add(image_patches)  # add patches to the index

query_patch = np.random.rand(1, 256).astype("float32")  # single query patch

# for each pixel, query the index to find its k-nearest neighbors
# query patch represents the patch centered on the pixel to be denoised

distances, indices = index.search(
    query_patch, num_neighbors
)  # search for nearest neighbors
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)
