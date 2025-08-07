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
