import torch
import torch.nn as nn

import math


# now the sequence has double width due to concat clean and noisy
class PatchEmbedding(nn.Module):
    def __init__(self, dim_in, d_model):
        super().__init__()

        # aim to embed the clean and noisy together for dim reduction
        # alternatively I could add the position embed directly in the
        # 200 dim

        # set bias to zero
        self.projection = torch.nn.Linear(dim_in, d_model, bias=False)

    def forward(self, x):
        # print(f"layer in patch embed {self.projection}")
        # print(f"debug patch embed shapes {x.shape}")
        x = self.projection(x)
        # (batch_size, embed_dim, num_patches)
        return x


patches = torch.randn(4, 8, 128)  # B, seq_len, fused patch dim
pemb = PatchEmbedding(
    128, 32
)  # first dim should be the patch size and second the emb dim

print(f"pemb layer {pemb}")

embedded = pemb(patches)

print(
    f"embedded seq shape {embedded.shape}"
)  # [4, 8, 32]) as expected (B, seq_len, d_model=embeddim)


########################Add Pos Embed
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        # TODO@DR check for correctness again
        pe = torch.zeros(max_len, d_model)  # (seq_len, d_model)
        # print(f"pos emb starts as {pe.shape}")

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # print(f"position is {position} of shape {position.shape}")
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # double the d_model size
        # print(f"div is {div_term} of shape {div_term.shape}")
        # print("torch.sin(position * div_term)", torch.sin(position * div_term).shape)
        # print("pe[:, 0::2]", pe[:, 0::2].shape, pe[:, 0::2])
        # print("pe[:, 1::2]", pe[:, 1::2].shape)

        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # start index at 0 with step size 2
        # print("pe[:, 0::2] and pe", pe[:, 0::2].shape, pe)
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # start index at 1 with step size 2

        # print("test that the two pe with sin and cos are not eq", pe[:, 0::2] == pe[:, 1::2]) # ok
        self.register_buffer("pe", pe.unsqueeze(0))  # Add a batch dimension

    def forward(self, x):
        # x is the sequence of patch embeddings, shape (batch_size, seq_len, d_model)
        # We need to add positional embeddings to each element in the sequence
        # for the moment only create the pos embeddings

        # print(self.pe.shape)
        # print(self.pe[:, : x.size(1)].shape)

        return self.pe[:, : x.size(1)]


# putting an index on the full sequence including the query last token; so max_len in dev is 8
pos_embed_layer = SinusoidalPositionalEmbedding(d_model=32, max_len=8)

print(f"pos embed layer {pos_embed_layer}")

# since the position index must match the embedded patches, apply pos embed directly on the
# embedded patches
pos_embed = pos_embed_layer(embedded)

print(f"pos embeddings shape {pos_embed.shape}")  # (B, seq_len, d_model)
# so shape is [1, 8, 32]) - it gets broadcasted accross the batch
# it has the same dim embedding= d_model as the patch embedding
# looks ok

total_embed = pos_embed + embedded
print(total_embed.shape)
# TODO@DR
# so if I subselect patches randomly for in-context task - I must first add the pos before the
# # subselection

# print(torch.arange(0, 8, 2).float()) # (0, 2, 4, 6)
# print((-math.log(10000.0) / 8))
# print(torch.arange(0, 8, 2).float() * (-math.log(10000.0) / 8))
# print(torch.exp(torch.arange(0, 8, 2).float() * (-math.log(10000.0) / 8)))

for name, param in pemb.named_parameters():
    print(f"param is {param} and name is {name} ")

# no param in position embeddings - OK
