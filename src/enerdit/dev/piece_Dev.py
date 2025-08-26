import torch
import torch.nn as nn


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
