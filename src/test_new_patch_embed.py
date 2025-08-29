"""
about patches in timm
"""

import timm
import torch
from timm.models.vision_transformer import PatchEmbed

model = timm.create_model("vit_base_patch16_224", pretrained=True)
patch_embedding_layer = model.patch_embed
print(patch_embedding_layer)
# PatchEmbed(
#   (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
#   (norm): Identity()
# )

# Assuming 'img_tensor' is a preprocessed image tensor (e.g., [1, 3, 224, 224])
# You would typically get this by applying transforms to an image
dummy_input = torch.randn(1, 3, 224, 224)  # Example dummy input
patches = model.patch_embed(dummy_input)
print("Shape after patch embedding:", patches.shape)  # torch.Size([1, 196, 768])

# ---------------my usecase
dummy_input = torch.randn(1, 256)  # 1 instance of d=1024
x = dummy_input.view((1, 1, 16, 16))  # gray scale so 1 input channel
patche_embedder = PatchEmbed(16, 2, 1, 32, bias=True)
x_patched = patche_embedder(x)

print(
    f"shape of x_patched {x_patched.shape}"
)  # shape of x_patched torch.Size([1, 256, 32])
# batch = 1, num patches is 256; embedding dim = d_model = 32
