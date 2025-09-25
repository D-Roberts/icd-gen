"""
about patches in timm
"""

import timm
import torch
from timm.models.vision_transformer import Patchencode

model = timm.create_model("vit_base_patch16_224", pretrained=True)
patch_encodeding_layer = model.patch_encode
print(patch_encodeding_layer)
# Patchencode(
#   (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
#   (norm): Identity()
# )

# Assuming 'img_tensor' is a preprocessed image tensor (e.g., [1, 3, 224, 224])
# You would typically get this by applying transforms to an image
dummy_input = torch.randn(1, 3, 224, 224)  # Example dummy input
patches = model.patch_encode(dummy_input)
print("Shape after patch encodeding:", patches.shape)  # torch.Size([1, 196, 768])

# ---------------my usecase
dummy_input = torch.randn(1, 256)  # 1 instance of d=1024
x = dummy_input.view((1, 1, 16, 16))  # gray scale so 1 input channel
patche_encodeder = Patchencode(16, 2, 1, 32, bias=True)
x_patched = patche_encodeder(x)

print(
    f"shape of x_patched {x_patched.shape}"
)  # shape of x_patched torch.Size([1, 256, 32])
# batch = 1, num patches is 256; encodeding dim = d_model = 32

# _---------

test_broadc = torch.zeros(3, 2)
a = torch.ones(1, 2)
# print(test_broadc)
# print(a)
# print(test_broadc + a)  # yes it does as expected


# --------------------
def sample_mixture(d=2, n=5, seeds=None):
    """the simplest sampling syntethic for enerdit learning debug.
    d is dimension of patch flattened say 32x32 images

    100,000 samples

    put no context prompt on
    From mixture N(0, 16Id) so var = 16, stdev = 4 and stdev=1
    """
    # torch.manual_seed(seeds)
    sig1, sig2 = 4, 1

    X = torch.zeros((n, d))
    print(
        f"X shape {X.shape} , X[i] shape {X[1, :].shape} and randn(1,d) shape {(torch.randn(1, d)).shape}"
    )
    for i in range(n):
        tensor_uniform = torch.rand(1)
        print(tensor_uniform)

        if tensor_uniform < 0.5:
            print(f"sig1?")
            X[[i], :] += sig1 * torch.randn(1, d)
        else:
            X[[i], :] += sig2 * torch.randn(1, d)
            print(f"sig2?")
    return X


res = sample_mixture()

print(res)
