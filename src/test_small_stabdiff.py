from diffusers import DiffusionPipeline, DiTTransformer2DModel, DiTPipeline
from diffusers.utils import load_image
from transformers import AutoConfig, AutoModel
import torch

# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16)
# pipe.to("mps")


# # # get image
# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/tarsila_do_amaral.png"
# image = load_image(url)

# # # run image variation
# image = pipe(image).images[0]
# print(image)

# -----------
# DiT

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
print(pipe)

# model = DiTTransformer2DModel.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
# model = DiTTransformer2DModel.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
pipe.model = DiTTransformer2DModel.from_config(pipe.transformer.config)
# config = AutoConfig.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
print(pipe.model)


# DiTTransformer2DModel(
#   (pos_embed): PatchEmbed(
#     (proj): Conv2d(4, 1152, kernel_size=(2, 2), stride=(2, 2))
#   )
#   (transformer_blocks): ModuleList(
#     (0-27): 28 x BasicTransformerBlock(
#       (norm1): AdaLayerNormZero(
#         (emb): CombinedTimestepLabelEmbeddings(
#           (time_proj): Timesteps()
#           (timestep_embedder): TimestepEmbedding(
#             (linear_1): Linear(in_features=256, out_features=1152, bias=True)
#             (act): SiLU()
#             (linear_2): Linear(in_features=1152, out_features=1152, bias=True)
#           )
#           (class_embedder): LabelEmbedding(
#             (embedding_table): Embedding(1001, 1152)
#           )
#         )
#         (silu): SiLU()
#         (linear): Linear(in_features=1152, out_features=6912, bias=True)
#         (norm): LayerNorm((1152,), eps=1e-06, elementwise_affine=False)
#       )
#       (attn1): Attention(
#         (to_q): Linear(in_features=1152, out_features=1152, bias=True)
#         (to_k): Linear(in_features=1152, out_features=1152, bias=True)
#         (to_v): Linear(in_features=1152, out_features=1152, bias=True)
#         (to_out): ModuleList(
#           (0): Linear(in_features=1152, out_features=1152, bias=True)
#           (1): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (norm3): LayerNorm((1152,), eps=1e-05, elementwise_affine=False)
#       (ff): FeedForward(
#         (net): ModuleList(
#           (0): GELU(
#             (proj): Linear(in_features=1152, out_features=4608, bias=True)
#           )
#           (1): Dropout(p=0.0, inplace=False)
#           (2): Linear(in_features=4608, out_features=1152, bias=True)
#         )
#       )
#     )
#   )
#   (norm_out): LayerNorm((1152,), eps=1e-06, elementwise_affine=False)
#   (proj_out_1): Linear(in_features=1152, out_features=2304, bias=True)
#   (proj_out_2): Linear(in_features=1152, out_features=32, bias=True)
