import faiss
import numpy as np

# from train_denoiser import args

# POC TODO@DR: alternative to faiss since issue with python3.12
# for now run with nlmeans python3.9 environment

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
