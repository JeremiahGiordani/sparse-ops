import numpy as np
print("LOADING data/layer1.npz")
x = np.load("data/layer1.npz")
print("bias.shape:", x["bias"].shape)
print("indptr.shape:", x["indptr"].shape)