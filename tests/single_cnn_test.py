import numpy as np
import torch
import torch.nn as nn

import sparseops_backend

BATCH_DIM   = 8
IN_CHANNELS = 9
CONV_OUT    = 5
IMG_SIZE    = 32
KERNEL_SIZE = 3
PADDING     = 1
STRIDE      = 1
BIAS        = False
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---- Torch reference
conv = nn.Conv2d(IN_CHANNELS, CONV_OUT, kernel_size=KERNEL_SIZE, padding=PADDING, stride=STRIDE, bias=BIAS)
input = torch.randn(BATCH_DIM, IN_CHANNELS, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
with torch.no_grad():
    output_ref = conv(input)
print("Output ref shape:", tuple(output_ref.shape))

# ---- NumPy views
conv_weight_np = conv.weight.detach().cpu().numpy().astype(np.float32)    # (Cout,Cin,kH,kW)
input_np       = input.detach().cpu().numpy().astype(np.float32)          # (B,Cin,H,W)

# Fortran input: batch-fast
input_f = np.asfortranarray(input_np)  # (B,Cin,H,W)_F

plan   = sparseops_backend.setup_conv(conv_weight_np, stride=STRIDE, padding=PADDING)
output = sparseops_backend.conv2d_ellpack(plan, input_f)  # (B,Cout,Hout,Wout)_F


print(f"Output ref shape {output_ref.shape}")
print(f"Output shape {output.shape}")
# print("="*50)
# print(output_ref)
# print("="*50)
# print(output)

print("Torch vs Conv:   ", np.allclose(output_ref.detach().cpu(), output, atol=1e-4))

