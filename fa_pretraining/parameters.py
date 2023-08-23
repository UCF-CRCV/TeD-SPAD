import numpy as np


# Dataset parameters.
fix_skip = 2
num_modes = 5
num_skips = 1

# Training parameters.
batch_size = 32
num_workers = 4
learning_rate = 1e-3
num_epochs = 100
warmup_array = list(np.linspace(0.01, 1, 5) + 1e-9)
warmup = len(warmup_array)
scheduled_drop = 5
lr_patience = 0

# Validation augmentation params.
hflip = [0]
cropping_fac1 = [0.8]

# Training augmentation params.
reso_h = 224
reso_w = 224
