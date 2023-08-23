import numpy as np
import math


# Job parameters.
run_id = 'baseline_action'
arch = 'largei3d'
saved_model = None
restart = False

# Dataset parameters.
num_classes = 102
num_frames = 16
fix_skip = 2
num_modes = 5
num_skips = 1
data_percentage = 1.0

# Training parameters.
batch_size = 16
v_batch_size = 16
num_workers = 4
learning_rate = 1e-4
num_epochs = 100
loss = 'ce'
temporal_loss = None # 'trip'
warmup_array = list(np.linspace(0.01, 1, 10) + 1e-9)
warmup = len(warmup_array)
momentum = 0.9
weight_decay = 1e-4
lr_warmup_method = 'linear'
lr_warmup_epochs = 10
lr_warmup_decay = 0.001
lr_gamma = 0.1
lr_milestones = [20, 30, 40]
lr_patience = 2
lr_reduce_factor = 2
distributed = False
lr_scheduler = 'cosine' #'patience_based' #'loss_based' (default) #'cosine'
cosine_lr_array = list(np.linspace(0.01, 1, 5)) + [(math.cos(x) + 1)/2 for x in np.linspace(0, math.pi/0.99, num_epochs-5)]
val_freq = 3
opt_type = 'adam' # 'sgd' # 'adamw'
ft_dropout = 0
eval_only = False
val_array = [1] + [5*x for x in range(1, 8)] + [2*x for x in range(21, 25)]

temporal_loss_weight = 0.1
temporal_distance = None

# Validation augmentation params.
hflip = [0]
cropping_facs = [0.8]
cropping_factor = 0.8
weak_aug = False
no_ar_distortion = False
aspect_ratio_aug = False

# Training augmentation params.
reso_h = 224
reso_w = 224
ori_reso_h = 240
ori_reso_w = 320
min_crop_factor_training = 0.6
temporal_align = False

# Tracking params.
wandb = False
