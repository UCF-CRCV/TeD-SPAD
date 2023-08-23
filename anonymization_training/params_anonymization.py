import numpy as np
import math
import os.path


# Job parameters.
run_id = 'baseline_anonymization'
arch_ft = 'largei3d'
arch_fa = 'unet++'
arch_fb = 'r50'
saved_model_fa = os.path.join('..', 'saved_models', 'fa_recon.pth')
saved_model_ft = None
saved_model_fb = os.path.join('..', 'saved_models', 'fb_ssl.pth')

# Dataset parameters.
num_classes = 102
num_frames = 16
fix_skip = 2
num_modes = 5
num_skips = 1
data_percentage = 0.1

# Number of VISPR privacy attributes.
num_pa = 7
data_percentage_vispr = 1.0

# Training parameters.
batch_size = 4
batch_size_vispr = 4
v_batch_size = 4
num_workers = 4
learning_rate = 1e-4
num_epochs = 100
loss = 'ce'
temporal_loss = 'trip'
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
lr_scheduler = 'cosine' # 'patience_based' #'loss_based' (default) #'cosine'
cosine_lr_array = list(np.linspace(0.01,1, 5)) + [(math.cos(x) + 1)/2 for x in np.linspace(0,math.pi/0.99, num_epochs-5)]
val_freq = 3
opt_type = 'adam' # 'sgd' # 'adamw'
ft_dropout = 0

# Anonymization training parameters.
# Scaled lr per model.
learning_rate_fa = 0.4*learning_rate
learning_rate_fb = 1.0*learning_rate
learning_rate_ft = 1.0*learning_rate
ft_loss_weight = 0.7
fb_loss_weight = 1.0
temporal_loss_weight = 0.1
weight_inv = 0.0
triplet_loss_margin = 1
temporal_distance = 4

# Validation augmentation params.
hflip = [0]
cropping_facs = [0.8]
weak_aug = False
no_ar_distortion = False
aspect_ratio_aug = False

# Training augmentation params.
reso_h = 224
reso_w = 224
min_crop_factor_training = 0.6
temporal_align = False

# Tracking params.
wandb = False
