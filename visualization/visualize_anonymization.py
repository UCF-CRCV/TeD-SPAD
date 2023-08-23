# This code is to visualize the videos after anonymization model.
import imageio
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Decord after torch.
import decord
decord.bridge.set_bridge('torch')

import sys
sys.path.insert(0, '..')
from aux_code import config as cfg
from aux_code.model_loaders import load_fa_model


if __name__ == '__main__':
    arch = 'unet++'
    model_path = os.path.join('..', 'saved_models', 'model_20_bestAcc_0.7504.pth')
    vid_path = os.path.join(cfg.ucf_crimes_path, 'Videos', 'Shoplifting', 'Shoplifting033_x264.mp4')
    use_cuda = torch.cuda.is_available()

    # Load in model to visualize.
    fa_model = load_fa_model(arch=arch, saved_model_file=model_path)
    fa_model.eval()

    vr = decord.VideoReader(vid_path, ctx=decord.cpu())
    frame_count = len(vr)

    # Load in video.
    frames = np.arange(670, 702, 2) # Hard coded to seek out interesting frames.
    frames = vr.get_batch(frames).permute(0, 3, 1, 2)

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    inputs = []
    for frame in frames:
        inputs.append(trans(frame))

    inputs = torch.stack(inputs, dim=0)

    if use_cuda:
        fa_model.cuda()
        inputs = inputs.cuda()

    with torch.no_grad():
        output = fa_model(inputs)
        vis_path = os.path.join('visualization', os.path.basename(model_path))
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

        anon_images = np.concatenate(torch.flip(output[::2], dims=[1]).cpu().numpy().transpose(0, 2, 3, 1), axis=1)
        input_images = np.concatenate(inputs[::2].cpu().numpy().transpose(0, 2, 3, 1), axis=1)

        imageio.imwrite(os.path.join(vis_path, f'raw_{os.path.basename(vid_path)}.png'), input_images)
        imageio.imwrite(os.path.join(vis_path, f'anon_{os.path.basename(vid_path)}.png'), anon_images)
