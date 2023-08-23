import numpy as np
import os.path
import torch
from torch.utils.data import DataLoader

import params_feature_ex as params
from shanghai_dl import shanghai_frames_dataset

import sys
sys.path.insert(0, '..')
import aux_code.config as cfg
from aux_code.model_loaders import load_fa_model, load_ft_model


# Run clips through model.
def extract_features(full_vid, vid_features, save_path, fa_model, ft_model, anonymized, segment=False):
    for idx, inputs in enumerate(full_vid):
        inputs = inputs.cuda()
        inputs = inputs.unsqueeze(dim=0)

        with torch.no_grad():
            if anonymized:
                # Reshape inputs for video.
                ori_bs, ori_t, ori_c, ori_h, ori_w = inputs.permute(0, 2, 1, 3, 4).shape
                inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])
                inputs = fa_model(inputs).reshape(ori_bs, ori_t, ori_c, ori_h, ori_w)
            try:
                output = ft_model.extract_features(inputs)
            except:
                output = ft_model.i3d.extract_features(inputs)
            vid_features[idx] = output.squeeze().cpu().numpy()

    if segment:
        segmented_features = segment_features(vid_features)
        np.save(save_path, segmented_features)
    else:
        np.save(save_path, vid_features)

# Concatenate features into non-overlapping segments, as in Sultani et al.
def segment_features(vid_features):
    segmented_features = np.zeros((32, 1024))
    segment_loc = np.linspace(0, vid_features.shape[0], 33, dtype=int)
    for idx in range(len(segment_loc) - 1):
        ss, es = segment_loc[idx], segment_loc[idx+1] - 1
        # Go to end of video.
        if idx == 31:
            es += 1
        if ss <= es:
            temp_vect = vid_features[ss][:]
        else:
            temp_vect = np.mean(vid_features[ss:es][:])

        temp_vect = temp_vect/np.linalg.norm(temp_vect)
        segmented_features[idx] = temp_vect
    return segmented_features


if __name__ == '__main__':
    anonymized = True
    # Load in reconstruction F_a model, loaded ft_model.
    saved_fa_model = os.path.join('..', 'saved_models', 'model_20_bestAcc_0.7504.pth')
    saved_ft_model = os.path.join('..', 'saved_models', 'model_20_bestAcc_0.7504.pth')
    save_features_folder = 'st_features_ours'
    print(f'Fa_model: {saved_fa_model}', flush=True)
    print(f'Ft_model: {saved_ft_model}', flush=True)
    print(f'Features folder: {save_features_folder}', flush=True)

    # Create features folder.
    if not os.path.exists(save_features_folder):
        os.makedirs(save_features_folder)

    fa_model = load_fa_model(arch='unet++', saved_model_file=saved_fa_model) if anonymized else None
    ft_model = load_ft_model(arch='largei3d', kin_pretrained=True, saved_model_file=saved_ft_model, num_classes=params.num_classes)
    
    if torch.cuda.is_available():
        if anonymized:
            fa_model.cuda()
        ft_model.cuda()
    
    if anonymized:
        fa_model.eval()
    ft_model.eval()
    all_dataset = shanghai_frames_dataset(reverse=False)
    all_dataloader = DataLoader(all_dataset, batch_size=1, shuffle=False, num_workers=params.num_workers)
    num_features = 2048

    for i, (full_vid, _, vid_path) in enumerate(all_dataset):
        save_path = os.path.join(save_features_folder, os.path.basename(vid_path).replace('.avi', '') + '.npy')
        if os.path.exists(save_path):
            continue
        print(f'Extracting features for {os.path.basename(vid_path)}.', flush=True)

        try:
            vid_features = np.zeros((len(full_vid), num_features))
        except:
            print(f'Video {os.path.basename(vid_path)} could not process.')
            # print(os.path.basename(vid_path))
            continue

        extract_features(full_vid, vid_features, save_path, fa_model, ft_model, anonymized, False)
