import glob as glob
import numpy as np
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import fn, types
import os.path
import torch
import torchvision.transforms.functional as F

import params_feature_ex as params

import sys
sys.path.insert(0, '..')
import aux_code.config as cfg
from aux_code.model_loaders import load_fa_model, load_ft_model


# Pytorch DataLoader.
class DALIDataloader(DALIGenericIterator):
    def __init__(self, pipeline, params, reader_name, batch_size, output_map=["data", "label"], auto_reset=False):
        self.params = params
        self.output_map = output_map
        super().__init__(pipelines=pipeline, reader_name=reader_name, auto_reset=auto_reset, output_map=output_map)
        self.iter_batch_size = batch_size
    
    def __len__(self):
        if self.size % self.iter_batch_size == 0:
            return self.size//self.iter_batch_size
        else:
            return self.size//self.iter_batch_size+1

    def __next__(self):
        data = super().__next__()[0]
        video, label = data[self.output_map[0]], data[self.output_map[1]] 
        video = self.val_augmentations(video)
        return video, label

    def val_augmentations(self, video):
        video = torch.transpose(video, 2, 4)
        video = torch.transpose(video, 3, 4)
        video = video / 255.
        self.ori_reso_w = int(video.shape[-1])
        self.ori_reso_h = int(video.shape[-2])
        self.min_size = min(self.ori_reso_h, self.ori_reso_w)
        if self.params.no_ar_distortion:
            video = F.center_crop(video.squeeze(), (int(self.min_size*self.params.cropping_factor),int(self.min_size*self.params.cropping_factor)))
        else:
            video = F.center_crop(video.squeeze(), (int(self.ori_reso_h*self.params.cropping_factor),int(self.ori_reso_w*self.params.cropping_factor)))
        video = F.resize(video, (self.params.reso_h, self.params.reso_w), antialias=True)
        return video.unsqueeze(dim=0)


class HybridValPipe(Pipeline):
    def __init__(self, data_dir, params, cropping_factor=0.8, num_threads=4, device_id=0, num_gpus=1):
        super(HybridValPipe, self).__init__(1, num_threads, device_id)
        dali_device = 'gpu'
        self.params = params
        self.input = fn.readers.video(
            filenames=data_dir,
            labels=[],
            # file_root=data_dir,
            sequence_length=params.num_frames, 
            num_shards=num_gpus, 
            shard_id=device_id, 
            device=dali_device, 
            random_shuffle=False, 
            # resize_x=params.reso_w, 
            # resize_y=params.reso_h, 
            # interp_type=types.INTERP_LINEAR, 
            # antialias=False, 
            pad_sequences=True,
            stride=params.fix_skip,
            step=params.num_frames*params.fix_skip,
            file_list_include_preceding_frame=True,
            dtype=types.DALIDataType.FLOAT,
            name='reader')


    def define_graph(self):
        videos, labels = self.input
        return [videos, labels]


# Concatenate features into non-overlapping segments, as in Sultani et al.
def segment_features(vid_features, num_features):
    segmented_features = np.zeros((32, num_features))
    segment_loc = np.linspace(0, vid_features.shape[0], 33, dtype=int)
    for idx in range(len(segment_loc) - 1):
        ss, es = segment_loc[idx], segment_loc[idx+1] - 1
        # Go to end of video.
        if idx == 31:
            es += 1
        if ss <= es or es < ss:
            temp_vect = vid_features[ss][:]
        else:
            temp_vect = np.mean(vid_features[ss:es][:])

        temp_vect = temp_vect/np.linalg.norm(temp_vect)
        segmented_features[idx] = temp_vect
    return segmented_features


if __name__ == '__main__':
    filenames = sorted(glob.glob(os.path.join(cfg.ucf_crimes_path, 'Videos', '*', '*')))
    # filenames = sorted(glob.glob(os.path.join(cfg.xd_path, 't*', '*')))
    # filenames.reverse()
    # Load in reconstruction F_a model, loaded ft_model.
    anonymized = True
    saved_fa_model = os.path.join('..', 'saved_models', 'model_20_bestAcc_0.7504.pth') if anonymized else None
    saved_ft_model = os.path.join('..', 'saved_models', 'model_20_bestAcc_0.7504.pth')
    save_features_folder = 'ucf_features_ours'
    print(f'Fa_model: {saved_fa_model}', flush=True)
    print(f'Ft_model: {saved_ft_model}', flush=True)
    print(f'Features folder: {save_features_folder}', flush=True)

    # Create features folder.
    if not os.path.exists(save_features_folder):
        os.makedirs(save_features_folder)

    # Remove all existing filenames.
    filenames = [x for x in filenames if not os.path.exists(os.path.join(save_features_folder, os.path.basename(x).replace('.mp4', '') + '.npy'))]
    fa_model = load_fa_model(arch='unet++', saved_model_file=saved_fa_model)
    ft_model = load_ft_model(arch='largei3d', kin_pretrained=True, saved_model_file=saved_ft_model, num_classes=params.num_classes)

    num_features = 2048
    devices = list(range(torch.cuda.device_count()))
    num_gpus = len(devices)
    if num_gpus > 1:
        ft_model = torch.nn.DataParallel(ft_model, device_ids=devices)
        ft_model.cuda()
        if anonymized:
            fa_model = torch.nn.DataParallel(fa_model, device_ids=devices)
            fa_model.cuda()
    else:
        ft_model.to(device=torch.device(devices[0]))
        if anonymized:
            fa_model.to(device=torch.device(devices[0]))

    ft_model.eval()
    if anonymized:
        fa_model.eval()
    
    # Dali dataloading.
    pipes = [HybridValPipe(data_dir=filenames, params=params, num_gpus=num_gpus, device_id=dev) for dev in devices]
    all_dataset = DALIDataloader(pipeline=pipes, params=params, reader_name='reader', batch_size=params.batch_size)
    
    prev_label = -1
    first_vid = True

    # Feature extraction loop.
    for _, (inputs, label) in enumerate(all_dataset):
        label = label.item()
        if label != prev_label:
            if not first_vid:
                # segmented_features = segment_features(vid_features[1:], num_features)
                np.save(save_path, vid_features[1:])
            prev_label = label
            vid_path = filenames[label]
            save_path = os.path.join(save_features_folder, os.path.basename(vid_path).replace('.mp4', '') + '.npy')
            if not os.path.exists(save_path):
                print(f'Extracting features for {os.path.basename(vid_path)}.', flush=True)

            vid_features = np.zeros(num_features)
            first_vid = False
            if os.path.exists(save_path):
                continue

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
            vid_features = np.vstack([vid_features, output.squeeze().cpu().numpy()])

    # segmented_features = segment_features(vid_features[1:], num_features)
    np.save(save_path, vid_features[1:])
