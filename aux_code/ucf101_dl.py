import json
import numpy as np
import os
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as trans
import traceback

# Decord after torch.
import decord

import sys
sys.path.insert(0, '..')
import aux_code.config as cfg


decord.bridge.set_bridge('torch')


# Training dataloader.
class single_train_dataloader(Dataset):

    def __init__(self, params, shuffle=True, data_percentage=1.0, split=1, frame_wise_aug=False):
        self.params = params
        
        if split <= 3:
            all_paths = open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'trainlist0{split}.txt'),'r').read().splitlines()
            self.all_paths = [x.replace('/', os.sep) for x in all_paths]
        else:
            print(f'Invalid split input: {split}')
        self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']
            
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.erase_size = 19
        self.framewise_aug = frame_wise_aug

    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path, frame_list = self.process_data(index)
        return clip, label, vid_path, frame_list

    def process_data(self, idx):
        # Label building.
        vid_path = os.path.join(cfg.ucf101_path, 'Videos', self.data[idx].split(' ')[0])
        label = self.classes[vid_path.split(os.sep)[-2]]  # This element should be activity name.

        # Clip building.
        clip, frame_list = self.build_clip(vid_path)

        return clip, label, vid_path, frame_list
    
    def build_clip(self, vid_path):
        frame_count = -1
        try:
            vr = decord.VideoReader(vid_path, ctx=decord.cpu())
            frame_count = len(vr)

            skip_frames_full = self.params.fix_skip

            left_over = frame_count - self.params.fix_skip*self.params.num_frames

            if left_over > 0:
                start_frame_full = np.random.randint(0, int(left_over)) 
            else:
                skip_frames_full /= 2
                left_over = frame_count - skip_frames_full*self.params.num_frames
                start_frame_full = np.random.randint(0, int(left_over)) 

            frames_full = start_frame_full + np.asarray([int(int(skip_frames_full)*f) for f in range(self.params.num_frames)])

            # Some edge case fixing.
            if frames_full[-1] >= frame_count:
                # print('some corner case not covered')
                frames_full[-1] = int(frame_count-1)
            
            full_clip = []
            list_full = frames_full
            frames = vr.get_batch(frames_full)
            self.ori_reso_h, self.ori_reso_w = frames.shape[1:3]
            self.min_size = min(self.ori_reso_h, self.ori_reso_w)

            random_array = np.random.rand(2,10)
            x_erase = np.random.randint(0,self.params.reso_w, size = (2,))
            y_erase = np.random.randint(0,self.params.reso_h, size = (2,))

            # On an average cropping, factor is 80% i.e. covers 64% area.
            cropping_factor1 = np.random.uniform(self.params.min_crop_factor_training, 1, size = (2,)) 

            if not self.params.no_ar_distortion:
                x0 = np.random.randint(0, (self.ori_reso_w - self.ori_reso_w*cropping_factor1[0]) + 1)
                if self.params.aspect_ratio_aug:
                    y0 = np.random.randint(0, (self.ori_reso_h - self.ori_reso_h*cropping_factor1[1]) + 1)
                else:
                    y0 = np.random.randint(0, (self.ori_reso_h - self.ori_reso_h*cropping_factor1[0]) + 1)
            else:
                x0 = np.random.randint(0, (self.ori_reso_w - self.min_size*cropping_factor1[0]) + 1)
                y0 = np.random.randint(0, (self.ori_reso_h - self.min_size*cropping_factor1[0]) + 1)

            # Here augmentations are not strong as self-supervised training.
            # if not self.framewise_aug:
            contrast_factor1 = np.random.uniform(0.9,1.1, size = (2,))
            hue_factor1 = np.random.uniform(-0.05,0.05, size = (2,))
            saturation_factor1 = np.random.uniform(0.9,1.1, size = (2,))
            brightness_factor1 = np.random.uniform(0.9,1.1,size = (2,))
            gamma1 = np.random.uniform(0.85,1.15, size = (2,))
            erase_size1 = np.random.randint(int((self.ori_reso_h/6)*(self.params.reso_h/224)),int((self.ori_reso_h/3)*(self.params.reso_h/224)), size = (2,))
            erase_size2 = np.random.randint(int((self.ori_reso_w/6)*(self.params.reso_h/224)),int((self.ori_reso_w/3)*(self.params.reso_h/224)), size = (2,))
            random_color_dropped = np.random.randint(0,3,(2))

            for frame in frames:
                frame = torch.flip(frame.permute(2, 0, 1), dims=[0])
                if self.framewise_aug:
                    contrast_factor1 = np.random.uniform(0.9,1.1, size = (2,))
                    hue_factor1 = np.random.uniform(-0.05,0.05, size = (2,))
                    saturation_factor1 = np.random.uniform(0.9,1.1, size = (2,))
                    brightness_factor1 = np.random.uniform(0.9,1.1,size = (2,))
                    gamma1 = np.random.uniform(0.85,1.15, size = (2,))
                    erase_size1 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
                    erase_size2 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
                    random_color_dropped = np.random.randint(0,3,(2))
                
                if self.params.weak_aug:
                    full_clip.append(self.weak_augmentation(frame, cropping_factor1[0], x0, y0))
                else:
                    full_clip.append(self.augmentation(frame, random_array[0], x_erase, y_erase, cropping_factor1[0],\
                        x0, y0, contrast_factor1[0], hue_factor1[0], saturation_factor1[0], brightness_factor1[0],\
                        gamma1[0], erase_size1, erase_size2, random_color_dropped[0]))

            return full_clip, list_full
        except:
            traceback.print_exc()
            # print(f'Clip {vid_path} Failed, frame_count {frame_count}.')
            return None, None


    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1,\
        x0, y0, contrast_factor1, hue_factor1, saturation_factor1, brightness_factor1,\
        gamma1,erase_size1,erase_size2, random_color_dropped):
        
        image = self.PIL(image)
        if self.params.no_ar_distortion:
            image = trans.functional.resized_crop(image,y0,x0,int(self.min_size*cropping_factor1),int(self.min_size*cropping_factor1),(self.params.reso_h,self.params.reso_w), antialias=True)
        else:
            image = trans.functional.resized_crop(image,y0,x0,int(self.ori_reso_h*cropping_factor1),int(self.ori_reso_w*cropping_factor1),(self.params.reso_h,self.params.reso_w), antialias=True)

        if random_array[0] < 0.125/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[1] < 0.3/2 :
            image = trans.functional.adjust_hue(image, hue_factor = hue_factor1) 
        if random_array[2] < 0.3/2 :
            image = trans.functional.adjust_saturation(image, saturation_factor = saturation_factor1) 
        if random_array[3] < 0.3/2 :
            image = trans.functional.adjust_brightness(image, brightness_factor = brightness_factor1) 
        if random_array[0] > 0.125/2 and random_array[0] < 0.25/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[4] > 0.9:
            image = trans.functional.to_grayscale(image, num_output_channels = 3)
            if random_array[5] > 0.25:
                image = trans.functional.adjust_gamma(image, gamma = gamma1, gain=1)
        if random_array[6] > 0.5:
            image = trans.functional.hflip(image)

        image = trans.functional.to_tensor(image)

        if random_array[7] < 0.4 :
            image = trans.functional.erase(image, x_erase[0], y_erase[0], erase_size1[0], erase_size2[0], v=0) 
        if random_array[8] <0.4 :
            image = trans.functional.erase(image, x_erase[1], y_erase[1], erase_size1[1], erase_size2[1], v=0) 

        return image
    
    def weak_augmentation(self, image, cropping_factor1, x0, y0):
        
        image = self.PIL(image)
        if self.params.no_ar_distortion:
            image = trans.functional.resized_crop(image,y0,x0,int(self.min_size*cropping_factor1),int(self.min_size*cropping_factor1),(self.params.reso_h,self.params.reso_w), antialias=True)
        else:
            image = trans.functional.resized_crop(image,y0,x0,int(self.ori_reso_h*cropping_factor1),int(self.ori_reso_w*cropping_factor1),(self.params.reso_h,self.params.reso_w), antialias=True) 
        
        image = trans.functional.to_tensor(image)

        return image

    
# Validation dataset.
class single_val_dataloader(Dataset):

    def __init__(self, params, shuffle=True, data_percentage=1.0, mode=0, \
                hflip=0, cropping_factor=0.8, split=1, threeCrop=False):
        
        self.total_num_modes = params.num_modes
        self.params = params
        self.threecrop = threeCrop

        self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']
        if split <= 3:
            all_paths = open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'testlist0{split}.txt'),'r').read().splitlines()
            self.all_paths = [x.replace('/', os.sep) for x in all_paths]
        else:
            print(f'Invalid split input: {split}')
                
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.mode = mode
        self.hflip = hflip
        self.cropping_factor = cropping_factor
        if self.cropping_factor == 1:
            self.output_reso_h = int(params.reso_h/0.8)
            self.output_reso_w = int(params.reso_w/0.8)
        else:
            self.output_reso_h = int(params.reso_h)
            self.output_reso_w = int(params.reso_w)                       
    
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path, frame_list = self.process_data(index)
        return clip, label, os.path.basename(vid_path), frame_list


    def process_data(self, idx):
        # Label building.
        vid_path = os.path.join(cfg.ucf101_path, 'Videos', self.data[idx].split(' ')[0])
        label = self.classes[vid_path.split(os.sep)[-2]]  # This element should be activity name.

        # Clip building.
        clip, frame_list = self.build_clip(vid_path)

        return clip, label, vid_path, idx

    def build_clip(self, vid_path):
        frame_count = -1
        try:
            vr = decord.VideoReader(vid_path, ctx=decord.cpu())
            frame_count = len(vr)

            N = frame_count
            n = self.params.num_frames

            skip_frames_full = self.params.fix_skip 

            if skip_frames_full*n > N:
                skip_frames_full /= 2

            left_over = skip_frames_full*n
            F = N - left_over

            start_frame_full = 0 + int(np.linspace(0,F-10, self.total_num_modes)[self.mode])

            if start_frame_full< 0:
                start_frame_full = self.mode

            full_clip_frames = []

            full_clip_frames = start_frame_full + np.asarray(
                [int(int(skip_frames_full) * f) for f in range(self.params.num_frames)])

            
            full_clip = []
            list_full = full_clip_frames
            frames = vr.get_batch(full_clip_frames)
            self.ori_reso_w, self.ori_reso_h = frames.shape[1:3]
            self.min_size = min(self.ori_reso_h, self.ori_reso_w)
            for frame in frames:
                frame = torch.flip(frame.permute(2, 0, 1), dims=[0])
                full_clip.append(self.augmentation(frame))

            return full_clip, list_full
        except:
            # traceback.print_exc()
            # print(f'Clip {vid_path} Failed, frame_count {frame_count}.')
            return None, None


    def augmentation(self, image):
        image = self.PIL(image)

        if self.cropping_factor <= 1:
            if self.params.no_ar_distortion:
                image = trans.functional.center_crop(image,(int(self.min_size*self.cropping_factor),int(self.min_size*self.cropping_factor)))
            else:
                image = trans.functional.center_crop(image,(int(self.ori_reso_h*self.cropping_factor),int(self.ori_reso_w*self.cropping_factor)))
                
            if self.threecrop:
                image1 = trans.functional.five_crop(image,(int(self.ori_reso_h*self.cropping_factor),int(self.ori_reso_w*self.cropping_factor))) #torchvision doc says this is non deteministic function, may not always return 5 crops, since I am using bigger overlapping crops, should be fine to just take 2 of the corner crops, let's see how it works. 
                image1_1 = image1[0]
                image1_2 = image1[-2]

        image = trans.functional.resize(image, (self.output_reso_h, self.output_reso_w), antialias=True)
        if self.threecrop:
            image1_1 = trans.functional.resize(image1_1, (self.output_reso_h, self.output_reso_w), antialias=True)
            image1_2 = trans.functional.resize(image1_2, (self.output_reso_h, self.output_reso_w), antialias=True)
        if self.hflip !=0:
            image = trans.functional.hflip(image)
        if self.threecrop:
            return trans.functional.to_tensor(image), trans.functional.to_tensor(image1_1), trans.functional.to_tensor(image1_2)

        return trans.functional.to_tensor(image)


# Training dataloader.
class contrastive_train_dataloader(Dataset):

    def __init__(self, params, shuffle=True, data_percentage=1.0, split=1, frame_wise_aug=False):
        self.params = params
        
        if split <= 3:
            all_paths = open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'trainlist0{split}.txt'),'r').read().splitlines()
            self.all_paths = [x.replace('/', os.sep) for x in all_paths]
        else:
            print(f'Invalid split input: {split}')
        self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']

        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.erase_size = 19
        self.framewise_aug = frame_wise_aug

    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path, frame_list = self.process_data(index)
        return clip, label, vid_path, frame_list


    def process_data(self, idx):
        # Label building.
        vid_path = os.path.join(cfg.ucf101_path, 'Videos', self.data[idx].split(' ')[0])
        label = self.classes[vid_path.split(os.sep)[-2]]  # This element should be activity name.

        # Clip building.
        if self.params.temporal_loss == 'trip':
            clip1, clip2, clip3, frame_list1, frame_list2, frame_list3 = self.build_clip(vid_path)
        else:
            clip1, clip2, frame_list1, frame_list2 = self.build_clip(vid_path)

        if clip1 is not None:
            clip = []
            clip.extend(clip1)
            clip.extend(clip2)

            frame_list = []
            frame_list.extend(frame_list1)
            frame_list.extend(frame_list2)

            if self.params.temporal_loss == 'trip':
                clip.extend(clip3)
                frame_list.extend(frame_list3)
        else:
            clip = None
            frame_list = None

        return clip, label, vid_path, frame_list
    
    def build_clip(self, vid_path):
        frame_count = -1
        try:
            vr = decord.VideoReader(vid_path, ctx=decord.cpu())
            frame_count = len(vr)

            ############################# frame_list maker start here #################################

            skip_frames_full = self.params.fix_skip

            left_over = frame_count - self.params.fix_skip*self.params.num_frames

            temporal_align = self.params.temporal_align

            if self.params.temporal_distance:
                left_over2 = left_over - skip_frames_full*self.params.num_frames - self.params.temporal_distance
                if left_over2 > 0:
                    start_frame_full = np.random.randint(0, int(left_over2))
                    start_frame_full2 = start_frame_full + skip_frames_full*(self.params.num_frames - 1) + self.params.temporal_distance
                else:
                    skip_frames_full /= 2
                    left_over = frame_count - skip_frames_full*self.params.num_frames
                    left_over2 = left_over - skip_frames_full*self.params.num_frames - self.params.temporal_distance
                    start_frame_full = np.random.randint(0, int(left_over2))
                    start_frame_full2 = start_frame_full + skip_frames_full*(self.params.num_frames - 1) + self.params.temporal_distance
                
                frames_full2 = start_frame_full2 + np.asarray([int(int(skip_frames_full)*f) for f in range(self.params.num_frames)])

            else:
                if left_over > 0:
                    start_frame_full = np.random.randint(0, int(left_over))
                else:
                    skip_frames_full /= 2
                    left_over = frame_count - skip_frames_full*self.params.num_frames
                    start_frame_full = np.random.randint(0, int(max(0, left_over))) 
                
                if self.params.temporal_loss == 'trip':
                    temporal_align = True
                if not temporal_align:
                    start_frame_full2 = np.random.randint(0, int(left_over))
                    frames_full2 = start_frame_full2 + np.asarray([int(int(skip_frames_full)*f) for f in range(self.params.num_frames)])

            frames_full = start_frame_full + np.asarray([int(int(skip_frames_full)*f) for f in range(self.params.num_frames)])

            # Some edge case fixing.
            if frames_full[-1] >= frame_count:
                # print('some corner case not covered')
                frames_full[-1] = int(frame_count-1)
            
            if not temporal_align:
                # Some edge case fixing.
                if frames_full2[-1] >= frame_count:
                    # print('some corner case not covered')
                    frames_full2[-1] = int(frame_count-1)

            if self.params.temporal_loss == 'trip':
                temporal_align = True
                frames_full2 = frames_full
                if self.params.temporal_distance:
                    start_frame_full3 = start_frame_full2
                else:
                    start_frame_full3 = np.random.randint(0, int(left_over))
    
                frames_full3 = start_frame_full3 + np.asarray([int(int(skip_frames_full)*f) for f in range(self.params.num_frames)])
                
                # Some edge case fixing.
                if frames_full3[-1] >= frame_count:
                    # print('some corner case not covered')
                    frames_full3[-1] = int(frame_count-1)
            else:
                frames_full3 = None

            ################################ frame list maker finishes here ###########################

            ################################ actual clip builder starts here ##########################

            full_clip, full_clip2, full_clip3 = [], [], []
            list_full, list_full2, list_full3 = frames_full, frames_full2, frames_full3

            frames = vr.get_batch(frames_full)
            if not temporal_align:
                frames2 = vr.get_batch(frames_full2)
            if self.params.temporal_loss == 'trip':
                frames3 = vr.get_batch(frames_full3)

            self.ori_reso_h, self.ori_reso_w = frames.shape[1:3]
            self.min_size = min(self.ori_reso_h, self.ori_reso_w)

            random_array = np.random.rand(3,10)
            x_erase = np.random.randint(0,self.params.reso_w, size = (3,2))
            y_erase = np.random.randint(0,self.params.reso_h, size = (3,2))

            # On an average cropping, factor is 80% i.e. covers 64% area.
            cropping_factor1 = np.random.uniform(self.params.min_crop_factor_training, 1, size = (3,)) 

            if not self.params.no_ar_distortion:
                x0 = np.random.randint(0, (self.ori_reso_w - self.ori_reso_w*cropping_factor1[0]) + 1)
                if self.params.aspect_ratio_aug:
                    y0 = np.random.randint(0, (self.ori_reso_h - self.ori_reso_h*cropping_factor1[1]) + 1)
                else:
                    y0 = np.random.randint(0, (self.ori_reso_h - self.ori_reso_h*cropping_factor1[0]) + 1)
            else:
                x0 = np.random.randint(0, (self.ori_reso_w - self.min_size*cropping_factor1[0]) + 1)
                y0 = np.random.randint(0, (self.ori_reso_h - self.min_size*cropping_factor1[0]) + 1)

            # Here augmentations are not strong as self-supervised training.
            # if not self.framewise_aug:
            contrast_factor1 = np.random.uniform(0.9,1.1, size = (3,))
            hue_factor1 = np.random.uniform(-0.05,0.05, size = (3,))
            saturation_factor1 = np.random.uniform(0.9,1.1, size = (3,))
            brightness_factor1 = np.random.uniform(0.9,1.1,size = (3,))
            gamma1 = np.random.uniform(0.85,1.15, size = (3,))
            erase_size1 = np.random.randint(int((self.ori_reso_h/6)*(self.params.reso_h/224)),int((self.ori_reso_h/3)*(self.params.reso_h/224)), size = (3,2))
            erase_size2 = np.random.randint(int((self.ori_reso_w/6)*(self.params.reso_h/224)),int((self.ori_reso_w/3)*(self.params.reso_h/224)), size = (3,2))
            random_color_dropped = np.random.randint(0,3,(3))

            for frame in frames:
                # frame = torch.flip(frame.permute(2, 0, 1), dims=[0])
                frame = frame.permute(2, 0, 1)
                if self.framewise_aug:
                    contrast_factor1 = np.random.uniform(0.9,1.1, size = (3,))
                    hue_factor1 = np.random.uniform(-0.05,0.05, size = (3,))
                    saturation_factor1 = np.random.uniform(0.9,1.1, size = (3,))
                    brightness_factor1 = np.random.uniform(0.9,1.1,size = (3,))
                    gamma1 = np.random.uniform(0.85,1.15, size = (3,))
                    erase_size1 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (3,2))
                    erase_size2 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (3,2))
                    random_color_dropped = np.random.randint(0,3,(3))

                if self.params.weak_aug:
                    full_clip.append(self.weak_augmentation(frame, cropping_factor1[0], x0, y0))
                    if temporal_align:
                        full_clip2.append(self.weak_augmentation(frame, cropping_factor1[1], x0, y0))
                else:
                    full_clip.append(self.augmentation(frame, random_array[0], x_erase[0], y_erase[0], cropping_factor1[0],\
                        x0, y0, contrast_factor1[0], hue_factor1[0], saturation_factor1[0], brightness_factor1[0],\
                        gamma1[0], erase_size1[0], erase_size2[0], random_color_dropped[0]))
                    if temporal_align:
                        full_clip2.append(self.augmentation(frame, random_array[1], x_erase[1], y_erase[1], cropping_factor1[1],\
                        x0, y0, contrast_factor1[1], hue_factor1[1], saturation_factor1[1], brightness_factor1[1],\
                        gamma1[1], erase_size1[1], erase_size2[1], random_color_dropped[1]))

            if not temporal_align:
                for frame in frames2:
                    # frame = torch.flip(frame.permute(2, 0, 1), dims=[0])
                    frame = frame.permute(2, 0, 1)
                    if self.params.weak_aug:
                        full_clip2.append(self.weak_augmentation(frame, cropping_factor1[1], x0, y0))
                    else:
                        full_clip2.append(self.augmentation(frame, random_array[1], x_erase[1], y_erase[1], cropping_factor1[1],\
                            x0, y0, contrast_factor1[1], hue_factor1[1], saturation_factor1[1], brightness_factor1[1],\
                            gamma1[1], erase_size1[1], erase_size2[1], random_color_dropped[1]))

            if self.params.temporal_loss == 'trip':
                for frame in frames3:
                    # frame = torch.flip(frame.permute(2, 0, 1), dims=[0])
                    frame = frame.permute(2, 0, 1)
                    if self.params.weak_aug:
                        full_clip3.append(self.weak_augmentation(frame, cropping_factor1[2], x0, y0))
                    else:
                        full_clip3.append(self.augmentation(frame, random_array[2], x_erase[2], y_erase[2], cropping_factor1[2],\
                            x0, y0, contrast_factor1[2], hue_factor1[2], saturation_factor1[2], brightness_factor1[2],\
                            gamma1[2], erase_size1[2], erase_size2[2], random_color_dropped[2]))

            if len(full_clip) < self.params.num_frames and len(full_clip)>(self.params.num_frames/2) :
                print(f'Clip {vid_path} is missing {self.params.num_frames-len(full_clip)} frames.')
                remaining_num_frames = self.params.num_frames - len(full_clip)
                full_clip = full_clip + full_clip[::-1][1:remaining_num_frames+1]
                list_full = list_full + list_full[::-1][1:remaining_num_frames+1]

            if len(full_clip2) < self.params.num_frames and len(full_clip2)>(self.params.num_frames/2) :
                print(f'Clip {vid_path} is missing {self.params.num_frames-len(full_clip2)} frames.')
                remaining_num_frames = self.params.num_frames - len(full_clip2)
                full_clip2 = full_clip2 + full_clip2[::-1][1:remaining_num_frames+1]
                list_full2 = list_full2 + list_full2[::-1][1:remaining_num_frames+1]

            if self.params.temporal_loss == 'trip':
                if len(full_clip3) < self.params.num_frames and len(full_clip3)>(self.params.num_frames/2) :
                    print(f'Clip {vid_path} is missing {self.params.num_frames-len(full_clip3)} frames.')
                    remaining_num_frames = self.params.num_frames - len(full_clip3)
                    full_clip3 = full_clip3 + full_clip3[::-1][1:remaining_num_frames+1]
                    list_full3 = list_full3 + list_full3[::-1][1:remaining_num_frames+1]

            try:
                assert(len(full_clip) == self.params.num_frames)
                assert(len(full_clip2) == self.params.num_frames)
                if self.params.temporal_loss == 'trip':
                    assert(len(full_clip3) == self.params.num_frames)
                    return full_clip, full_clip2, full_clip3, list_full, list_full2, list_full3
                return full_clip, full_clip2, list_full, list_full2
            except:
                print(frames_full)
                if not temporal_align:
                    print(frames_full2)
                
                if self.params.temporal_loss == 'trip':
                    print(frames_full3)
                    print(f'Clip {vid_path} Failed')
                    return None, None, None, None, None, None

                print(f'Clip {vid_path} Failed')
                return None, None, None, None
        except:
            # traceback.print_exc()
            print(f'Clip {vid_path} Failed')
            if self.params.temporal_loss == 'trip':
                return None, None, None, None, None, None
            return None, None, None, None


    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1,\
        x0, y0, contrast_factor1, hue_factor1, saturation_factor1, brightness_factor1,\
        gamma1,erase_size1,erase_size2, random_color_dropped):
        
        image = self.PIL(image)
        if self.params.no_ar_distortion:
            image = trans.functional.resized_crop(image,y0,x0,int(self.min_size*cropping_factor1),int(self.min_size*cropping_factor1),(self.params.reso_h,self.params.reso_w), antialias=True)
        else:
            image = trans.functional.resized_crop(image,y0,x0,int(self.ori_reso_h*cropping_factor1),int(self.ori_reso_w*cropping_factor1),(self.params.reso_h,self.params.reso_w), antialias=True)

        if random_array[0] < 0.125/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[1] < 0.3/2 :
            image = trans.functional.adjust_hue(image, hue_factor = hue_factor1) 
        if random_array[2] < 0.3/2 :
            image = trans.functional.adjust_saturation(image, saturation_factor = saturation_factor1) 
        if random_array[3] < 0.3/2 :
            image = trans.functional.adjust_brightness(image, brightness_factor = brightness_factor1) 
        if random_array[0] > 0.125/2 and random_array[0] < 0.25/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[4] > 0.9:
            image = trans.functional.to_grayscale(image, num_output_channels = 3)
            if random_array[5] > 0.25:
                image = trans.functional.adjust_gamma(image, gamma = gamma1, gain=1)
        if random_array[6] > 0.5:
            image = trans.functional.hflip(image)

        image = trans.functional.to_tensor(image)

        if random_array[7] < 0.4 :
            image = trans.functional.erase(image, x_erase[0], y_erase[0], erase_size1[0], erase_size2[0], v=0) 
        if random_array[8] <0.4 :
            image = trans.functional.erase(image, x_erase[1], y_erase[1], erase_size1[1], erase_size2[1], v=0) 

        return image
    
    def weak_augmentation(self, image, cropping_factor1, x0, y0):
        
        image = self.PIL(image)
        if self.params.no_ar_distortion:
            image = trans.functional.resized_crop(image,y0,x0,int(self.min_size*cropping_factor1),int(self.min_size*cropping_factor1),(self.params.reso_h,self.params.reso_w), antialias=True)
        else:
            image = trans.functional.resized_crop(image,y0,x0,int(self.ori_reso_h*cropping_factor1),int(self.ori_reso_w*cropping_factor1),(self.params.reso_h,self.params.reso_w), antialias=True) 
        
        image = trans.functional.to_tensor(image)

        return image


# Validation dataset.
class contrastive_val_dataloader(Dataset):

    def __init__(self, params, shuffle=True, data_percentage=1.0, mode=0, hflip=0, cropping_factor=0.8, split=1, threeCrop=False):
        
        self.total_num_modes = params.num_modes
        self.params = params
        self.threecrop = threeCrop

        self.classes = json.load(open(cfg.ucf101_class_mapping))['classes']
        if split <= 3:
            all_paths = open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'testlist0{split}.txt'),'r').read().splitlines()
            self.all_paths = [x.replace('/', os.sep) for x in all_paths]
        else:
            print(f'Invalid split input: {split}')    
                
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.mode = mode
        self.hflip = hflip
        self.cropping_factor = cropping_factor
        if self.cropping_factor == 1:
            self.output_reso_h = int(params.reso_h/0.8)
            self.output_reso_w = int(params.reso_w/0.8)
        else:
            self.output_reso_h = int(params.reso_h)
            self.output_reso_w = int(params.reso_w)                       
    
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path, frame_list = self.process_data(index)
        return clip, label, vid_path, frame_list


    def process_data(self, idx):
        # Label building.
        vid_path = os.path.join(cfg.ucf101_path, 'Videos', self.data[idx].split(' ')[0])
        label = self.classes[vid_path.split(os.sep)[-2]]  # This element should be activity name.

        # Clip building.
        if self.params.temporal_loss == 'trip':
            clip1, clip2, clip3, frame_list1, frame_list2, frame_list3 = self.build_clip(vid_path)
        else:
            clip1, clip2, frame_list1, frame_list2 = self.build_clip(vid_path)

        if clip1 is not None:
            clip = []
            clip.extend(clip1)
            clip.extend(clip2)

            frame_list = []
            frame_list.extend(frame_list1)
            frame_list.extend(frame_list2)

            if self.params.temporal_loss == 'trip':
                clip.extend(clip3)
                frame_list.extend(frame_list3)
        else:
            clip = None
            frame_list = None

        return clip, label, vid_path, frame_list

    def build_clip(self, vid_path):
        frame_count = -1
        try:
            vr = decord.VideoReader(vid_path, ctx=decord.cpu())
            frame_count = len(vr)

            N = frame_count
            n = self.params.num_frames

            skip_frames_full = self.params.fix_skip

            if skip_frames_full*n > N:
                skip_frames_full /= 2

            temporal_align = self.params.temporal_align
            if self.params.temporal_loss == 'trip':
                temporal_align = True
            if self.params.temporal_distance:
                left_over = N - n*skip_frames_full
                left_over2 = left_over - skip_frames_full*n - self.params.temporal_distance
                if left_over2 < 0:
                    if self.params.temporal_loss == 'trip':
                        return None, None, None, None, None, None
                    return None, None, None, None
                start_frame_full = 0 + int(np.linspace(0,left_over2-1, self.total_num_modes)[self.mode])
                start_frame_full2 = start_frame_full + (n-1)*skip_frames_full + self.params.temporal_distance
                
                frames_full2 = start_frame_full2 + np.asarray([int(int(skip_frames_full)*f) for f in range(self.params.num_frames)])

            else:
                left_over = skip_frames_full*n
                F = N - left_over

                start_frame_full = 0 + int(np.linspace(0,F-10, self.total_num_modes)[self.mode])

                if start_frame_full< 0:
                    start_frame_full = self.mode
                if not temporal_align:
                    start_frame_full2 = 0 + int(np.linspace(0,F-10, self.total_num_modes)[self.mode])
                    frames_full2 = start_frame_full2 + np.asarray([int(int(skip_frames_full)*f) for f in range(self.params.num_frames)])

            frames_full1 = start_frame_full + np.asarray([int(int(skip_frames_full)*f) for f in range(self.params.num_frames)])

            # Some edge case fixing.
            if frames_full1[-1] >= frame_count:
                # print('some corner case not covered')
                frames_full1[-1] = int(frame_count-1)
            
            if not temporal_align:
                # Some edge case fixing.
                if frames_full2[-1] >= frame_count:
                    # print('some corner case not covered')
                    frames_full2[-1] = int(frame_count-1)

            if self.params.temporal_loss == 'trip':
                temporal_align = True
                frames_full2 = frames_full1
                if self.params.temporal_distance:
                    start_frame_full3 = start_frame_full2
                else:
                    start_frame_full3 = np.random.randint(0, int(left_over))
                frames_full3 = start_frame_full3 + np.asarray([int(int(skip_frames_full)*f) for f in range(self.params.num_frames)])
                # Some edge case fixing.
                if frames_full3[-1] >= frame_count:
                    # print('some corner case not covered')
                    frames_full3[-1] = int(frame_count-1)
            else:
                frames_full3 = None
            
            # Clip builder here.
            full_clip1, full_clip2, full_clip3 = [], [], []
            list_full1, list_full2, list_full3 = frames_full1, frames_full2, frames_full3

            frames = vr.get_batch(frames_full1)
            if not temporal_align:
                frames2 = vr.get_batch(frames_full2)
            if self.params.temporal_loss == 'trip':
                frames3 = vr.get_batch(frames_full3)

            self.ori_reso_h, self.ori_reso_w = frames.shape[1:3]
            self.min_size = min(self.ori_reso_h, self.ori_reso_w)

            for frame in frames:
                frame = torch.flip(frame.permute(2, 0, 1), dims=[0])

                if self.threecrop:
                    full_clip1.extend(self.augmentation(frame))
                    if temporal_align:
                        full_clip2.extend(self.augmentation(frame))
                else:
                    full_clip1.append(self.augmentation(frame))
                    if temporal_align:
                        full_clip2.append(self.augmentation(frame))

            if not temporal_align:
                for frame in frames2:
                    frame = torch.flip(frame.permute(2, 0, 1), dims=[0])
                    full_clip2.append(self.augmentation(frame))

            if self.params.temporal_loss == 'trip':
                for frame in frames3:
                    frame = torch.flip(frame.permute(2, 0, 1), dims=[0])
                    full_clip3.append(self.augmentation(frame))

            # Appending the remaining frames in case of clip length < required frames.
            if not self.threecrop:
                if len(full_clip1) < self.params.num_frames and len(full_clip1)>(self.params.num_frames/2):
                    remaining_num_frames = self.params.num_frames - len(full_clip1)
                    full_clip1 = full_clip1 + full_clip1[::-1][1:remaining_num_frames+1]
                    list_full1 = list_full1 + list_full1[::-1][1:remaining_num_frames+1]
                if len(full_clip2) < self.params.num_frames and len(full_clip2)>(self.params.num_frames/2):
                    remaining_num_frames = self.params.num_frames - len(full_clip2)
                    full_clip2 = full_clip2 + full_clip2[::-1][1:remaining_num_frames+1]
                    list_full2 = list_full2 + list_full2[::-1][1:remaining_num_frames+1]
                if self.params.temporal_loss == 'trip':
                    if len(full_clip3) < self.params.num_frames and len(full_clip3)>(self.params.num_frames/2) :
                        remaining_num_frames = self.params.num_frames - len(full_clip3)
                        full_clip3 = full_clip3 + full_clip3[::-1][1:remaining_num_frames+1]
                        list_full3 = list_full3 + list_full3[::-1][1:remaining_num_frames+1]
            else:
                if len(full_clip1) < 3*self.params.num_frames and len(full_clip1)>(3*self.params.num_frames/2):
                    remaining_num_frames = 3*self.params.num_frames - len(full_clip1)
                    full_clip1 = full_clip1 + full_clip1[::-1][1:remaining_num_frames+1]
                    list_full1 = list_full1 + list_full1[::-1][1:remaining_num_frames+1]
                if len(full_clip2) < 3*self.params.num_frames and len(full_clip2)>(3*self.params.num_frames/2):
                    remaining_num_frames = 3*self.params.num_frames - len(full_clip2)
                    full_clip2 = full_clip2 + full_clip2[::-1][1:remaining_num_frames+1]
                    list_full2 = list_full2 + list_full2[::-1][1:remaining_num_frames+1]
                if self.params.temporal_loss == 'trip':
                    if len(full_clip3) < 3*self.params.num_frames and len(full_clip3)>(3*self.params.num_frames/2) :
                        remaining_num_frames = 3*self.params.num_frames - len(full_clip3)
                        full_clip3 = full_clip3 + full_clip3[::-1][1:remaining_num_frames+1]
                        list_full3 = list_full3 + list_full3[::-1][1:remaining_num_frames+1]
            if not self.threecrop:
                assert (len(full_clip1) == self.params.num_frames)
                assert (len(full_clip2) == self.params.num_frames)
                if self.params.temporal_loss == 'trip':
                    assert(len(full_clip3) == self.params.num_frames)
                    return full_clip1, full_clip2, full_clip3, list_full1, list_full2, list_full3
            else:
                assert (len(full_clip1) == 3*self.params.num_frames)
                assert (len(full_clip2) == 3*self.params.num_frames)
                if self.params.temporal_loss == 'trip':
                    assert(len(full_clip3) == 3*self.params.num_frames)
                    return full_clip1, full_clip2, full_clip3, list_full1, list_full2, list_full3

            return full_clip1, full_clip2, list_full1, list_full2
        except:
            traceback.print_exc()
            print(f'Clip {vid_path} Failed, frame_count {frame_count}.')
            if self.params.temporal_loss == 'trip':
                return None, None, None, None, None, None
            return None, None, None, None


    def augmentation(self, image):
        image = self.PIL(image)

        if self.cropping_factor <= 1:
            if self.params.no_ar_distortion:
                image = trans.functional.center_crop(image,(int(self.min_size*self.cropping_factor),int(self.min_size*self.cropping_factor)))
            else:
                image = trans.functional.center_crop(image,(int(self.ori_reso_h*self.cropping_factor),int(self.ori_reso_w*self.cropping_factor)))
                
            if self.threecrop:
                image1 = trans.functional.five_crop(image,(int(self.ori_reso_h*self.cropping_factor),int(self.ori_reso_w*self.cropping_factor))) #torchvision doc says this is non deteministic function, may not always return 5 crops, since I am using bigger overlapping crops, should be fine to just take 2 of the corner crops, let's see how it works. 
                image1_1 = image1[0]
                image1_2 = image1[-2]

        image = trans.functional.resize(image, (self.output_reso_h, self.output_reso_w), antialias=True)
        if self.threecrop:
            image1_1 = trans.functional.resize(image1_1, (self.output_reso_h, self.output_reso_w), antialias=True)
            image1_2 = trans.functional.resize(image1_2, (self.output_reso_h, self.output_reso_w), antialias=True)
        if self.hflip !=0:
            image = trans.functional.hflip(image)
        if self.threecrop:
            return trans.functional.to_tensor(image), trans.functional.to_tensor(image1_1), trans.functional.to_tensor(image1_2)

        return trans.functional.to_tensor(image)



def collate_fn_val(batch):

    f_clip, label, vid_path, frame_list = [], [], [], []

    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0],dim=0)) 
            label.append(item[1])
            vid_path.append(item[2])
            frame_list.append(item[3])

    if len(f_clip) < 2:
        return None, None, None, None
    f_clip = torch.stack(f_clip, dim=0)
    label = torch.tensor(label)
    frame_list = torch.tensor(frame_list)
    
    return f_clip, label, vid_path, frame_list 


def collate_fn_train(batch):

    f_clip, label, vid_path, frame_list = [], [], [], []
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0],dim=0)) 
            label.append(item[1])
            vid_path.append(item[2])
            # frame_list.append(item[3])

    if len(f_clip) < 2:
        return None, None, None, None
    f_clip = torch.stack(f_clip, dim=0)
    label = torch.tensor(label)
    # frame_list = torch.stack(frame_list, dim=0)

    return f_clip, label, vid_path, frame_list


if __name__ == '__main__':
    import anonymization_training.params_anonymization as params
    
    # train_dataset = single_train_dataloader(params=params, shuffle=True, data_percentage=0.1)
    # train_dataset = single_val_dataloader(params=params, shuffle=False, data_percentage=0.1)
    train_dataset = contrastive_train_dataloader(params=params, shuffle=True, data_percentage=0.1)
    # train_dataset = contrastive_val_dataloader(params=params, shuffle=False, data_percentage=0.1)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn_train, num_workers=0)#params.num_workers)

    print(f'Length of training dataset: {len(train_dataset)}')
    print(f'Steps involved: {len(train_dataset)/params.batch_size}')
    t = time.time()

    import matplotlib.pyplot as plt

    for i, (clip, label, vid_path, frame_list) in enumerate(train_dataloader):
        if i % 10 == 0:
            print()
            print(f'Full_clip shape is {clip.shape}')

            inputs1, inputs2, inputs3 = torch.split(clip, [params.num_frames, params.num_frames, params.num_frames], dim=1)
            print(f'Inputs1 shape is {inputs1.shape}')
            print(f'Inputs2 shape is {inputs2.shape}')
            print(f'Inputs3 shape is {inputs3.shape}')

            # Plot 8 frames from each input on same plot. Like this: anon_images = np.concatenate(torch.flip(output[::2], dims=[1]).cpu().numpy().transpose(0, 2, 3, 1), axis=1)
            fig, ax = plt.subplots(3, 1, figsize=(10, 5))
            clip1 = np.concatenate(inputs1[0][::2].cpu().numpy().transpose(0, 2, 3, 1), axis=1)
            clip2 = np.concatenate(inputs2[0][::2].cpu().numpy().transpose(0, 2, 3, 1), axis=1)
            clip3 = np.concatenate(inputs3[0][::2].cpu().numpy().transpose(0, 2, 3, 1), axis=1)
            ax[0].imshow(clip1)
            ax[0].axis('off')
            ax[1].imshow(clip2)
            ax[1].axis('off')
            ax[2].imshow(clip3)
            ax[2].axis('off')
            plt.tight_layout()
            plt.show()
            exit()

            # print(f'Label is {label}')
            # print(f'Frame list is {frame_list}')
            continue
