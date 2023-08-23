import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import glob
import time
import torchvision
import torchvision.transforms as trans

import parameters as params
import sys
sys.path.insert(0, '..')
from aux_code import config as cfg


# UCF_101 + VISPR dataset.
class reconstruction_dataset(Dataset):
    def __init__(self, data_split, shuffle=True, ucf101_percentage=1.0, data_percentage=1.0):
        self.data_split = data_split
        vispr_path = os.path.join(cfg.vispr_path, f'{self.data_split}2017')    
        vispr_data_list = glob.glob(os.path.join(vispr_path, '*.jpg'))
        # Requires UCF101-frames folder to be present in the same directory.
        ucf101_path = os.path.join(cfg.ucf101_path, 'UCF101-frames' f'{self.data_split}')
        ucf101_data_list = glob.glob(os.path.join(ucf101_path, '*', '*.jpg'))
        self.shuffle = shuffle

        # UCF101 data limiter.
        ucf101_limit = int(len(ucf101_data_list)*ucf101_percentage)
        ucf101_data = ucf101_data_list[0: ucf101_limit]

        # Combine data.
        data_list = ucf101_data + vispr_data_list

        if self.shuffle:
            random.shuffle(data_list)
        
        # Overall data limiter.
        self.data_percentage = data_percentage
        data_limit = int(len(data_list)*self.data_percentage)
        self.data = data_list[0: data_limit]

        # Augmentation parameter.
        self.erase_size = 19

    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):      
        return self.build_image(self.data[idx]), self.data[idx]


    # Build and augment image.
    def build_image(self, img_path):
        try:
            img = torchvision.io.read_image(img_path)
            # Ensure image is proper shape.
            if img.shape[0] == 1:
                # print(img.shape)
                img = img.repeat(3, 1, 1)
            if not img.shape[0] == 3:
                print(f'{img_path} has {img.shape[0]} channels')
                return None
            
            # Apply augmentations to the images.
            img = self.augmentation(img)

            try:
                assert (len(img.shape) != 0) 
                return img
            except:
                print(f'Image {img_path} Failed')
                return None   

        except:
            print(f'Image {img_path} Failed')
            return None

    def augmentation(self, image):

        if self.data_split == 'train':
            # Compute augmenation strength.
            ori_reso_h, ori_reso_w = image.shape[1], image.shape[-1]
            x_erase = np.random.randint(0, params.reso_h, size=(2,))
            y_erase = np.random.randint(0, params.reso_w, size=(2,))
            # An average cropping factor is 80% i.e. covers 64% area.
            cropping_factor1 = np.random.uniform(0.6, 1, size=(2,))
            x0 = np.random.randint(0, ori_reso_w - ori_reso_w*cropping_factor1[0] + 1) 
            y0 = np.random.randint(0, ori_reso_h - ori_reso_h*cropping_factor1[0] + 1)
            contrast_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            hue_factor1 = np.random.uniform(-0.05, 0.05, size= (2,))
            saturation_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            brightness_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            gamma1 = np.random.uniform(0.85, 1.15, size=(2,))
            erase_size1 = np.random.randint(int(self.erase_size/2), self.erase_size, size=(2,))
            erase_size2 = np.random.randint(int(self.erase_size/2), self.erase_size, size=(2,))
            random_color_dropped = np.random.randint(0,3,(2))

            # Convert to PIL for transforms. 
            # image = trans.functional.to_pil_image(image)

            # Always resize crop the image.
            image = trans.functional.resized_crop(image, y0, x0, int(ori_reso_h*cropping_factor1[0]), int(ori_reso_w*cropping_factor1[0]), (params.reso_h, params.reso_w), antialias=True)

            # Random augmentation probabilities.
            random_array = np.random.rand(8)

            if random_array[0] < 0.125/2:
                image = trans.functional.adjust_contrast(image, contrast_factor=contrast_factor1[0]) # 0.75 to 1.25
            if random_array[1] < 0.3/2:
                image = trans.functional.adjust_hue(image, hue_factor=hue_factor1[0]) # hue factor will be between [-0.25, 0.25]*0.4 = [-0.1, 0.1]
            if random_array[2] < 0.3/2:
                image = trans.functional.adjust_saturation(image, saturation_factor=saturation_factor1[0]) # brightness factor will be between [0.75, 1,25]
            if random_array[3] < 0.3/2:
                image = trans.functional.adjust_brightness(image, brightness_factor=brightness_factor1[0]) # brightness factor will be between [0.75, 1,25]
            if random_array[0] > 0.125/2 and random_array[0] < 0.25/2:
                image = trans.functional.adjust_contrast(image, contrast_factor=contrast_factor1[0]) #0.75 to 1.25
            if random_array[4] > 0.9:
                image = trans.functional.rgb_to_grayscale(image, num_output_channels=3)
                if random_array[5] > 0.25:
                    image = trans.functional.adjust_gamma(image, gamma=gamma1[0], gain=1) #gamma range [0.8, 1.2]
            if random_array[6] > 0.5:
                image = trans.functional.hflip(image)

            # image = trans.functional.to_tensor(image)

            if random_array[6] < 0.5/2 :
                image = trans.functional.erase(image, x_erase[0], y_erase[0], erase_size1[0], erase_size2[0], v=0) 
        else:
            h, w = image.shape[1], image.shape[-1]
            # Convert to PIL for transforms. 
            # image = trans.functional.to_pil_image(image)
            side = min(h, w)
            image = trans.functional.center_crop(image, side)
            image = trans.functional.resize(image, (params.reso_h, params.reso_w), antialias=True)
            image = image

        return image / 255.0


if __name__ == '__main__':

    train_dataset = reconstruction_dataset(data_split='test', shuffle=False, ucf101_percentage=1.0, data_percentage=1.0)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, \
        shuffle=False, num_workers=params.num_workers)

    print(f'Step involved: {len(train_dataset)/params.batch_size}')
    t = time.time()

    for i, (img, img_path) in enumerate(train_dataloader):
        if i%10 == 0:
            print()
            print(f'Image shape is {img.shape}')

    print(f'Time taken to load data is {time.time()-t}')
