import glob
import numpy as np
import os
import pickle
import random
import time
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as trans

import sys
sys.path.insert(0, '..')
from privacy_training import params_privacy as params
from aux_code import config as cfg


# VISPR dataset.
class vispr_dataset(Dataset):
    def __init__(self, data_split, shuffle=True, data_percentage=1.0):
        self.data_split = data_split
        self.labels = pickle.load(open(os.path.join(cfg.vispr_path , f'{self.data_split}_labels.pkl'), 'rb'))
        data_path = os.path.join(cfg.vispr_path, f'{self.data_split}2017')  
        data_list = glob.glob(os.path.join(data_path, '*.jpg'))                   
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(data_list)
        
        # Data limiter.
        self.data_percentage = data_percentage
        data_limit = int(len(data_list)*self.data_percentage)
        self.data = data_list[0: data_limit]

        # Augmentation parameter.
        self.erase_size = 19

    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):      
        label = self.labels[os.path.basename(self.data[idx]).replace('.jpg', '')]
        img = self.build_image(self.data[idx])
        return img, label, self.data[idx]


    # Build and augment image.
    def build_image(self, img_path):
        try:
            img = torchvision.io.read_image(img_path)
            # Ensure image is proper shape.
            if img.shape[0] == 1:
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
            hue_factor1 = np.random.uniform(-0.05, 0.05, size=(2,))
            saturation_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            brightness_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            gamma1 = np.random.uniform(0.85, 1.15, size=(2,))
            erase_size1 = np.random.randint(int(self.erase_size/2), self.erase_size, size=(2,))
            erase_size2 = np.random.randint(int(self.erase_size/2), self.erase_size, size=(2,))
            random_color_dropped = np.random.randint(0, 3, (2,))

            # Convert to PIL for transforms. 
            # image = trans.functional.to_pil_image(image)

            # Always resize crop the image.
            image = trans.functional.resized_crop(image, y0, x0, int(ori_reso_h*cropping_factor1[0]), int(ori_reso_w*cropping_factor1[0]), (params.reso_h, params.reso_w), antialias=True)

            # Random augmentation probabilities image 1.
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

        return image / 255.0


# VISPR dataset.
class vispr_ssl_dataset(Dataset):
    def __init__(self, data_split, shuffle=True, data_percentage=1.0):
        self.data_split = data_split
        self.labels = pickle.load(open(os.path.join(cfg.vispr_path , f'{self.data_split}_labels.pkl'), 'rb'))
        data_path = os.path.join(cfg.vispr_path, f'{self.data_split}2017')    
        data_list = glob.glob(os.path.join(data_path, '*.jpg'))                   
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(data_list)
        
        # Data limiter.
        self.data_percentage = data_percentage
        data_limit = int(len(data_list)*self.data_percentage)
        self.data = data_list[0: data_limit]

        # Augmentation parameter.
        self.erase_size = 19

    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):      
        label = self.labels[os.path.basename(self.data[idx]).replace('.jpg', '')]
        img1, img2 = self.build_images(self.data[idx])
        return img1, img2, label, self.data[idx]


    # Build and augment image.
    def build_images(self, img_path):
        try:
            img = torchvision.io.read_image(img_path)
            # Ensure image is proper shape.
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            if not img.shape[0] == 3:
                print(f'{img_path} has {img.shape[0]} channels')
                return None, None
            
            # Apply augmentations to the images.
            img1, img2 = self.augmentation([img, img])

            try:
                assert (len(img.shape) != 0) 
                return img1, img2
            except:
                print(f'Image {img_path} Failed')
                return None, None

        except:
            print(f'Image {img_path} Failed')
            return None, None

    def augmentation(self, image_list):
        output_image_list = []
        if self.data_split == 'train':
            augmentation_count = len(image_list)
            # Compute augmenation strength.
            ori_reso_h, ori_reso_w = image_list[0].shape[1], image_list[0].shape[-1]
            x_erase = np.random.randint(0, params.reso_h, size = (augmentation_count,))
            y_erase = np.random.randint(0, params.reso_w, size = (augmentation_count,))
            # An average cropping factor is 80% i.e. covers 64% area.
            cropping_factor1 = np.random.uniform(0.6, 1, size = (augmentation_count,))
            x0 = np.random.randint(0, ori_reso_w - ori_reso_w*cropping_factor1[0] + 1) 
            y0 = np.random.randint(0, ori_reso_h - ori_reso_h*cropping_factor1[0] + 1)
            contrast_factor1 = np.random.uniform(0.9, 1.1, size=(augmentation_count,))
            hue_factor1 = np.random.uniform(-0.05, 0.05, size=(augmentation_count,))
            saturation_factor1 = np.random.uniform(0.9, 1.1, size=(augmentation_count,))
            brightness_factor1 = np.random.uniform(0.9, 1.1, size=(augmentation_count,))
            gamma1 = np.random.uniform(0.85, 1.15, size=(augmentation_count,))
            erase_size1 = np.random.randint(int(self.erase_size/2), self.erase_size, size=(augmentation_count,))
            erase_size2 = np.random.randint(int(self.erase_size/2), self.erase_size, size=(augmentation_count,))
            random_color_dropped = np.random.randint(0, 3, (augmentation_count,))

            # Loop through image list, augmenting each.
            for i, image in enumerate(image_list):
                # Convert to PIL for transforms. 
                # image = trans.functional.to_pil_image(image)
                
                # Always resize crop the image.
                image = trans.functional.resized_crop(image, y0, x0, int(ori_reso_h*cropping_factor1[i]), int(ori_reso_w*cropping_factor1[i]), (params.reso_h, params.reso_w), antialias=True)

                # Random augmentation probabilities.
                random_array = np.random.rand(8)

                if random_array[0] < 0.125/2:
                    image = trans.functional.adjust_contrast(image, contrast_factor=contrast_factor1[i]) # 0.75 to 1.25
                if random_array[1] < 0.3/2:
                    image = trans.functional.adjust_hue(image, hue_factor=hue_factor1[i]) # hue factor will be between [-0.25, 0.25]*0.4 = [-0.1, 0.1]
                if random_array[2] < 0.3/2:
                    image = trans.functional.adjust_saturation(image, saturation_factor=saturation_factor1[i]) # brightness factor will be between [0.75, 1,25]
                if random_array[3] < 0.3/2:
                    image = trans.functional.adjust_brightness(image, brightness_factor=brightness_factor1[i]) # brightness factor will be between [0.75, 1,25]
                if random_array[0] > 0.125/2 and random_array[0] < 0.25/2:
                    image = trans.functional.adjust_contrast(image, contrast_factor=contrast_factor1[i]) #0.75 to 1.25
                if random_array[4] > 0.9:
                    image = trans.functional.rgb_to_grayscale(image, num_output_channels=3)
                    if random_array[5] > 0.25:
                        image = trans.functional.adjust_gamma(image, gamma=gamma1[i], gain=1) #gamma range [0.8, 1.2]
                if random_array[6] > 0.5:
                    image = trans.functional.hflip(image)

                # image = trans.functional.to_tensor(image)

                if random_array[6] < 0.5/2 :
                    image = trans.functional.erase(image, x_erase[i], y_erase[i], erase_size1[i], erase_size2[i], v=0) 
                
                output_image_list.append(image / 255.0)
        else:
            h, w = image_list[0].shape[1], image_list[0].shape[-1]
            # Convert to PIL for transforms. 
            # image = trans.functional.to_pil_image(image)
            side = min(h, w)
            for image in image_list:
                image = trans.functional.center_crop(image, side)
                image = trans.functional.resize(image, (params.reso_h, params.reso_w), antialias=True)
                output_image_list.append(image / 255.0)

        return output_image_list


# VISPR boring dataset.
class vispr_boring_dataset(Dataset):
    def __init__(self, data_split, shuffle=True, data_percentage=1.0):
        self.data_split = data_split
        self.labels = pickle.load(open(os.path.join(cfg.vispr_path , f'{self.data_split}_labels.pkl'), 'rb'))
        data_path = os.path.join(cfg.vispr_path, f'{self.data_split}2017')    
        data_list = glob.glob(os.path.join(data_path, '*.jpg'))                   
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(data_list)
        
        # Data limiter.
        self.data_percentage = data_percentage
        data_limit = int(len(data_list)*self.data_percentage)
        self.data = data_list[0: data_limit]

        # Augmentation parameter.
        self.erase_size = 19

    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):      
        label = self.labels[os.path.basename(self.data[idx]).replace('.jpg', '')]
        vid = self.build_video(self.data[idx])
        return vid, label, self.data[idx]


    # Build and augment video.
    def build_video(self, img_path):
        try:
            img = torchvision.io.read_image(img_path)
            # Ensure image is proper shape.
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            if not img.shape[0] == 3:
                print(f'{img_path} has {img.shape[0]} channels')
                return None
            
            # Apply augmentations to the images.
            img = self.augmentation(img)

            try:
                assert (len(img.shape) != 0)
                img = img.unsqueeze(0).repeat(params.num_frames, 1, 1, 1)
                # print(img.repeat(params.num_frames).shape)
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
            hue_factor1 = np.random.uniform(-0.05, 0.05, size=(2,))
            saturation_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            brightness_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            gamma1 = np.random.uniform(0.85, 1.15, size=(2,))
            erase_size1 = np.random.randint(int(self.erase_size/2), self.erase_size, size=(2,))
            erase_size2 = np.random.randint(int(self.erase_size/2), self.erase_size, size=(2,))
            random_color_dropped = np.random.randint(0, 3, (2,))

            # Convert to PIL for transforms. 
            # image = trans.functional.to_pil_image(image)

            # Always resize crop the image.
            image = trans.functional.resized_crop(image, y0, x0, int(ori_reso_h*cropping_factor1[0]), int(ori_reso_w*cropping_factor1[0]), (params.reso_h, params.reso_w), antialias=True)

            # Random augmentation probabilities image 1.
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

        return image / 255.0


if __name__ == '__main__':
    # VISPR visualization.
    train_dataset = vispr_boring_dataset(data_split='train', shuffle=False, data_percentage=0.01)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size_vispr, shuffle=True, num_workers=params.num_workers)

    print(f'Step involved: {len(train_dataset)/params.batch_size_vispr}')
    t = time.time()

    for i, (img, label, img_path) in enumerate(train_dataloader):
        if i % 10 == 0:
            print()
            print(f'Image shape is {img.shape}')
            print(f'Label is {label}')

    print(f'Time taken to load data is {time.time()-t}')
