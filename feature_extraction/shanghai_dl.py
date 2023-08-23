import glob
import os.path
import torch
from torch.utils.data import Dataset
from torchvision import transforms as trans
import cv2

import params_feature_ex as params
import sys
sys.path.insert(0, '..')
import aux_code.config as cfg


# Dataloader for feature extraction.
class shanghai_frames_dataset(Dataset):
    def __init__(self, reverse=False):
        self.video_list = sorted(glob.glob(os.path.join(cfg.shanghai_path,'t*', 'videos', '*')))
        if reverse:
            self.video_list.reverse()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        return self.read_video(self.video_list[idx])

    def augmentation(self, image):
        self.ori_reso_h = image.shape[0]
        self.ori_reso_w = image.shape[1]
        self.min_size = min(image.shape)
        image = trans.functional.to_pil_image(image)
        if params.no_ar_distortion:
            image = trans.functional.center_crop(image, (int(self.min_size*params.cropping_factor), int(self.min_size*params.cropping_factor)))
        else:
            image = trans.functional.center_crop(image, (int(self.ori_reso_h*params.cropping_factor), int(self.ori_reso_h*params.cropping_factor)))

        image = trans.functional.resize(image, (params.reso_h, params.reso_w), antialias=True)
        # Convert frame to tensor.
        image = trans.functional.to_tensor(image)
        return image

    # Clip builder, reads video frames from custom start and end times (if necessary), stacks them.
    def read_video(self, vid_path):
        try:
            cap = cv2.VideoCapture(vid_path)
            # Fix fps to 30.
            cap.set(cv2.CAP_PROP_FPS, 25)
            cap.set(1, 0)
     
            total_frames = cap.get(7)
            # print(total_frames)

            full_vid = []
            full_frame_pos = []
            frame_pos = []
            clip_frames = []
            count = 0
            repeat = False
            if total_frames < params.fix_skip*params.num_frames:
                fix_skip = 1
            else:
                fix_skip = params.fix_skip

            if total_frames < params.num_frames:
                repeat = True
            # Loops from start to end, adding frames and timestamps to list.
            while cap.isOpened():
                count += 1
                ret, frame = cap.read()
                if repeat and count == total_frames:
                    keep_frame = frame
                if ret:
                    if count % fix_skip == 0:
                        clip_frames.append(self.augmentation(frame))
                        frame_pos.append(count)
                        if count % (params.num_frames*fix_skip) == 0:
                            full_vid.append(torch.stack(clip_frames, dim=0))
                            full_frame_pos.append(frame_pos)
                            frame_pos = []
                            clip_frames = []
                else:
                    break
            cap.release()
            if repeat:
                count -= 1
                last_frame = count
                # In case of size mismatch, stack last frame.
                while count % 16 != 0:
                    count += 1
                    clip_frames.append(self.augmentation(keep_frame))
                    frame_pos.append(last_frame)
                    if count % 16 == 0:
                        full_vid.append(torch.stack(clip_frames, dim=0))
                        full_frame_pos.append(frame_pos)
            # print(full_vid)
            return full_vid, full_frame_pos, vid_path
        except:
            return None, None, vid_path


if __name__ == '__main__':
    dataset = shanghai_frames_dataset()
    print(len(dataset))
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    def show_images(image_batch):
        columns = 4
        rows = (16 + 1) // (columns)
        fig = plt.figure(figsize = (24,(24 // columns) * rows))
        gs = gridspec.GridSpec(rows, columns)
        for j in range(rows*columns):
            plt.subplot(gs[j])
            plt.axis("off")
            plt.imshow(image_batch[j].permute(1, 2, 0))
        plt.show()

    for vid, pos, path in dataset:
        print(len(vid))
        print(vid[0].shape)
        show_images(vid[0])
        exit()
