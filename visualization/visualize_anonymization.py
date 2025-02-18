# This code is to visualize and save videos after anonymization.
import imageio
import numpy as np
import os
import glob
from segmentation_models_pytorch import UnetPlusPlus
import torch
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

# Decord after torch.
import decord
decord.bridge.set_bridge('torch')


# Fa model loading function.
def load_fa_model(saved_model_file=None, arch='unet++'):
    if arch == 'unet++':
        fa_model = UnetPlusPlus(
            encoder_name='resnet18',
            encoder_depth=4,
            encoder_weights="imagenet",
            decoder_channels=(256, 128, 64, 32),
            decoder_attention_type=None,
            decoder_use_batchnorm=True,
            in_channels=3,
            classes=3,
            activation=None,
            aux_params=None
        )
    # elif arch == 'unet':
    #     fa_model = UNet(n_channels=3, n_classes=3)
    else:
        print(f'Architecture {arch} invalid for fa_model. Try \'unet\' or \'unet++\'')

    if saved_model_file:
        saved_dict = torch.load(saved_model_file)
        # Remove 'module.' prefix from state dict keys.
        saved_dict = {k.replace('module.', ''): v for k, v in saved_dict['fa_model_state_dict'].items()}
        fa_model.load_state_dict(saved_dict, strict=True)

        print(f'fa_model loaded from {saved_model_file} successfully!')
    else:
        print(f'fa_model freshly initialized!')

    fa_model.eval()

    return fa_model


def save_video(tensor, filename, fps=30):
    # Ensure the tensor is on the CPU and convert it to numpy.
    tensor = tensor.permute(0, 2, 3, 1).cpu().numpy()  # Assuming tensor is in (T, C, H, W) format and C=3
    
    # Normalize to [0,1] range.
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    # Convert to uint8 [0,255]
    tensor = (tensor * 255).astype(np.uint8)
    
    imageio.mimsave(filename, tensor, fps=fps)
    print(f"Video saved as {filename}")


def anonymize_videos(vid_paths, save_path='visualization'):
    use_cuda = torch.cuda.is_available()
    # Set to the model you want to use.
    model_path = os.path.join('..', 'saved_models', 'ted_spad.pth')
    fa_model = load_fa_model(model_path)

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    # Create save directory if it doesn't exist
    vis_path = os.path.join(save_path, os.path.basename(model_path))
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    if use_cuda:
        fa_model.cuda()

    for vid_path in vid_paths:
        print(f"Processing video: {os.path.basename(vid_path)}")
        try:
            vr = decord.VideoReader(vid_path, ctx=decord.cpu())
            frame_count = len(vr)

            # Load in video
            skip_rate = 1 # If you want to skip frames, set skip_rate > 1.
            frame_indices = np.arange(0, frame_count, skip_rate)
            frames = vr.get_batch(frame_indices).permute(0, 3, 1, 2)
            
            inputs = []
            for frame in frames:
                inputs.append(trans(frame))

            inputs = torch.stack(inputs, dim=0)

            if use_cuda:
                inputs = inputs.cuda()

            with torch.no_grad():
                output = fa_model(inputs)

            # Color flip.
            output = torch.flip(output, dims=[1])
            filename = os.path.join(vis_path, f'anon_{os.path.basename(vid_path).split(".")[0]}.mp4')
            # Save output as video
            save_video(output, filename, fps=vr.get_avg_fps())

        except Exception as e:
            print(f"Error processing {os.path.basename(vid_path)}: {str(e)}")
            continue


if __name__ == '__main__':
    # Set the path to the videos you want to anonymize.
    video_path = 'videos/*'
    save_path = './anon_visualization'
    vid_paths = glob.glob(video_path)
    anonymize_videos(vid_paths=vid_paths, save_path=save_path)
