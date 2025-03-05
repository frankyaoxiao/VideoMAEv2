"""Extract features for temporal action detection datasets"""
import argparse
import os
import random

import numpy as np
import torch
from timm.models import create_model
from torchvision import transforms
from tqdm import tqdm

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
from dataset.loader import get_video_loader


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)


class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


def get_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument(
        '--data_set',
        default='THUMOS14',
        choices=['THUMOS14', 'FINEACTION', 'ANIMALKINGDOM'],
        type=str,
        help='dataset')

    parser.add_argument(
        '--data_path',
        default='YOUR_PATH/thumos14_video',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--save_path',
        default='YOUR_PATH/thumos14_video/th14_vit_g_16_4',
        type=str,
        help='path for saving features')

    parser.add_argument(
        '--model',
        default='pretrain_videomae_giant_patch14_224',
        type=str,
        metavar='MODEL',
        help='Name of model')
    parser.add_argument(
        '--ckpt_path',
        default='YOUR_PATH/vit_g_hyrbid_pt_1200e_k710_ft.pth',
        help='load from checkpoint')
    
    parser.add_argument(
        '--slice_id',
        default=0,
        type=int,
        choices=range(8),
        help='which slice to process (0-7) out of 8 total slices')

    return parser.parse_args()


def get_start_idx_range(data_set):

    def thumos14_range(num_frames):
        if num_frames < 16:
            return range(0, 1)  # Return at least one index for very short videos
        return range(0, num_frames - 15, 4)

    def fineaction_range(num_frames):
        if num_frames < 16:
            return range(0, 1)
        return range(0, num_frames - 15, 16)
    
    def animalkingdom_range(num_frames):
        if num_frames < 16:
            return range(0, 1)
        return range(0, num_frames - 15, 4)

    if data_set == 'THUMOS14':
        return thumos14_range
    elif data_set == 'FINEACTION':
        return fineaction_range
    elif data_set == 'ANIMALKINGDOM':
        return animalkingdom_range
    else:
        raise NotImplementedError()


def extract_feature(args):
    # preparation
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    video_loader = get_video_loader()
    start_idx_range = get_start_idx_range(args.data_set)
    transform = transforms.Compose(
        [ToFloatTensorInZeroOne(),
         Resize((224, 224))])

    vid_list = os.listdir(args.data_path)
    vid_list.sort()
    
    total_videos = len(vid_list)
    slice_size = (total_videos + 7) // 8
    start_idx = args.slice_id * slice_size
    end_idx = min((args.slice_id + 1) * slice_size, total_videos)
    vid_list = vid_list[start_idx:end_idx]
    
    print(f"Processing slice {args.slice_id}/7: videos {start_idx} to {end_idx-1} out of {total_videos} total videos")

    model = create_model(
        'pretrain_videomae_giant_patch14_224',
        pretrained=False,
        all_frames=16,
        tubelet_size=2)
    
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint
        
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Missing keys: {msg.missing_keys}") # All of these are decoder keys and thus expected
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")
    
    model.eval()
    model.cuda()

    to_process = []
    for vid_name in vid_list:
        url = os.path.join(args.save_path, '.'.join(vid_name.split('.')[:-1]) + '_mean.npy')
        if not os.path.exists(url):
            to_process.append(vid_name)
    
    if len(to_process) == 0:
        print("All features have already been extracted!")
        return
    
    print(f"Extracting features for {len(to_process)} videos (skipping {len(vid_list) - len(to_process)} already processed)")
    
    for vid_name in tqdm(to_process, desc="Extracting features", dynamic_ncols=True):
        url = os.path.join(args.save_path, '.'.join(vid_name.split('.')[:-1]) + '_mean.npy')
        video_path = os.path.join(args.data_path, vid_name)
        
        try:
            vr = video_loader(video_path)
            
            feature_list = []
            for start_idx in start_idx_range(len(vr)):
                try:
                    # Handle case where video is shorter than 16 frames
                    end_idx = min(start_idx + 16, len(vr))
                    if end_idx - start_idx < 16:
                        indices = list(range(start_idx, end_idx))
                        indices.extend([end_idx - 1] * (16 - (end_idx - start_idx)))
                        indices = np.array(indices)
                    else:
                        indices = np.arange(start_idx, start_idx + 16)
                    
                    data = vr.get_batch(indices).asnumpy()
                    frame = torch.from_numpy(data)  # torch.Size([16, H, W, 3])
                    frame_q = transform(frame)  # torch.Size([3, 16, 224, 224])
                    input_data = frame_q.unsqueeze(0).cuda()

                    with torch.no_grad():
                        # We just pass in a mask where everything is visible
                        num_patches = model.encoder.patch_embed.num_patches
                        mask = torch.zeros(1, num_patches, dtype=torch.bool, device=input_data.device)
                        
                        # Use the encoder part of the pretrained model to get features
                        feature = model.encoder.forward_features(input_data, mask)
                        feature = torch.mean(feature, dim=1)
                        feature_list.append(feature.cpu().numpy())
                except Exception as e:
                    print(f"Error processing frames {start_idx}-{start_idx+15} in {vid_name}: {str(e)}")
                    # Add a dummy feature vector
                    feature_list.append(np.zeros((1, 1408), dtype=np.float32))  
            
            if feature_list:
                np.save(url, np.vstack(feature_list))
            else:
                print(f"No features extracted for {vid_name}, saving dummy feature")
                np.save(url, np.zeros((1, 1408), dtype=np.float32))
                
        except Exception as e:
            print(f"Error loading video {vid_name}: {str(e)}")
            np.save(url, np.zeros((1, 1408), dtype=np.float32))


if __name__ == '__main__':
    args = get_args()
    extract_feature(args)


