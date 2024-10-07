import argparse
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from glob import glob

import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


def parse_args_paired_testing(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", type=str, default=None,)
    parser.add_argument("--base_config", default="./configs/sr_test.yaml", type=str)
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--sd_path")
    parser.add_argument("--de_net_path")
    parser.add_argument("--pretrained_path", type=str, default=None,)
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)

    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--chop_size", type=int, default=128, choices=[512, 256, 128], help="Chopping forward.")
    parser.add_argument("--chop_stride", type=int, default=96, help="Chopping stride.")
    parser.add_argument("--padding_offset", type=int, default=32, help="padding offset.")

    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    parser.add_argument("--align_method", type=str, default="wavelet")
    
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")
    parser.add_argument("--neg_prompt", type=str, default="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth")

    # training details
    parser.add_argument("--output_dir", type=str, default='output/')
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class PlainDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(PlainDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        # support multiple type of data: file path and meta data, remove support of lmdb
        self.lr_paths = []
        if 'lr_path' in opt:
            if isinstance(opt['lr_path'], str):
                self.lr_paths.extend(sorted(
                    [str(x) for x in Path(opt['lr_path']).glob('*.png')] +
                    [str(x) for x in Path(opt['lr_path']).glob('*.jpg')] +
                    [str(x) for x in Path(opt['lr_path']).glob('*.jpeg')]
                ))
            else:
                self.lr_paths.extend(sorted([str(x) for x in Path(opt['lr_path'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['lr_path']) > 1:
                    for i in range(len(opt['lr_path'])-1):
                        self.lr_paths.extend(sorted([str(x) for x in Path(opt['lr_path'][i+1]).glob('*.'+opt['image_type'])]))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        lr_path = self.lr_paths[index]

        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                lr_img_bytes = self.file_client.get(lr_path, 'gt')
            except (IOError, OSError) as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                lr_path = self.lr_paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        img_lr = imfrombytes(lr_img_bytes, float32=True)
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lr = img2tensor([img_lr], bgr2rgb=True, float32=True)[0]

        return_d = {'lr': img_lr, 'lr_path': lr_path}
        return return_d

    def __len__(self):
        return len(self.lr_paths)


def lr_proc(config, batch, device):
    im_lr = batch['lr'].cuda()
    im_lr = im_lr.to(memory_format=torch.contiguous_format).float()    

    ori_lr = im_lr

    im_lr = F.interpolate(
            im_lr,
            size=(im_lr.size(-2) * config.sf,
                  im_lr.size(-1) * config.sf),
            mode='bicubic',
            )

    im_lr = im_lr.contiguous() 
    im_lr = im_lr * 2 - 1.0
    im_lr = torch.clamp(im_lr, -1.0, 1.0)

    ori_h, ori_w = im_lr.size(-2), im_lr.size(-1)

    pad_h = (math.ceil(ori_h / 64)) * 64 - ori_h
    pad_w = (math.ceil(ori_w / 64)) * 64 - ori_w
    im_lr = F.pad(im_lr, pad=(0, pad_w, 0, pad_h), mode='reflect')

    return im_lr.to(device), ori_lr.to(device), (ori_h, ori_w)
