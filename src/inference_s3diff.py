import os
import gc
import tqdm
import math
import lpips
import pyiqa
import argparse
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
# from tqdm.auto import tqdm

import diffusers
import utils.misc as misc

from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from de_net import DEResNet
from s3diff_tile import S3Diff
from my_utils.testing_utils import parse_args_paired_testing, PlainDataset, lr_proc
from utils.util_image import ImageSpliterTh
from my_utils.utils import instantiate_from_config
from pathlib import Path
from utils import util_image
from utils.wavelet_color import wavelet_color_fix, adain_color_fix

def evaluate(in_path, ref_path, ntest):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    metric_dict["niqe"] = pyiqa.create_metric('niqe').to(device)
    metric_dict["maniqa"] = pyiqa.create_metric('maniqa').to(device)
    metric_paired_dict = {}
    
    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    assert in_path.is_dir()
    
    ref_path_list = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_path_list = sorted([x for x in ref_path.glob("*.[jpJP][pnPN]*[gG]")])
        if ntest is not None: ref_path_list = ref_path_list[:ntest]
        
        metric_paired_dict["psnr"]=pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        metric_paired_dict["lpips"]=pyiqa.create_metric('lpips').to(device)
        metric_paired_dict["dists"]=pyiqa.create_metric('dists').to(device)
        metric_paired_dict["ssim"]=pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr' ).to(device)
        
    lr_path_list = sorted([x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")])
    if ntest is not None: lr_path_list = lr_path_list[:ntest]
    
    print(f'Find {len(lr_path_list)} images in {in_path}')
    result = {}
    for i in tqdm.tqdm(range(len(lr_path_list))):
        _in_path = lr_path_list[i]
        _ref_path = ref_path_list[i] if ref_path_list is not None else None
        
        im_in = util_image.imread(_in_path, chn='rgb', dtype='float32')  # h x w x c
        im_in_tensor = util_image.img2tensor(im_in).cuda()              # 1 x c x h x w
        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                result[key] = result.get(key, 0) + metric(im_in_tensor).item()
        
        if ref_path is not None:
            im_ref = util_image.imread(_ref_path, chn='rgb', dtype='float32')  # h x w x c
            im_ref_tensor = util_image.img2tensor(im_ref).cuda()    
            for key, metric in metric_paired_dict.items():
                result[key] = result.get(key, 0) + metric(im_in_tensor, im_ref_tensor).item()
    
    if ref_path is not None:
        fid_metric = pyiqa.create_metric('fid')
        result['fid'] = fid_metric(in_path, ref_path)

    print_results = []
    for key, res in result.items():
        if key == 'fid':
            print(f"{key}: {res:.2f}")
            print_results.append(f"{key}: {res:.2f}")
        else:
            print(f"{key}: {res/len(lr_path_list):.5f}")
            print_results.append(f"{key}: {res/len(lr_path_list):.5f}")
    return print_results


def main(args):
    config = OmegaConf.load(args.base_config)

    if args.pretrained_path is None:
        from huggingface_hub import hf_hub_download
        pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")
    else:
        pretrained_path = args.pretrained_path

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # initialize net_sr
    net_sr = S3Diff(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, sd_path=sd_path, pretrained_path=pretrained_path, args=args)
    net_sr.set_eval()

    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.cuda()
    net_de.eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_val = PlainDataset(config.validation)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Prepare everything with our `accelerator`.
    net_sr, net_de = accelerator.prepare(net_sr, net_de)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_de.to(accelerator.device, dtype=weight_dtype)
      
    offset = args.padding_offset
    for step, batch_val in enumerate(dl_val):
        lr_path = batch_val['lr_path'][0]
        (path, name) = os.path.split(lr_path)

        im_lr = batch_val['lr'].cuda()
        im_lr = im_lr.to(memory_format=torch.contiguous_format).float()    

        ori_h, ori_w = im_lr.shape[2:]
        im_lr_resize = F.interpolate(
            im_lr,
            size=(ori_h * config.sf,
                  ori_w * config.sf),
            mode='bilinear',
            align_corners=False # align_corners with this model causes the output to be shifted, presumably due to training without align_corners
            )

        im_lr_resize = im_lr_resize.contiguous() 
        im_lr_resize_norm = im_lr_resize * 2 - 1.0
        im_lr_resize_norm = torch.clamp(im_lr_resize_norm, -1.0, 1.0)
        resize_h, resize_w = im_lr_resize_norm.shape[2:]

        pad_h = (math.ceil(resize_h / 64)) * 64 - resize_h
        pad_w = (math.ceil(resize_w / 64)) * 64 - resize_w
        im_lr_resize_norm = F.pad(im_lr_resize_norm, pad=(0, pad_w, 0, pad_h), mode='reflect')
        
        B = im_lr_resize.size(0)
        with torch.no_grad():
            # forward pass
            deg_score = net_de(im_lr)
            pos_tag_prompt = [args.pos_prompt for _ in range(B)]
            neg_tag_prompt = [args.neg_prompt for _ in range(B)]
            x_tgt_pred = accelerator.unwrap_model(net_sr)(im_lr_resize_norm, deg_score, pos_prompt=pos_tag_prompt, neg_prompt=neg_tag_prompt)
            x_tgt_pred = x_tgt_pred[:, :, :resize_h, :resize_w]
            out_img = (x_tgt_pred * 0.5 + 0.5).cpu().detach()

        output_pil = transforms.ToPILImage()(out_img[0])

        if args.align_method == 'nofix':
            output_pil = output_pil
        else:
            im_lr_resize = transforms.ToPILImage()(im_lr_resize[0].cpu().detach())
            if args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(output_pil, im_lr_resize)
            elif args.align_method == 'adain':
                output_pil = adain_color_fix(output_pil, im_lr_resize)

        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.output_dir, fname+'.png')
        output_pil.save(outf)

    print_results = evaluate(args.output_dir, args.ref_path, None)
    out_t = os.path.join(args.output_dir, 'results.txt')
    with open(out_t, 'w', encoding='utf-8') as f:
        for item in print_results:
            f.write(f"{item}\n")

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args_paired_testing()
    main(args)
