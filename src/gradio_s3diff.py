import gradio as gr
import os
import sys
import math
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.utils.import_utils import is_xformers_available

from my_utils.testing_utils import parse_args_paired_testing
from de_net import DEResNet
from s3diff_tile import S3Diff
from torchvision import transforms
from utils.wavelet_color import wavelet_color_fix, adain_color_fix

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

args = parse_args_paired_testing()

# Load scheduler, tokenizer and models.
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

de_net_path = 'assets/mm-realsr/de_net.pth'

# initialize net_sr
net_sr = S3Diff(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, sd_path=sd_path, pretrained_path=pretrained_path, args=args)
net_sr.set_eval()

# initalize degradation estimation network
net_de = DEResNet(num_in_ch=3, num_degradation=2)
net_de.load_model(de_net_path)
net_de = net_de.cuda()
net_de.eval()

if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
        net_sr.unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

if args.gradient_checkpointing:
    net_sr.unet.enable_gradient_checkpointing()

weight_dtype = torch.float32
device = "cuda"

# Move text_encode and vae to gpu and cast to weight_dtype
net_sr.to(device, dtype=weight_dtype)
net_de.to(device, dtype=weight_dtype)

@torch.no_grad()
def process(
    input_image: Image.Image,
    scale_factor: float,
    cfg_scale: float,
    latent_tiled_size: int,
    latent_tiled_overlap: int,
    align_method: str,
    ) -> List[np.ndarray]:

    # positive_prompt = ""
    # negative_prompt = ""

    net_sr._set_latent_tile(latent_tiled_size = latent_tiled_size, latent_tiled_overlap = latent_tiled_overlap)

    im_lr = tensor_transforms(input_image).unsqueeze(0).to(device)
    ori_h, ori_w = im_lr.shape[2:]
    im_lr_resize = F.interpolate(
        im_lr,
        size=(int(ori_h * scale_factor),
              int(ori_w * scale_factor)),
        mode='bilinear',
        align_corners=True
        )
    im_lr_resize = im_lr_resize.contiguous() 
    im_lr_resize_norm = im_lr_resize * 2 - 1.0
    im_lr_resize_norm = torch.clamp(im_lr_resize_norm, -1.0, 1.0)
    resize_h, resize_w = im_lr_resize_norm.shape[2:]

    pad_h = (math.ceil(resize_h / 64)) * 64 - resize_h
    pad_w = (math.ceil(resize_w / 64)) * 64 - resize_w
    im_lr_resize_norm = F.pad(im_lr_resize_norm, pad=(0, pad_w, 0, pad_h), mode='reflect')
      
    try:
        with torch.autocast("cuda"):
            deg_score = net_de(im_lr)

            pos_tag_prompt = [args.pos_prompt]
            neg_tag_prompt = [args.neg_prompt]

            x_tgt_pred = net_sr(im_lr_resize_norm, deg_score, pos_prompt=pos_tag_prompt, neg_prompt=neg_tag_prompt)
            x_tgt_pred = x_tgt_pred[:, :, :resize_h, :resize_w]
            out_img = (x_tgt_pred * 0.5 + 0.5).cpu().detach()

        output_pil = transforms.ToPILImage()(out_img[0])

        if align_method == 'no fix':
            image = output_pil
        else:
            im_lr_resize = transforms.ToPILImage()(im_lr_resize[0])
            if align_method == 'wavelet':
                image = wavelet_color_fix(output_pil, im_lr_resize)
            elif align_method == 'adain':
                image = adain_color_fix(output_pil, im_lr_resize)

    except Exception as e:
        print(e)
        image = Image.new(mode="RGB", size=(512, 512))

    return image


#
MARKDOWN = \
"""
## Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors

[GitHub](https://github.com/ArcticHare105/S3Diff) | [Paper](https://arxiv.org/abs/2409.17058)

If S3Diff is helpful for you, please help star the GitHub Repo. Thanks!
"""

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", type="pil")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Options", open=True):
                cfg_scale = gr.Slider(label="Classifier Free Guidance Scale (Set a value larger than 1 to enable it!)", minimum=1.0, maximum=1.1, value=1.07, step=0.01)
                scale_factor = gr.Number(label="SR Scale", value=4)
                latent_tiled_size = gr.Slider(label="Tile Size", minimum=64, maximum=160, value=96, step=1)
                latent_tiled_overlap = gr.Slider(label="Tile Overlap", minimum=16, maximum=48, value=32, step=1)
                align_method = gr.Dropdown(label="Color Correction", choices=["wavelet", "adain", "no fix"], value="wavelet")
        with gr.Column():
            result_image = gr.Image(label="Output", show_label=False, elem_id="result_image", source="canvas", width="100%", height="auto")

    inputs = [
        input_image,
        scale_factor,
        cfg_scale,
        latent_tiled_size,
        latent_tiled_overlap,
        align_method
    ]
    run_button.click(fn=process, inputs=inputs, outputs=[result_image])

block.launch()

