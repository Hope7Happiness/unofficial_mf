from models.dit import MFDiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from meanflow import MeanFlow
from accelerate import Accelerator
import time
import os
from PIL import Image
from pathlib import Path

import wandb

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=5)
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()


    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=1,
        # in_channels=int('cifar' in dataset.__class__.__name__.lower())*2+1,
        dim=256,
        # dim=384,
        depth=8,
        # depth=12,
        num_heads=8,
        # num_heads=6,
        num_classes=10,
    )
    
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=True), strict=True)
    
    model.cuda()

    meanflow = MeanFlow(channels=1,
                        image_size=32,
                        num_classes=10,
                        flow_ratio=0.50,
                        time_dist=['lognorm', -0.4, 1.0],
                        cfg_ratio=0.10,
                        cfg_scale=2.0,
                        # experimental
                        cfg_uncond='u')
    
    model_module = model.module if hasattr(model, 'module') else model
    z = meanflow.sample_each_class(model_module, 10)
    # z = meanflow.sample_uncond(model_module, 10, sample_steps=args.n_steps)
    
    log_img = make_grid(z, nrow=10)
    img_save_path = Path('exps') / 'inference_outputs' / f'samples_{time.strftime("%Y%m%d_%H%M%S")}.png'
    img_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_image(log_img, img_save_path)
    print(f'saved image to {img_save_path}')