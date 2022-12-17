import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid, save_image

from model.DDPM import DDPM
from model.unet import UNet


# ===== Config yaml files (helper functions)

class Config(object):
    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])


def init_seeds(RANDOM_SEED=1337):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ===== sampling =====

def sample(opt):
    yaml_path = opt.config
    with open(yaml_path, 'r') as f:
        conf = yaml.full_load(f)
        opt = vars(opt)
        opt.update(conf)

    opt = Config(opt)
    print(opt)
    mode = opt.mode
    steps = opt.steps
    eta = opt.eta
    batches = opt.batches
    w = opt.w

    # ep = opt.n_epoch - 1
    ep = 199


    device = "cuda"
    ddpm = DDPM(nn_model=UNet(image_channels=opt.in_channels,
                              n_channels=opt.n_channels,
                              ch_mults=opt.ch_mults,
                              is_attn=opt.is_attn,
                              dropout=opt.dropout,
                              n_blocks=opt.n_blocks,
                              use_res_for_updown=opt.biggan,
                              n_classes=opt.n_classes),
                betas=opt.betas,
                n_T=opt.n_T,
                device=device,
                drop_prob=0.1)
    ddpm.to(device)

    # target = os.path.join(opt.save_dir, "ckpts", f"model_{ep}.pth")
    target = "./final_ckpt/model_for_64.pth"
    print("loading model at", target)
    checkpoint = torch.load(target, map_location=device)

    ddpm.load_state_dict(checkpoint['DDPM'])



    model = ddpm
    import datetime
    prefix = os.path.join("sample", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    model.eval()

    if mode == 'DDPM':
        gen_dir = os.path.join(opt.save_dir, f"{prefix}_ep{ep}_w{w}_ddpm")
    elif mode == 'DDIM':
        gen_dir = os.path.join(opt.save_dir, f"{prefix}_ep{ep}_w{w}_ddim_steps{steps}_eta{eta}")
    else:
        raise NotImplementedError()
    os.makedirs(gen_dir, exist_ok=True)
    gen_dir_png = os.path.join(gen_dir, "pngs")
    os.makedirs(gen_dir_png, exist_ok=True) 
    res = []
    print(opt.n_classes)
    for batch in range(batches):
        with torch.no_grad():
            y = [random.randint(0, opt.n_classes - 1) for i in range(16)]
            # y = [68 for i in range(4)]
            y = np.array(y)
            assert isinstance(y, np.ndarray)
            assert y.ndim == 1
            samples_per_process = len(y)
            if mode == 'DDPM':
                x_gen = model.sample(samples_per_process, (opt.in_channels, opt.shape, opt.shape),
                                     guide_w=w, notqdm=0, n_classes=opt.n_classes, selected=y)
            else:
                x_gen = model.ddim_sample(samples_per_process, (opt.in_channels, opt.shape, opt.shape),
                                          guide_w=w, steps=steps, eta=eta, notqdm=0, n_classes=opt.n_classes,
                                          selected=y)
        res.append(x_gen)
        grid = make_grid(x_gen.cpu(), nrow=4)
        png_path = os.path.join(gen_dir, f"grid_{batch}_y{y}.png")
        save_image(grid, png_path)

    res = torch.cat(res)
    for no, img in enumerate(res):
        png_path = os.path.join(gen_dir_png, f"{no}.png")
        save_image(img, png_path)


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config_64.yaml")
    parser.add_argument("--mode", type=str, choices=['DDPM', 'DDIM'], default='DDIM')
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument("--ema", action='store_true', default=False)
    parser.add_argument("--w", type=float, default=0.3)
    opt = parser.parse_args()

    # init_seeds(5465)
    sample(opt)
