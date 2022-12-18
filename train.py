import argparse
import os
import random

import numpy as np
import torch
import yaml
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from model.DDPM import DDPM
from model.unet import UNet
from image_datasets import get_loader


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


# ===== training =====

def up_lr(optim, eta_min, eta_max, T_cur, T_max):
    new_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(T_cur / T_max * np.pi))
    for params in optim.param_groups:
        params['lr'] = new_lr


def train(opt):
    yaml_path = opt.config

    with open(yaml_path, 'r') as f:
        conf = yaml.full_load(f)
        conf.update(vars(opt))
        opt = conf

    print(opt)
    opt = Config(opt)

    model_dir = os.path.join(opt.save_dir, "ckpts")
    vis_dir = os.path.join(opt.save_dir, "visual")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    train_loader = get_loader("./dataset/images", opt.batch_size, opt.shape, opt.flip, opt.in_channels == 1)

    # tf = [transforms.ToTensor()]
    # tf = transforms.Compose(tf)
    # train_set = CIFAR10("./CIFAR10", train=True, download=False, transform=tf)
    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=1,
    #     num_workers=0,
    # )

    lr = opt.lrate

    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)
    # sched = CosineAnnealingLR(optim, opt.n_epoch)

    if opt.load_epoch != -1 or opt.load_latest:
        if opt.load_latest:
            target = os.path.join(model_dir, f"model_latest.pth")
        else:
            target = os.path.join(model_dir, f"model_{opt.load_epoch}.pth")
        print("loading model at", target)
        checkpoint = torch.load(target, map_location=device)
        ddpm.load_state_dict(checkpoint['DDPM'])
        optim.load_state_dict(checkpoint['opt'])
        # sched.load_state_dict(checkpoint['sched'])
        if opt.load_latest:
            opt.load_epoch = checkpoint['epoch']

    for ep in range(opt.load_epoch + 1, opt.n_epoch):
        # training
        ddpm.train()
        up_lr(optim, eta_min=0., eta_max=lr, T_cur=ep, T_max=opt.n_epoch) # same as CosineAnnealingLR()
        now_lr = optim.param_groups[0]['lr']
        print(f'epoch {ep}, lr {now_lr:f}')
        pbar = tqdm(train_loader)

        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=ddpm.parameters(), max_norm=1.0)
            optim.step()

            # logging
            pbar.set_description(f"loss: {loss.item():.4f}")

        # cosine scheduler
        # sched.step()

        # testing
        if ep % opt.test_every == 0 or ep == opt.n_epoch - 1:
            ddpm.eval()
            y = c.cpu().numpy()
            samples_per_process = len(y)
            with torch.no_grad():
                x_gen = ddpm.ddim_sample(samples_per_process, x.shape[1:],
                                         guide_w=opt.w, steps=100, eta=0., n_classes=opt.n_classes, selected=y)
            # save an image of currently generated samples (top rows)
            # followed by real images (bottom rows)
            x_real = x.cpu()
            x_all = torch.cat([x_gen.cpu(), x_real])
            grid = make_grid(x_all, nrow=opt.batch_size)

            save_path = os.path.join(vis_dir, f"image_ep{ep}_w{opt.w}_y{y}.png")
            save_image(grid, save_path)
            print('saved image at', save_path)

        # optionally save model
        if opt.save_model:
            checkpoint = {
                'DDPM': ddpm.state_dict(),
                'opt': optim.state_dict(),
                # 'sched': sched.state_dict(),
                'epoch': ep,
            }

            if ep % opt.save_every == 0 or ep == opt.n_epoch - 1:
                save_path = os.path.join(model_dir, f"model_{ep}.pth")
                torch.save(checkpoint, save_path)
                print('saved model at', save_path)

            save_path = os.path.join(model_dir, f"model_latest.pth")
            torch.save(checkpoint, save_path)
            print('saved latest model at', save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_64.yaml")
    parser.add_argument("--load_latest", action="store_true", default=False)
    args = parser.parse_args()

    # init_seeds()
    train(args)
