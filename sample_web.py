import random

import numpy as np
import torch
import yaml
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


def has_keys(dic, *keys):
    for k in keys:
        if k not in dic.keys():
            return False
    return True


def sample(opt):
    assert isinstance(opt, dict)
    assert has_keys(opt, "shape", "choice")
    size = opt["shape"]
    max_per_batch = 4
    opt.update(dict(
        config=f"./config/config_{size}.yaml",
        steps=50,
        eta=0.0,
        w=0.3,
        batches=(256 // size) ** 2 // max_per_batch,
        model_path=f"./final_ckpt/model_for_{size}.pth",
    ))
    yaml_path = opt["config"]
    with open(yaml_path, 'r') as f:
        conf = yaml.full_load(f)
        conf.update(opt)
        opt = conf

    opt = Config(opt)
    # print(opt)

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

    target = opt.model_path
    print("loading model at", target)
    checkpoint = torch.load(target, map_location=device)

    ddpm.load_state_dict(checkpoint['DDPM'])

    model = ddpm
    model.eval()

    res = []
    for batch in range(opt.batches):
        with torch.no_grad():
            if isinstance(opt.choice, int) and 0 <= opt.choice < opt.n_classes:
                y = [opt.choice for i in range(max_per_batch)]
            else:
                y = [random.randint(0, opt.n_classes - 1) for i in range(max_per_batch)]
            y = np.array(y)
            assert isinstance(y, np.ndarray)
            assert y.ndim == 1
            samples_per_process = len(y)
            x_gen = model.ddim_sample(samples_per_process, (opt.in_channels, opt.shape, opt.shape),
                                      guide_w=opt.w, steps=opt.steps, eta=opt.eta, notqdm=0, n_classes=opt.n_classes,
                                      selected=y)
        res.append(x_gen)

    res = torch.cat(res)
    grid = make_grid(res, nrow=(256 // size))
    img_path = "static/TEMP/result.png"
    save_image(grid, img_path)
    # trans = ToPILImage()
    # img_np = np.array(trans(grid))
    # img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # img_res = cv2.imencode('.jpg', img_rgb)[1].tostring()
    # img_res = base64.b64encode(img_res).decode()
    # return img_res
    import datetime
    return f"{img_path}?{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


if __name__ == "__main__":
    dic = dict(
        shape=64,
        w=0.3,
        choice=68,
    )
    sample(dic)
