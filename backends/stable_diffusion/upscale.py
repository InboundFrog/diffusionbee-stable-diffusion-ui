# 4x image upscaling with Real-ESRGAN x4plus (RRDBNet, BSD-3 weights) on MPS.
# `diffusionbee_backend upscale <in.png> <out.png>`; weights (67 MB safetensors) come from the HF cache.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

WEIGHTS = ("Comfy-Org/Real-ESRGAN_repackaged", "RealESRGAN_x4plus.safetensors")
TILE, PAD = 256, 16


class RDB(nn.Module):  # residual dense block
    def __init__(self, nf=64, gc=32):
        super().__init__()
        for i in range(5):
            setattr(self, f"conv{i + 1}", nn.Conv2d(nf + i * gc, gc if i < 4 else nf, 3, 1, 1))

    def forward(self, x):
        feats = [x]
        for i in range(4):
            feats.append(F.leaky_relu(getattr(self, f"conv{i + 1}")(torch.cat(feats, 1)), 0.2))
        return self.conv5(torch.cat(feats, 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = RDB(), RDB(), RDB()

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, nf=64, nb=23):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB() for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        for conv in (self.conv_up1, self.conv_up2):
            feat = F.leaky_relu(conv(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
        return self.conv_last(F.leaky_relu(self.conv_hr(feat), 0.2))


def load_model(device):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    net = RRDBNet()
    net.load_state_dict(load_file(hf_hub_download(*WEIGHTS)))
    return net.eval().to(device, torch.float16 if device == "mps" else torch.float32)


@torch.inference_mode()
def upscale(in_path, out_path):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = load_model(device)
    img = Image.open(in_path)
    alpha = img.getchannel("A") if "A" in img.getbands() else None
    x = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1)[None].to(device, next(net.parameters()).dtype) / 255
    _, _, h, w = x.shape
    y = torch.zeros(1, 3, h * 4, w * 4, device=device, dtype=x.dtype)
    # ponytail: fixed 256px tiles with 16px overlap bound memory; seams are hidden by the overlap crop, no blending
    for top in range(0, h, TILE):
        for left in range(0, w, TILE):
            t0, l0 = max(top - PAD, 0), max(left - PAD, 0)
            t1, l1 = min(top + TILE + PAD, h), min(left + TILE + PAD, w)
            tile = net(x[:, :, t0:t1, l0:l1])
            bh, bw = min(TILE, h - top), min(TILE, w - left)
            y[:, :, top * 4:(top + bh) * 4, left * 4:(left + bw) * 4] = \
                tile[:, :, (top - t0) * 4:(top - t0 + bh) * 4, (left - l0) * 4:(left - l0 + bw) * 4]
    arr = (y[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    res = Image.fromarray(arr)
    if alpha is not None:
        res.putalpha(alpha.resize(res.size, Image.LANCZOS))
    res.save(out_path)
