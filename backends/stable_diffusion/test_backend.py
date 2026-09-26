"""Backend tests. Run: backends/.venv/bin/python -m pytest -q backends/stable_diffusion/test_backend.py
The end-to-end tests drive the real stdin/stdout protocol with hf-internal-testing/tiny-sdxl-pipe (11 MB, downloaded once)."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, "diffusionbee_backend.py")
sys.path.insert(0, HERE)
import diffusionbee_backend as db  # noqa: E402  (torch/diffusers are only imported when needed)

TMP = tempfile.mkdtemp()


class Backend:
    def __init__(self):
        self.p = subprocess.Popen([sys.executable, BACKEND], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self.read_until("sdbk inrd")

    def read_until(self, prefix):
        lines = []
        for line in self.p.stdout:
            lines.append(line.rstrip("\n"))
            if lines[-1].startswith(prefix):
                return lines
        raise AssertionError("backend exited:\n" + "\n".join(lines[-20:]))

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def job(self, **d):
        self.send("b2py t2im " + json.dumps(d))
        lines = self.read_until("sdbk inrd")
        imgs = [json.loads(x[len("sdbk nwim "):]) for x in lines if x.startswith("sdbk nwim ")]
        errs = [x[len("sdbk errr "):] for x in lines if x.startswith("sdbk errr ")]
        return imgs, errs, lines

    def close(self):
        self.p.stdin.close()
        assert self.p.wait(30) == 0


def run_cmd(*args):
    r = subprocess.run([sys.executable, BACKEND, *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.splitlines()


def st_file(name, tensors):
    path = os.path.join(TMP, name)
    save_file({k: np.zeros(shape, np.float16) for k, shape in tensors.items()}, path)
    return path


def test_family_detection():
    assert db.family_from_class("StableDiffusionPipeline") == "sd15"
    assert db.family_from_class("StableDiffusionInpaintPipeline") == "sd15"
    assert db.family_from_class("StableDiffusionXLPipeline") == "sdxl"
    assert db.family_from_class("StableDiffusion3Pipeline") == "sd3"
    assert db.family_from_class("FluxPipeline") == "flux"
    assert db.family_from_class("Flux2KleinPipeline") == "flux2"
    assert db.family_from_class("ZImagePipeline") == "zimage"
    assert db.family_from_class("QwenImagePipeline") == "qwenimage"
    assert db.family_from_class("KandinskyPipeline") is None
    assert db.family_from_type("xl_inpaint") == "sdxl" and db.family_from_type("sd35_medium") == "sd3"
    assert db.family_from_type("flux-2-dev") == "flux2" and db.family_from_type("flux-schnell") == "flux"
    assert db.family_from_type("z-image-turbo") == "zimage" and db.family_from_type("inpainting") == "sd15"

    lora_k = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight"
    assert db.inspect_model(st_file("l15.safetensors", {lora_k: (4, 768)})) == {"family": "sd15", "is_inpaint": False, "type": "lora"}
    peft_k = "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight"
    assert db.inspect_model(st_file("lxl.safetensors", {peft_k: (4, 2048)}))["family"] == "sdxl"
    v1 = "model.diffusion_model.output_blocks.11.0.skip_connection.weight"
    assert db.inspect_model(st_file("v1.safetensors", {v1: (320, 640)})) == {"family": "sd15", "is_inpaint": False, "type": "sd_model"}
    inp = {v1: (320, 640), "model.diffusion_model.input_blocks.0.0.weight": (320, 9, 3, 3)}
    assert db.inspect_model(st_file("inp.safetensors", inp))["is_inpaint"] is True
    xl = {"conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias": (1280,)}
    assert db.inspect_model(st_file("xl.safetensors", xl))["family"] == "sdxl"
    assert db.inspect_model(st_file("junk.safetensors", {"foo": (1,)}))["family"] is None


def test_pick_files():
    sd15 = ["model_index.json", "v1-5-pruned.safetensors", "v1-5-pruned.ckpt", "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors",
            "unet/diffusion_pytorch_model.non_ema.safetensors", "unet/diffusion_pytorch_model.fp16.bin",
            "safety_checker/model.safetensors", "safety_checker/config.json", "tokenizer/vocab.json",
            "text_encoder/model.safetensors", "README.md"]
    got = set(db.pick_files(sd15, ["unet", "text_encoder", "tokenizer"], "fp16"))
    assert got == {"model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
                   "tokenizer/vocab.json", "text_encoder/model.safetensors"}
    sharded = ["text_encoder_3/model-00001-of-00002.safetensors", "text_encoder_3/model.fp16-00001-of-00002.safetensors"]
    assert db.pick_files(sharded, ["text_encoder_3"], "fp16") == ["text_encoder_3/model.fp16-00001-of-00002.safetensors"]
    assert db.pick_files(sharded, ["text_encoder_3"], None) == ["text_encoder_3/model-00001-of-00002.safetensors"]
    tile = ["config.json", "diffusion_pytorch_model.bin", "images/a.png"]
    assert db.pick_files(tile, None, "fp16") == ["config.json", "diffusion_pytorch_model.bin"]
    cn = ["config.json", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors",
          "diffusion_pytorch_model.fp16.safetensors"]
    assert db.pick_files(cn, None, "fp16") == ["config.json", "diffusion_pytorch_model.fp16.safetensors"]


def test_refuses_unsafe_and_legacy_models():
    for bad in ("model.tdict", "model.ckpt", "model.bin"):
        try:
            db.check_path(os.path.join(TMP, bad))
            raise AssertionError("accepted " + bad)
        except ValueError as e:
            assert ("TDict" in str(e)) == bad.endswith(".tdict")


def test_offline_uses_cached_snapshot():
    # download_model pins revision=<sha>, which leaves no refs/main; offline it must still find the newest snapshot
    hf = tempfile.mkdtemp()
    snaps = os.path.join(hf, "hub", "models--org--model", "snapshots")
    for t, rev in ((1000, "old"), (2000, "new")):
        os.makedirs(os.path.join(snaps, rev))
        os.utime(os.path.join(snaps, rev), (t, t))
    r = subprocess.run([sys.executable, BACKEND, "download_model", "org/model"], capture_output=True, text=True,
                       env=dict(os.environ, HF_HOME=hf, HF_HUB_OFFLINE="1"))
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.splitlines()[-1] == "done " + os.path.join(snaps, "new")


def test_end_to_end():
    lines = run_cmd("download_model", "hf-internal-testing/tiny-sdxl-pipe")
    model = lines[-1][len("done "):]
    assert lines[-1].startswith("done ") and "/snapshots/" in model and os.path.isdir(model)
    assert json.loads(run_cmd("inspect_model", model)[-1]) == {"family": "sdxl", "is_inpaint": False, "type": "sd_model"}

    red = os.path.join(TMP, "red.png")
    Image.new("RGB", (256, 192), (255, 0, 0)).save(red)
    mask = os.path.join(TMP, "mask.png")
    m = Image.new("RGB", (256, 192))
    m.paste((255, 255, 255), (96, 64, 160, 128))
    m.save(mask)

    b = Backend()
    try:
        _, errs, _ = b.job(prompt="x", model_tdict_path="/old/model.tdict")
        assert errs and "TDict" in errs[0], errs

        common = dict(prompt="a cat", model_path=model, num_steps=3, guidance_scale=5, seed=7)
        imgs, errs, lines = b.job(**common, img_width=128, img_height=96, num_imgs=2)
        assert not errs and len(imgs) == 2, lines[-20:]
        assert [i["seed"] for i in imgs] == [7, 7 + 1234]
        assert Image.open(imgs[0]["generated_img_path"]).size == (128, 96)
        assert any(x.startswith("sdbk dnpr 100") for x in lines)

        imgs, errs, lines = b.job(**common, input_img=red, input_image_strength=40, force_use_given_size=True,
                                  img_width=128, img_height=96)
        assert not errs and len(imgs) == 1, lines[-20:]
        assert Image.open(imgs[0]["generated_img_path"]).size == (128, 96)

        imgs, errs, lines = b.job(**common, input_img=red, mask_image=mask, force_use_given_size=True,
                                  img_width=256, img_height=192, sd_mode_override="txt2img")
        assert not errs and len(imgs) == 1, lines[-20:]
        out = Image.open(imgs[0]["generated_img_path"]).convert("RGB")
        assert out.getpixel((5, 5)) == (255, 0, 0)  # outside the mask: original pixels kept
        assert out.getpixel((128, 96)) != (255, 0, 0)  # inside: repainted

        _, errs, _ = b.job(**common, input_img=red, get_mask_from_image_alpha=True, sd_mode_override="txt2img")
        assert errs and "mask is empty" in errs[0], errs

        # stop: ask for many images, stop right away
        b.send("b2py t2im " + json.dumps(dict(common, num_steps=20, num_imgs=50, img_width=128, img_height=128)))
        b.read_until("sdbk inwk")
        b.send("b2py t2im __stop__")
        lines = b.read_until("sdbk inrd")
        assert sum(x.startswith("sdbk nwim") for x in lines) < 50
    finally:
        b.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_upscale():
    # 300x200 crosses the 256px tile boundary in both directions; Real-ESRGAN weights are 67 MB, downloaded once
    src, dst = os.path.join(TMP, "up_in.png"), os.path.join(TMP, "up_out.png")
    grad = np.linspace(0, 255, 300, dtype=np.uint8)
    Image.fromarray(np.stack([np.tile(grad, (200, 1))] * 3, -1)).save(src)
    r = subprocess.run([sys.executable, BACKEND, "upscale", src, dst], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "done " + dst, r.stderr[-2000:]
    out = np.asarray(Image.open(dst).convert("L"), dtype=np.float32)
    assert out.shape == (800, 1200)
    # a smooth gradient must stay smooth across the tile seam at x=1024 (no stitching offset)
    assert abs(out[:, 1023].mean() - out[:, 1024].mean()) < 4
    assert abs(out[:, 600].mean() - 127) < 12
