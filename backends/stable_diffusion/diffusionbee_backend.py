"""DiffusionBee backend: PyTorch (MPS) + Hugging Face diffusers.

No args: stdin/stdout job loop (docs/backend_protocol.md).
`download_model <repo_id> [--variant fp16]`: fetch the files a pipeline loads into the HF cache.
`inspect_model <path>`: print {"family", "is_inpaint", "type"} for a .safetensors file or diffusers folder.
"""
import os
import sys
from pathlib import Path

DB_HOME = Path.home() / ".diffusionbee"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before torch is imported
os.environ.setdefault("HF_HOME", str(DB_HOME / "hf"))

import gc
import glob
import inspect
import json
import math
import queue
import random
import struct
import threading
import traceback
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps

IMAGES_DIR = DB_HOME / "images"

# family -> (default size, steps, cfg); used when the job leaves them out
FAMILIES = {
    "sd15": (512, 25, 7.5), "sdxl": (1024, 30, 6.0), "sd3": (1024, 40, 4.5), "flux": (1024, 28, 3.5),
    "flux2": (1024, 4, 1.0), "zimage": (1024, 8, 0.0), "qwenimage": (1328, 50, 4.0),
}
DIT_FAMILIES = {"sd3", "flux", "flux2", "zimage", "qwenimage"}  # loaded in bf16
# ponytail: diffusers bf16 peak (GB) at 1024², measured on an M4 Pro (docs/mlx_benchmark_notes.md). Above 75% of RAM
# the family runs on the MLX engine instead. Unlisted families always use diffusers; measure one to add it.
DIT_PEAK_GB = {"zimage": 30.5, "flux2": 23}
CLASS_FAMILIES = [("StableDiffusionXL", "sdxl"), ("StableDiffusion3", "sd3"), ("StableDiffusion", "sd15"),
                  ("Flux2", "flux2"), ("Flux", "flux"), ("ZImage", "zimage"), ("QwenImage", "qwenimage")]
# single-file checkpoints: family -> (pipeline class, inpaint pipeline class)
SINGLE_FILE = {"sd15": ("StableDiffusionPipeline", "StableDiffusionInpaintPipeline"),
               "sdxl": ("StableDiffusionXLPipeline", "StableDiffusionXLInpaintPipeline"),
               "sd3": ("StableDiffusion3Pipeline",) * 2, "flux": ("FluxPipeline",) * 2,
               "zimage": ("ZImagePipeline",) * 2}
CROSS_ATTN_FAMILIES = {768: "sd15", 1024: "sd15", 2048: "sdxl"}  # UNet cross-attention width -> family
SCHEDULERS = {  # UI name -> diffusers scheduler (UNet families only; flow-matching models keep theirs)
    "karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "ddim": ("DDIMScheduler", {}),
    "lmsd": ("EulerDiscreteScheduler", {}),  # LMSDiscreteScheduler needs scipy, which we don't ship
    "pndm": ("PNDMScheduler", {"skip_prk_steps": True}),
    "k_euler_ancestral": ("EulerAncestralDiscreteScheduler", {}),
    "k_euler": ("EulerDiscreteScheduler", {}),
    "lcm": ("LCMScheduler", {}),  # for LCM-LoRA: 4-8 steps, cfg 1-2
}
PICKLE_EXTS = (".ckpt", ".pt", ".pth", ".bin", ".pkl", ".pickle")
CONFIG_EXTS = (".json", ".txt", ".model", ".jinja", ".tiktoken")  # configs, tokenizers, templates


def out(line):
    print(line, flush=True)


def ram_gb():
    """Physical RAM, or DIFFUSIONBEE_RAM_GB: picks the engine as if the Mac had that much (test the tiers, or override)."""
    try:
        return float(os.environ["DIFFUSIONBEE_RAM_GB"])
    except (KeyError, ValueError):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3


def one_line(e):
    return " ".join(str(e).split())[:500] or type(e).__name__


# ---------------------------------------------------------------- model files

def check_path(path):
    low = path.lower()
    if low.endswith(".tdict"):
        raise ValueError("TDict models from DiffusionBee 1.x/2.x are no longer supported. "
                         "Re-import the original .safetensors file.")
    if low.endswith(PICKLE_EXTS):
        raise ValueError("Only .safetensors files or diffusers folders are supported "
                         "(.ckpt and other pickle files can run code and are refused).")
    if not os.path.exists(path):
        raise ValueError(f"Model not found: {path}")
    return path


def family_from_class(name):
    return next((f for prefix, f in CLASS_FAMILIES if name.startswith(prefix)), None)


def family_from_type(t):
    if t.startswith(("xl_", "controlnet_xl", "playground")):
        return "sdxl"
    for prefix, family in (("sd3", "sd3"), ("flux-2", "flux2"), ("flux", "flux"), ("z-image", "zimage")):
        if t.startswith(prefix):
            return family
    return "sd15" if t in ("v1", "v2", "inpainting", "inpainting_v2", "controlnet") else None


def safetensors_shapes(path):
    """Tensor shapes from the .safetensors header, without reading the weights."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 200_000_000:
            raise ValueError("Not a valid .safetensors file")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return {k: SimpleNamespace(shape=tuple(v["shape"])) for k, v in header.items()}


def inspect_model(path):
    if os.path.isdir(path):
        index_file = os.path.join(path, "model_index.json")
        if os.path.exists(index_file):
            cls = json.load(open(index_file)).get("_class_name", "")
            unet_cfg = os.path.join(path, "unet", "config.json")
            in_ch = json.load(open(unet_cfg)).get("in_channels") if os.path.exists(unet_cfg) else None
            return {"family": family_from_class(cls), "is_inpaint": "Inpaint" in cls or in_ch == 9, "type": "sd_model"}
        cfg_file = os.path.join(path, "config.json")
        cfg = json.load(open(cfg_file)) if os.path.exists(cfg_file) else {}
        if "ControlNet" in cfg.get("_class_name", ""):
            family = CROSS_ATTN_FAMILIES.get(cfg.get("cross_attention_dim"))
            return {"family": family, "is_inpaint": False, "type": "controlnet"}
        raise ValueError("Not a diffusers model folder (needs model_index.json, or config.json of a ControlNet)")

    shapes = safetensors_shapes(check_path(path))
    if any("lora" in k for k in shapes):
        dims = [v.shape[-1] for k, v in shapes.items()
                if "attn2" in k and "to_k" in k and ("lora_down" in k or "lora_A" in k)]
        return {"family": CROSS_ATTN_FAMILIES.get(dims[0]) if dims else None, "is_inpaint": False, "type": "lora"}
    from diffusers.loaders.single_file_utils import CHECKPOINT_KEY_NAMES, infer_diffusers_model_type
    t = infer_diffusers_model_type(shapes)
    family = family_from_type(t)
    if t == "v1" and CHECKPOINT_KEY_NAMES["v1"] not in shapes:  # diffusers falls back to "v1" for anything unknown
        family = None
    return {"family": family, "is_inpaint": t in ("inpainting", "inpainting_v2", "xl_inpaint"),
            "type": "controlnet" if t.startswith("controlnet") else "sd_model"}


def pick_files(files, components, variant):
    """The repo files a diffusers pipeline actually loads: configs plus one set of .safetensors per component
    folder (variant preferred). Skips root single-file checkpoints, .bin/.ckpt copies, onnx, non_ema, etc.
    components=None means the repo root is the model (ControlNet repos)."""
    comps = {""} if components is None else set(components)
    keep, weights = [], {}
    for f in files:
        folder, _, name = f.rpartition("/")
        if name.endswith(CONFIG_EXTS) and (folder in comps or folder == ""):
            keep.append(f)
        elif folder in comps and name.endswith(".safetensors"):
            weights.setdefault(folder, []).append(f)
    for folder in comps:
        ws = weights.get(folder, [])
        if not ws and components is None:
            # e.g. lllyasviel/control_v11f1e_sd15_tile only ships a .bin; diffusers reads it with
            # torch.load(weights_only=True), which cannot run pickled code
            ws = [f for f in files if "/" not in f and f.endswith(".bin")]
        chosen = [f for f in ws if variant and (f".{variant}." in f or f".{variant}-" in f)]
        keep += chosen or [f for f in ws if os.path.basename(f).count(".") == 1]
    return keep


def fetch_repo(repo, variant="fp16", progress=None):
    """Download what the pipeline needs into the HF cache; returns the snapshot folder."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download, try_to_load_from_cache
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    from tqdm import tqdm

    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except RepositoryNotFoundError:
        raise ValueError(f"Model repo not found (or gated without a token): {repo}")
    except Exception:  # offline: use the newest cached snapshot. Not snapshot_download(local_files_only=True):
        # downloads below pin revision=sha, which writes no refs/main for it to resolve
        from huggingface_hub.constants import HF_HUB_CACHE
        snaps = glob.glob(os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "snapshots", "*"))
        if not snaps:
            raise
        return max(snaps, key=os.path.getmtime)
    try:
        sizes = {s.rfilename: s.size or 0 for s in info.siblings}
        components = None
        if "model_index.json" in sizes:
            index = json.load(open(hf_hub_download(repo, "model_index.json", revision=info.sha)))
            components = [k for k, v in index.items() if isinstance(v, list) and v[0] and k != "safety_checker"]
        files = pick_files(sizes, components, variant)
        total = sum(sizes[f] for f in files) or 1
        cached = sum(sizes[f] for f in files
                     if isinstance(try_to_load_from_cache(repo, f, revision=info.sha), str))
        done = {}

        class Bar(tqdm):  # snapshot_download reports aggregated byte counts through tqdm_class
            def __init__(self, *args, **kwargs):
                kwargs["file"] = open(os.devnull, "w")
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                super().update(n)
                if self.unit == "B" and progress:
                    done[id(self)] = self.n
                    progress(min(99, int(100 * (cached + max(done.values())) / total)))

        path = snapshot_download(repo, revision=info.sha, allow_patterns=files, tqdm_class=Bar)
    except GatedRepoError:
        raise ValueError(f"{repo} is gated: accept its license on huggingface.co and set a Hugging Face token")
    if progress:
        progress(100)
    return path


def model_source(d, key):
    """Local path for job field <key>_path (or <key>_repo, downloaded on demand)."""
    path = d.get(key + "_path") or (d.get("model_tdict_path") if key == "model" else None)
    if path:
        return check_path(path)
    if d.get(key + "_repo"):
        return fetch_repo(d[key + "_repo"], progress=lambda p: out(f"sdbk gnms Downloading model {p}%"))
    return None


# ---------------------------------------------------------------- stdin

_stdin = queue.Queue()


def _read_stdin():
    for line in sys.stdin:
        _stdin.put(line.strip())
    _stdin.put(None)


def stop_requested():
    """Drain stdin looking for __stop__. Other lines that arrive mid-job are dropped (as in v2)."""
    try:
        while True:
            line = _stdin.get_nowait()
            if line is None:  # app went away: stop now, main loop exits
                _stdin.put(None)
                return True
            if "__stop__" in line:
                return True
    except queue.Empty:
        return False


# ---------------------------------------------------------------- images

def fit(img, size):
    return ImageOps.fit(img, size, Image.LANCZOS)


def job_size(d, family, img):
    default = FAMILIES.get(family, (1024,))[0]
    w, h = d.get("img_width"), d.get("img_height")
    if img is not None and not (d.get("force_use_given_size") and w and h):
        scale = default / math.sqrt(img.size[0] * img.size[1])  # input aspect at the model's native area
        w, h = img.size[0] * scale, img.size[1] * scale
    w, h = float(w or default), float(h or default)
    s = min(1.0, 2048 / math.sqrt(w * h))  # cap at 2048x2048 worth of pixels
    return max(64, int(w * s) // 16 * 16), max(64, int(h * s) // 16 * 16)


def build_mask(d, rgba, mask_path, size):
    """White = repaint. Returns (binary mask for the pipeline, soft mask for pasting back) or None if empty."""
    import cv2
    m = np.zeros((size[1], size[0]), bool)
    if mask_path:
        m |= np.asarray(fit(Image.open(mask_path).convert("RGB"), size)).max(2) >= 128
    if d.get("get_mask_from_image_alpha"):
        transparent = (np.asarray(rgba.getchannel("A")) < 128).astype(np.uint8)
        m |= cv2.dilate(transparent, np.ones((20, 20), np.uint8)).astype(bool)
    if not m.any():
        return None
    mask = m.astype(np.uint8) * 255
    soft = cv2.blur(cv2.dilate(mask, np.ones((10, 10), np.uint8)), (10, 10)) if d.get("blur_mask") else mask
    return Image.fromarray(mask), Image.fromarray(soft)


def infill(rgba):
    import cv2
    holes = (np.asarray(rgba.getchannel("A")) < 128).astype(np.uint8)
    rgb = np.asarray(rgba.convert("RGB"))
    return Image.fromarray(cv2.inpaint(rgb, holes, 5, cv2.INPAINT_TELEA)) if holes.any() else Image.fromarray(rgb)


def controlnet_image(d, size):
    """Control image for a non-inpaint ControlNet, optionally made by an ONNX preprocessor. Returns (img, aux path)."""
    path = d.get("controlnet_input_image_path")
    if not path or path == "NULL":
        raise ValueError("ControlNet needs an input image")
    if d.get("do_controlnet_preprocess") in (True, "Yes", "yes"):
        from control_processors.process_body_pose import process_image_body_pose
        from control_processors.process_lineart import process_image_lineart
        from control_processors.process_midas_depth import process_image_midas_depth
        name, fn = {"Depth": ("midas_depth", process_image_midas_depth),
                    "BodyPose": ("body_pose", process_image_body_pose),
                    "LineArt": ("line_art", process_image_lineart)}.get(d.get("controlnet_model"), (None, None))
        if fn is None:
            raise ValueError(f"No preprocessor for ControlNet {d.get('controlnet_model')}; give a ready control image")
        processed = f"{path}.controlnet_processed_{name}.jpg"
        if not os.path.exists(processed):
            fn(path, processed, d.get("controlnet_inp_img_preprocesser_model_path"))
        return fit(Image.open(processed).convert("RGB"), size), processed
    return fit(Image.open(path).convert("RGB"), size), None


def slerp(a, b, t):
    import torch
    af, bf = a.flatten().float(), b.flatten().float()
    omega = torch.acos(torch.clamp(torch.dot(af / af.norm(), bf / bf.norm()), -1, 1))
    so = torch.sin(omega)
    if so.abs() < 1e-6:
        return a
    return ((torch.sin((1 - t) * omega) / so) * a.float() + (torch.sin(t * omega) / so) * b.float()).to(a.dtype)


@contextmanager
def noise_variation(pipe, mod_seed, amount=0.1):
    """small_mod_seed: blend a little noise from a second seed into the pipeline's initial noise."""
    import torch
    module = sys.modules[type(pipe).__module__]
    orig = getattr(module, "randn_tensor", None)
    if mod_seed < 0 or orig is None:  # ponytail: pipelines that don't import randn_tensor get no variation
        yield
        return
    gen = torch.Generator("cpu").manual_seed(mod_seed)

    def varied(shape, *args, **kwargs):
        return slerp(orig(shape, *args, **kwargs), orig(shape, *args, **dict(kwargs, generator=gen)), amount)

    module.randn_tensor = varied
    try:
        yield
    finally:
        module.randn_tensor = orig


def call_params(pipe):
    return inspect.signature(pipe.__call__).parameters


# ---------------------------------------------------------------- engine

class Engine:
    """One cached base pipeline (+ LoRAs, + ControlNet); mode pipelines are made from it with from_pipe."""

    def __init__(self):
        import torch
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.ram = ram_gb()
        self.low_ram = self.ram <= 16
        self.base = self.base_key = self.family = self.cn = self.cn_path = None
        self.loras = ()

    def dtype(self, bf16):
        t = self.torch
        return t.float32 if self.device == "cpu" else (t.bfloat16 if bf16 else t.float16)

    def unload(self):
        self.base = self.base_key = self.family = self.cn = self.cn_path = None
        self.loras = ()
        if "torch._dynamo" in sys.modules:
            self.torch._dynamo.reset()  # compiled graphs of the old transformer
        gc.collect()
        if self.device == "mps":
            self.torch.mps.empty_cache()
        if "mlx.core" in sys.modules:
            sys.modules["mlx.core"].clear_cache()

    def load_base(self, src, family_hint=None):
        if src == self.base_key:
            return
        import diffusers
        self.unload()
        out("sdbk gnms Loading model")
        if os.path.isdir(src):
            index_file = os.path.join(src, "model_index.json")
            if not os.path.exists(index_file):
                raise ValueError(f"Not a diffusers pipeline folder (no model_index.json): {src}")
            index = json.load(open(index_file))
            cls_name = index.get("_class_name", "")
            if not hasattr(diffusers, cls_name):
                raise ValueError(f"{cls_name} needs a newer diffusers than {diffusers.__version__}")
            family = family_hint or family_from_class(cls_name)
            if DIT_PEAK_GB.get(family, 0) > 0.75 * self.ram:  # diffusers would swap: quantized MLX engine
                try:
                    import mflux_pipe
                except ModuleNotFoundError as e:
                    if e.name not in ("mlx", "mflux"):
                        raise
                else:
                    bits = 4 if self.low_ram else 8
                    print(f"{family}: MLX engine, {bits}-bit", file=sys.stderr, flush=True)
                    self.base = mflux_pipe.load(family, src, bits)
                    self.base_key, self.family = src, family
                    return
            kw = dict(dtype=self.dtype("transformer" in index), use_safetensors=True)
            if "safety_checker" in index:
                kw.update(safety_checker=None, requires_safety_checker=False)
            if glob.glob(os.path.join(src, "*", "*.fp16*.safetensors")):
                kw["variant"] = "fp16"
            pipe = diffusers.DiffusionPipeline.from_pretrained(src, **kw)
        else:
            info = inspect_model(src)
            if info["type"] != "sd_model":
                raise ValueError(f"{os.path.basename(src)} is a {info['type']}, not a base model")
            family = family_hint or info["family"]
            names = SINGLE_FILE.get(family)
            if not names:
                raise ValueError(f"Single-file {family or 'unknown'} checkpoints are not supported; use a diffusers folder")
            cls = getattr(diffusers, names[info["is_inpaint"]])
            pipe = cls.from_single_file(src, dtype=self.dtype(family in DIT_FAMILIES))
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        if self.low_ram and hasattr(pipe, "unet"):
            pipe.enable_attention_slicing()
        peak = DIT_PEAK_GB.get(family)
        if peak and self.ram >= 1.5 * peak and self.device == "mps" and os.environ.get("DIFFUSIONBEE_COMPILE") != "0":
            print(f"{family}: torch.compile", file=sys.stderr, flush=True)
            pipe.transformer.compile()  # 9-15% faster; ~10 s compile on the first step at each new size
        self.base, self.base_key, self.family = pipe, src, family

    def load_controlnet(self, path):
        if path != self.cn_path:
            from diffusers import ControlNetModel
            self.cn = self.cn_path = None
            gc.collect()
            dtype = self.base.dtype
            if os.path.isdir(path):
                has_st = bool(glob.glob(os.path.join(path, "*.safetensors")))
                fp16 = glob.glob(os.path.join(path, "*.fp16.safetensors" if has_st else "*.fp16.bin"))
                # no .safetensors (Tile ControlNet): diffusers uses torch.load(weights_only=True)
                cn = ControlNetModel.from_pretrained(path, dtype=dtype, use_safetensors=has_st,
                                                     **({"variant": "fp16"} if fp16 else {}))
            else:
                cn = ControlNetModel.from_single_file(check_path(path), dtype=dtype)
            self.cn, self.cn_path = cn.to(self.device), path
        return self.cn

    def set_loras(self, paths, weights):
        paths = tuple(check_path(p) for p in paths)
        if paths != self.loras:
            if self.loras != ():
                self.base.unload_lora_weights()
            self.loras = None  # unknown state until loading succeeds
            for i, p in enumerate(paths):
                self.base.load_lora_weights(p, adapter_name=f"lora{i}", use_safetensors=True)
            self.loras = paths
        if paths:
            weights = list(weights or []) + [1.0] * len(paths)
            self.base.set_adapters([f"lora{i}" for i in range(len(paths))], [float(w) for w in weights[:len(paths)]])

    def mode_pipe(self, mode, controlnet):
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoPipelineForText2Image
        auto = {"txt2img": AutoPipelineForText2Image, "img2img": AutoPipelineForImage2Image,
                "inpaint": AutoPipelineForInpainting}[mode]
        extra = {} if controlnet is None else {"controlnet": controlnet}
        try:
            pipe = auto.from_pipe(self.base, **extra)
            pipe.set_progress_bar_config(disable=True)
            return pipe
        except ValueError:
            needed = {"txt2img": "prompt", "img2img": "image", "inpaint": "mask_image"}[mode]
            if not extra and needed in call_params(self.base):
                return self.base
            what = mode + (" with ControlNet" if extra else "")
            raise ValueError(f"{what} is not supported for {self.family or 'this model'}")

    def set_scheduler(self, pipe, d):
        import diffusers
        sched = self.base.scheduler
        if "FlowMatch" in type(sched).__name__:
            return  # flow-matching families keep their own scheduler
        name, extra = SCHEDULERS.get(d.get("scheduler"), SCHEDULERS["karras"])
        if d.get("do_v_prediction"):
            extra = dict(extra, prediction_type="v_prediction")
        pipe.scheduler = getattr(diffusers, name).from_config(sched.config, **extra)

    def prepare(self, d):
        """Load models and build the pipeline call for a job. Returns a SimpleNamespace used by render()."""
        src = model_source(d, "model")
        if not src:
            raise ValueError("No model selected")
        inp_path = d.get("input_image") or d.get("input_img") or None
        mask_path = d.get("mask_image") or d.get("mask_image_path") or None
        if inp_path and (mask_path or d.get("get_mask_from_image_alpha")):
            mode = "inpaint"
        elif inp_path and d.get("sd_mode_override") != "txt2img":
            mode = "img2img"
        else:
            mode = "txt2img"
        inpaint_src = model_source(d, "inpaint_model")
        if mode == "inpaint" and inpaint_src:
            src = inpaint_src  # dedicated inpainting checkpoint; same path as model_path is loaded once
        self.load_base(src, d.get("model_family"))
        family = self.family
        if "Inpaint" in type(self.base).__name__:
            mode = "inpaint"  # an inpainting checkpoint can only inpaint; see blank image / full mask below

        size_d, steps_d, cfg_d = FAMILIES.get(family, (1024, 30, 5.0))
        steps = int(d.get("num_steps") or steps_d)
        cfg = float(d["guidance_scale"]) if d.get("guidance_scale") is not None else cfg_d
        rgba = Image.open(inp_path).convert("RGBA") if inp_path else None
        w, h = job_size(d, family, rgba)
        kw = dict(prompt=str(d.get("prompt") or ""), num_inference_steps=steps, guidance_scale=cfg,
                  width=w, height=h, output_type="pil")
        neg = str(d.get("negative_prompt") or "")
        if cfg > 1 and neg:
            kw["negative_prompt"] = neg
        if family == "qwenimage":  # real CFG needs true_cfg_scale and a negative prompt
            kw.update(true_cfg_scale=cfg, negative_prompt=neg or " ", guidance_scale=None)
        if family == "sd15" and d.get("is_clip_skip_2"):
            kw["clip_skip"] = 1  # diffusers counts skipped layers: A1111 "clip skip 2" == 1
            te = self.base.text_encoder
            if not hasattr(te, "text_model"):  # diffusers 0.40 clip_skip expects the transformers 4 CLIPTextModel layout
                object.__setattr__(te, "text_model", SimpleNamespace(final_layer_norm=te.final_layer_norm))

        strength = float(d.get("input_image_strength") if d.get("input_image_strength") is not None else 30)
        strength = strength / 100 if strength > 1 else strength  # UI slider: how much of the input to keep
        strength = min(1.0, max(1.0 - strength, 1.0 / steps + 1e-3))

        job = SimpleNamespace(mode=mode, steps=steps, init=None, soft_mask=None, aux=None)
        if rgba is not None:
            rgba = fit(rgba, (w, h))
            job.init = infill(rgba) if d.get("infill_alpha") else rgba.convert("RGB")
        masks = build_mask(d, rgba, mask_path, (w, h)) if mode == "inpaint" and rgba is not None else None
        if mode == "inpaint" and masks is None:
            if "Inpaint" in type(self.base).__name__:  # inpainting checkpoint used for txt2img/img2img
                masks = (Image.new("L", (w, h), 255),) * 2
                if job.init is None:
                    job.init = Image.new("RGB", (w, h))
            elif d.get("sd_mode_override") == "txt2img":
                raise ValueError("The mask is empty: paint over the area to regenerate")
            else:
                mode = job.mode = "img2img"  # nothing painted on the img2img canvas
        if mode == "img2img":
            kw.update(image=job.init, strength=strength)
        elif mode == "inpaint":
            use_strength = d.get("input_image_strength") is not None and d.get("sd_mode_override") != "txt2img"
            kw.update(image=job.init, mask_image=masks[0], strength=strength if use_strength else 1.0)
            if d.get("do_masking_diffusion") or d.get("inp_only_update_masked", True):
                job.soft_mask = masks[1]

        cn_src = model_source(d, "controlnet")
        controlnet = None
        if cn_src:
            if family not in ("sd15", "sdxl"):
                raise ValueError(f"ControlNet is not supported for {family}")
            controlnet = self.load_controlnet(cn_src)
            if d.get("controlnet_model") == "Inpaint" or d.get("controlnet_inp_img_preprocesser") == "Inpaint":
                if mode != "inpaint":
                    raise ValueError("ControlNet Inpaint needs an image and a mask")
                c = np.asarray(job.init, np.float32) / 255.0
                c[np.asarray(masks[0]) > 127] = -1.0  # masked pixels marked as -1 (control_v11p_sd15_inpaint)
                control = self.torch.from_numpy(c.transpose(2, 0, 1).copy())[None]
            else:
                control, job.aux = controlnet_image(d, (w, h))
            kw.update(controlnet_conditioning_scale=float(d.get("control_weight") or 1.0),
                      guess_mode=bool(d.get("controlnet_guess_mode")))

        if type(self.base).__name__ == "MfluxPipe":
            if mode == "inpaint" or d.get("lora_paths"):  # ponytail: mflux takes lora_paths at load; wire up when asked
                what = "inpaint" if mode == "inpaint" else "LoRA"
                raise ValueError(f"{what} is not supported for {family} on the low-memory MLX engine this Mac uses")
            pipe = self.base
        else:
            self.set_loras(d.get("lora_paths") or [], d.get("lora_weights"))
            pipe = self.mode_pipe(mode, controlnet)
            self.set_scheduler(pipe, d)
        params = call_params(pipe)
        if controlnet is not None:
            kw["control_image" if "control_image" in params else "image"] = control
        job.pipe = pipe
        job.kw = {k: v for k, v in kw.items() if k in params}
        return job

    def render(self, job, seed, mod_seed=-1, **overrides):
        """Run the pipeline once. Returns a PIL image, or None when the user pressed stop."""
        stopped = False

        def on_step(pipe, i, t, cb_kwargs):
            nonlocal stopped
            if self.device == "mps":  # MPS queues steps asynchronously; wait so progress and stop track real work
                self.torch.mps.synchronize()
            total = getattr(pipe, "_num_timesteps", None) or job.steps
            out(f"sdbk dnpr {min(100, int(100 * (i + 1) / total))}")
            if stop_requested():
                pipe._interrupt = True
                stopped = True
            return {}

        if stop_requested():
            return None
        kw = dict(job.kw, generator=self.torch.Generator("cpu").manual_seed(seed), **overrides)
        if "callback_on_step_end" in call_params(job.pipe):
            kw["callback_on_step_end"] = on_step
        with noise_variation(job.pipe, mod_seed):
            try:
                img = job.pipe(**kw).images[0]
            except Exception as e:
                tr = getattr(job.pipe, "transformer", None)
                if getattr(tr, "_compiled_call_impl", None) is None:
                    raise
                print(f"torch.compile failed, running eager: {one_line(e)}", file=sys.stderr, flush=True)
                tr._compiled_call_impl = None  # undo transformer.compile() for good
                kw["generator"] = self.torch.Generator("cpu").manual_seed(seed)
                img = job.pipe(**kw).images[0]
        if stopped:
            return None
        if job.soft_mask is not None:  # keep the unmasked pixels exactly
            img = Image.composite(img.resize(job.init.size), job.init, job.soft_mask)
        return img

    def run_job(self, d):
        job = self.prepare(d)
        seed = int(d.get("seed") or 0)
        if seed < 1:
            seed = random.randint(1, 2 ** 31 - 1)
        mod_seed = d.get("small_mod_seed")
        mod_seed = int(mod_seed) if mod_seed not in (None, "") else -1
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(int(d.get("num_imgs") or 1)):
            s, m = (seed, mod_seed + 1234 * i) if mod_seed >= 0 else (seed + 1234 * i, -1)
            img = self.render(job, s, m)
            if img is None:
                return
            name = "".join(filter(str.isalnum, str(d.get("prompt") or "")[:30]))
            path = IMAGES_DIR / f"{name}_{random.randint(0, 100000000)}.png"
            img.save(path)
            res = {"generated_img_path": str(path), "seed": s}
            if job.aux:
                res["aux_output_image_path"] = job.aux
            out("sdbk nwim " + json.dumps(res))


# ---------------------------------------------------------------- main

def serve():
    sys.stdout.reconfigure(line_buffering=True)
    from applets.applets import register_applet, run_applet
    from applets.frame_interpolator import FrameInterpolator

    out("sdbk mltl Loading Model")
    engine = Engine()
    register_applet(engine, FrameInterpolator)
    threading.Thread(target=_read_stdin, daemon=True).start()
    out("sdbk mdld")
    while True:
        out("sdbk inrd")
        line = _stdin.get()
        if line is None:
            return
        if "__stop__" in line:
            continue
        try:
            if line.startswith("b2py t2im"):
                d = json.loads(line[len("b2py t2im"):])
                out("sdbk inwk")
                engine.run_job(d)
            elif line.startswith("b2py rapp"):
                name, _, params = line[len("b2py rapp"):].strip().partition(" ")
                d = json.loads(params)
                out("sdbk inwk")
                run_applet(name, d)
        except Exception as e:
            traceback.print_exc()
            out("sdbk errr " + one_line(e))


def main(argv):
    cmd = argv[1] if len(argv) > 1 else None
    if cmd not in ("download_model", "inspect_model", "upscale"):
        return serve()
    try:
        if cmd == "download_model":
            args = argv[2:]
            variant = args[args.index("--variant") + 1] if "--variant" in args else "fp16"
            last = [-1]

            def progress(p):
                if p != last[0]:
                    last[0] = p
                    out(f"progress {p}")
            out("done " + fetch_repo(args[0], variant, progress))
        elif cmd == "upscale":
            from upscale import upscale
            upscale(argv[2], argv[3])
            out("done " + argv[3])
        else:
            out(json.dumps(inspect_model(argv[2])))
    except Exception as e:
        traceback.print_exc()
        print(one_line(e), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # PyInstaller
    main(sys.argv)
