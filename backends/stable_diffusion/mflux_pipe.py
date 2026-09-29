"""Low-RAM engine for zimage/flux2/flux, and the only engine for qwenimage21 (diffusers 0.40 has no pipeline for it):
an mflux (MLX) model behind the part of the diffusers pipeline call that Engine.render() uses
(prompt/steps/guidance/size/generator/callback_on_step_end, image+strength for img2img, negative_prompt,
.images[0], pipe._interrupt to stop). See docs/mlx_benchmark_notes.md. Raises ImportError without mlx/mflux."""
import inspect
import json
import os
import re
import tempfile
from types import SimpleNamespace

import mlx.core as mx
from mflux.utils.exceptions import StopImageGenerationException

mx.set_cache_limit(2 << 30)  # MLX keeps freed buffers by default: +20 GB after one 1024² image, then swap


def load(family, src, bits, transformer=None):
    """src: the diffusers snapshot dir load_base() gets; mflux converts and quantizes it at load (3-9 s).
    transformer: a fine-tuned transformer .safetensors, already quantized to `bits` in mflux's own layout,
    that replaces the base one. The base weights are lazy and replaced before evaluation, so they're never read.
    The mflux ModelConfig comes from the model's configs, so fine-tunes and the other variants of a family load too."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.vae.tiling_config import TilingConfig
    cfg = lambda *p: json.load(open(os.path.join(src, *p)))
    if family == "zimage":  # Turbo is CFG-distilled; base Z-Image has twice its scheduler shift (6 vs 3)
        from mflux.models.z_image import ZImage
        base = cfg("scheduler", "scheduler_config.json").get("shift", 3) > 3
        model = ZImage(quantize=bits, model_path=src, model_config=ModelConfig.z_image() if base else ModelConfig.z_image_turbo())
    elif family == "flux2":  # klein 4B or 9B, by transformer size (base and distilled klein share a config)
        from mflux.models.flux2 import Flux2Klein
        t = cfg("transformer", "config.json")
        mc = next((c for c in (ModelConfig.flux2_klein_4b(), ModelConfig.flux2_klein_9b())
                   if all(t.get(k) == v for k, v in c.transformer_overrides.items())), None)
        if not mc:
            raise ValueError("The MLX engine runs FLUX.2-klein 4B and 9B, not this FLUX.2 model")
        model = Flux2Klein(quantize=bits, model_path=src, model_config=mc)
    elif family == "flux":  # FLUX.1: dev and Krea-dev have a guidance embedding, schnell doesn't
        from mflux.models.flux.variants.txt2img.flux import Flux1
        dev = cfg("transformer", "config.json").get("guidance_embeds")
        model = Flux1(quantize=bits, model_path=src, model_config=ModelConfig.dev() if dev else ModelConfig.schnell())
    elif family == "qwenimage21":
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
        model = QwenImage21(quantize=bits, model_path=src)
    else:
        raise ValueError(f"No MLX engine for {family}")
    if transformer:
        load_transformer(model.transformer, transformer, family)
    mx.eval(model.parameters())  # quantize now; left lazy, the first image holds bf16 + q4 (Z-Image: 30 GB peak)
    mx.clear_cache()
    model.tiling_config = TilingConfig()  # VAE tiling: -4.5 GB peak at 1024², same speed, no seams
    return MfluxPipe(model, family)


# other MLX ports' names for mflux transformer weights (e.g. abenzerps/Qwen-Image-2.1-Uncensored-GGUF's MLX files)
# ponytail: add renames as other ports turn up
RENAMES = [(re.compile(r"^modulation\.0\."), "modulation.layers.1."),
           (re.compile(r"^time_text_embed\.linear_"), "time_text_embed.timestep_embedder.linear_")]


def load_transformer(module, path, family):
    """Weights from `path` into the (quantized) mflux transformer. Every file tensor must fit and every weight be
    covered, so a file for another family or bit width fails here, not mid-image. Computed buffers (RoPE tables,
    timestep freqs) are never in a file, which is why this isn't load_weights(strict=True)."""
    from mlx.utils import tree_flatten
    w = {}
    for k, v in mx.load(path).items():
        for pattern, new in RENAMES:
            k = pattern.sub(new, k)
        w[k] = v
    have = dict(tree_flatten(module.parameters()))
    bad = [k for k in w if k not in have or have[k].shape != w[k].shape]
    bad += [k for k in have if k not in w and k.rsplit(".", 1)[-1] in ("weight", "bias", "scales", "biases")]
    if bad:
        raise ValueError(f"{os.path.basename(path)} doesn't fit the {family} MLX transformer ({len(bad)} tensors, e.g. {bad[0]})")
    module.load_weights(list(w.items()), strict=False)


def lora_mapping(family):
    """mflux's LoRA key -> module mapping. mflux has none for Qwen-Image-2.1: its module names, as LoRA files
    (diffusers/PEFT, ComfyUI, kohya) and the MLX ports name them, run through RENAMES."""
    from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
    if family == "zimage":
        from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping
        return ZImageLoRAMapping.get_mapping()
    if family == "flux":
        from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping
        return FluxLoRAMapping.get_mapping()
    if family == "flux2":
        from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping
        return Flux2LoRAMapping.get_mapping()
    names =[f"transformer_blocks.{{block}}.{m}" for m in
             ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out")]
    names += ["img_in", "txt_in.in_layer", "txt_in.out_layer", "proj_out", "norm_out.linear", "modulation.0",
              "time_text_embed.linear_1", "time_text_embed.linear_2"]
    targets = []
    for n in names:
        path = n + "."
        for pattern, new in RENAMES:
            path = pattern.sub(new, path)
        keys = [p + n for p in ("", "transformer.", "diffusion_model.")]
        kohya = "lora_unet_" + n.replace(".", "_")
        ends = lambda b, u: [f"{k}.{e}" for k in keys for e in (f"lora_{b}.weight", f"lora_{b}.default.weight",
                                                                   f"lora_{u}.weight", f"lora.{u}.weight")] + [f"{kohya}.lora_{u}.weight"]
        targets.append(LoRATarget(model_path=path[:-1], possible_up_patterns=ends("B", "up"),
                                  possible_down_patterns=ends("A", "down"),
                                  possible_alpha_patterns=[k + ".alpha" for k in keys] + [kohya + ".alpha"]))
    return targets


def strip_loras(module):
    """Put back the layers LoRA adapters wrap."""
    from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
    from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
    from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
    from mflux.models.common.lora.mapping.lora_loader import LoRALoader
    done = []
    for name, m in sorted(module.named_modules(), key=lambda x: len(x[0])):  # a wrapper before the ones inside it
        if isinstance(m, (LoRALinear, LoKrLinear, FusedLoRALinear)) and not any(name.startswith(d + ".") for d in done):
            LoRALoader._replace_target_module(module, name, m.base_linear if isinstance(m, FusedLoRALinear) else m.linear)
            done.append(name)


class MfluxPipe:
    def __init__(self, model, family=None):
        self.model, self.family, self.loras = model, family, ((), [])
        self._interrupt, self._num_timesteps, self._cb = False, None, None
        model.callbacks.register(self)

    def set_loras(self, paths, weights):
        """LoRAs as adapter layers on the (quantized) transformer, not baked into it: changing them strips the
        layers instead of reloading and requantizing the model."""
        paths = tuple(paths)
        weights = [float(w) for w in list(weights or []) + [1.0] * len(paths)][:len(paths)]
        if (paths, weights) == self.loras:
            return
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
        from mflux.models.common.lora.mapping.lora_loader import LoRALoader
        t = self.model.transformer
        self.loras = None  # unknown state until loading succeeds
        try:
            strip_loras(t)
            if paths:
                LoRALoader.load_and_apply_lora(lora_mapping(self.family), t, list(paths), weights, bake_lora=False)
            for _, m in t.named_modules():  # an fp16 or fp32 LoRA would turn the bf16 activations fp32
                if isinstance(m, LoRALinear):
                    m.lora_A, m.lora_B = m.lora_A.astype(ModelConfig.precision), m.lora_B.astype(ModelConfig.precision)
        except Exception:
            strip_loras(t)  # a half-applied LoRA is neither model
            self.loras = ((), [])
            raise
        finally:
            if hasattr(t, "_step_fn"):
                t._step_fn = None  # Qwen-Image-2.1 keeps its compiled step, which holds the old layers
        self.loras = (paths, weights)

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):  # mflux in-loop callback, once per step
        mx.eval(latents)  # MLX is lazy: without this, progress and stop run ahead of the GPU
        self._num_timesteps = len(time_steps)  # img2img runs only the last steps, like diffusers
        if self._cb:
            self._cb(self, t - config.init_time_step, None, {})
        if self._interrupt:
            raise KeyboardInterrupt  # mflux turns this into StopImageGenerationException

    def set_progress_bar_config(self, **kw):
        pass

    def __call__(self, prompt, num_inference_steps, guidance_scale, width, height, generator,
                 callback_on_step_end=None, image=None, strength=None, negative_prompt=None, output_type="pil"):
        self._interrupt, self._num_timesteps, self._cb = False, num_inference_steps, callback_on_step_end
        kw = dict(seed=generator.initial_seed(), prompt=prompt, num_inference_steps=num_inference_steps,
                  width=width, height=height, guidance=guidance_scale)
        if "negative_prompt" in inspect.signature(self.model.generate_image).parameters:
            # CFG runs when guidance > 1: flux2 (klein base) only with a negative prompt, "" as in diffusers. zimage
            # takes "" as none; qwenimage21 skips CFG on "", so Engine.prepare() gives it " ". FLUX.1 ignores it
            kw["negative_prompt"] = negative_prompt or ""
        with tempfile.TemporaryDirectory() as d:
            if image is not None:  # mflux img2img takes a file and the fraction of steps to skip (1 - diffusers strength)
                kw.update(image_path=os.path.join(d, "init.png"), image_strength=1 - strength)
                image.save(kw["image_path"])
            try:
                r = self.model.generate_image(**kw)
            except StopImageGenerationException:
                return SimpleNamespace(images=[None])  # render() has set stopped and returns None
            finally:
                mx.clear_cache()
        return SimpleNamespace(images=[getattr(r, "image", r)])
