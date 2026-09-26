"""Low-RAM engine for zimage/flux2/flux, and the only engine for qwenimage21 (diffusers 0.40 has no pipeline for it):
an mflux (MLX) model behind the part of the diffusers pipeline call that Engine.render() uses
(prompt/steps/guidance/size/generator/callback_on_step_end, image+strength for img2img, negative_prompt,
.images[0], pipe._interrupt to stop). See docs/mlx_benchmark_notes.md. Raises ImportError without mlx/mflux."""
import inspect
import json
import os
import tempfile
from types import SimpleNamespace

import mlx.core as mx
from mflux.utils.exceptions import StopImageGenerationException

mx.set_cache_limit(2 << 30)  # MLX keeps freed buffers by default: +20 GB after one 1024² image, then swap


def load(family, src, bits):
    """src: the diffusers snapshot dir load_base() gets; mflux converts and quantizes it at load (3-9 s).
    ponytail: assumes the catalog's Z-Image-Turbo / FLUX.2-klein-4B / FLUX.1 dev or schnell / Qwen-Image-2.1;
    other variants need their mflux ModelConfig."""
    from mflux.models.common.vae.tiling_config import TilingConfig
    if family == "zimage":
        from mflux.models.z_image import ZImageTurbo
        model = ZImageTurbo(quantize=bits, model_path=src)
    elif family == "flux2":
        from mflux.models.flux2 import Flux2Klein
        model = Flux2Klein(quantize=bits, model_path=src)
    elif family == "flux":  # FLUX.1: dev and Krea-dev have a guidance embedding, schnell doesn't
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux.variants.txt2img.flux import Flux1
        dev = json.load(open(os.path.join(src, "transformer", "config.json"))).get("guidance_embeds")
        model = Flux1(quantize=bits, model_path=src, model_config=ModelConfig.dev() if dev else ModelConfig.schnell())
    elif family == "qwenimage21":
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
        model = QwenImage21(quantize=bits, model_path=src)
    else:
        raise ValueError(f"No MLX engine for {family}")
    mx.eval(model.parameters())  # quantize now; left lazy, the first image holds bf16 + q4 (Z-Image: 30 GB peak)
    mx.clear_cache()
    model.tiling_config = TilingConfig()  # VAE tiling: -4.5 GB peak at 1024², same speed, no seams
    return MfluxPipe(model)


class MfluxPipe:
    def __init__(self, model):
        self.model, self._interrupt, self._num_timesteps, self._cb = model, False, None, None
        model.callbacks.register(self)

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
        if negative_prompt and "negative_prompt" in inspect.signature(self.model.generate_image).parameters:
            kw["negative_prompt"] = negative_prompt  # qwenimage21 (true CFG), zimage; FLUX.1 ignores it
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
