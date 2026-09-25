import json
import math
import os
import random
import tempfile

from .applets import AppletBase
from .form_utils import get_textbox, get_output_text, get_textarea, get_output_img
from .options import options

EMBED_NAMES = ["prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"]


def tmp_path(ext):
    return os.path.join(tempfile.gettempdir(), f"{random.randint(0, 100000000)}{ext}")


class FrameInterpolator(AppletBase):
    """Video from interpolating two (seed, prompt) pairs: noise is mixed, prompt embeddings are lerped.
    SD 1.5 / SDXL only (their prompt embeddings can be passed to the pipeline directly)."""

    applet_name = "frame_interpolate"
    applet_title = "Interpolator"
    applet_description = "Generate videos by interpolating prompts and seeds"
    is_stop_avail = True
    applet_icon_fname = "interpolate.gif"

    def get_input_form(self):
        my_el = [
            get_textbox("num_frames", type="int", default=40, title="Number of frames", description="The number of frames you want in the video."),
            get_textbox("seed1", type="int", title="Start Seed", description="The seed from which it will start the video."),
            get_textarea("prompt1", title="Start Prompt", description="The prompt from which it will start the video."),
            get_textbox("seed2", type="int", title="End Seed", description="The seed to which it will end the video. Keep it same as start seed if you dont want to change the seed in the video."),
            get_textarea("prompt2", title="End Prompt", description="The prompt to which it will end the video. Keep it same as start prompt if you dont want to change the prompt in the video."),
            get_output_text("The following options will remain constant in the whole video:"),
        ]
        form = json.loads(options)
        form = [f for f in form if f['id'] not in ['controlnet_acc', 'seed_acc', 'seed_desc', 'prompt', 'num_imgs_desc']]
        return my_el + form

    def update_progress(self, n_img_done, n_total, cur_img=None):
        oo = [get_output_text(f"Rendering... \n\n Rendered {n_img_done} out of {n_total} images.")]
        if cur_img is not None:
            fn = tmp_path(".jpg")
            cur_img.resize([s // 2 for s in cur_img.size]).save(fn)
            oo.append(get_output_text("Current frame:"))
            oo.append(get_output_img(fn))
        self.update_state("outputs", oo)

    def run(self, params):
        n_total = int(params.get('num_frames') or 0)
        seed1, seed2 = int(params.get('seed1') or 0), int(params.get('seed2') or 0)
        prompt1, prompt2 = str(params.get('prompt1') or ""), str(params.get('prompt2') or "")
        if n_total < 3:
            raise ValueError("Enter a valid number of frames")
        if seed1 < 1 or seed2 < 1:
            raise ValueError("Enter valid start and end seeds")
        if len(prompt1) < 5 or len(prompt2) < 5:
            raise ValueError("Enter valid start and end prompts")

        engine = self.model_container
        job = engine.prepare(dict(params, prompt=prompt1, input_img=None, input_image=None, mask_image=None,
                                  sd_mode_override="txt2img", controlnet_path=None))
        if engine.family not in ("sd15", "sdxl"):
            raise ValueError("The Interpolator works with SD 1.5 and SDXL models")
        pipe, kw, torch = job.pipe, job.kw, engine.torch
        from diffusers.utils.torch_utils import randn_tensor

        def embeds(prompt):
            res = pipe.encode_prompt(prompt=prompt, device=engine.device, num_images_per_prompt=1,
                                     do_classifier_free_guidance=kw["guidance_scale"] > 1,
                                     negative_prompt=kw.get("negative_prompt"), clip_skip=kw.get("clip_skip"))
            return {k: v for k, v in zip(EMBED_NAMES, res) if v is not None}

        def noise(seed):
            shape = (1, pipe.unet.config.in_channels, kw["height"] // pipe.vae_scale_factor, kw["width"] // pipe.vae_scale_factor)
            return randn_tensor(shape, generator=torch.Generator("cpu").manual_seed(seed), device=torch.device(engine.device), dtype=pipe.unet.dtype)

        with torch.no_grad():
            emb_a, emb_b = embeds(prompt1), embeds(prompt2)
        lat_a, lat_b = noise(seed1), noise(seed2)

        frames = []
        self.update_progress(0, n_total)
        for i in range(n_total):
            frac = i / (n_total - 1)
            fa, fb = (math.sqrt(1 - frac), math.sqrt(frac)) if seed1 != seed2 else (1.0, 0.0)
            emb = {k: emb_a[k] * (1 - frac) + emb_b[k] * frac for k in emb_a}
            img = engine.render(job, seed1, prompt=None, negative_prompt=None, latents=lat_a * fa + lat_b * fb, **emb)
            if img is None:
                return self.update_state("outputs", [get_output_text("Stopped")])
            frames.append(img)
            self.update_progress(i + 1, n_total, img)

        fn = tmp_path(".gif")
        frames[0].save(fn, format='GIF', append_images=frames[1:], save_all=True, loop=0)
        self.update_state("outputs", [get_output_img(fn, save_ext='.gif', is_save=True)])
