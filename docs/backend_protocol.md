# Backend protocol (v3, diffusers)

The Electron app talks to the Python backend over stdin/stdout, one line per message.
The line protocol is unchanged from v2; only the job fields changed.
Implementation: `backends/stable_diffusion/diffusionbee_backend.py` (PyTorch MPS + Hugging Face diffusers 0.40).

## Long-running backend (`diffusionbee_backend` with no args)

Backend → app (stdout). Anything else on stdout/stderr is log output.

| Line | Meaning |
|---|---|
| `sdbk mltl <text>` | loading title |
| `sdbk mdld` | backend ready |
| `sdbk inrd` | waiting for input |
| `sdbk inwk` | job accepted |
| `sdbk dnpr <0-100>` | progress of the current image, once per denoising step |
| `sdbk gnms <text>` | status text (`Loading model`, `Downloading model 42%`) |
| `sdbk nwim {"generated_img_path": "...", "seed": 123, "aux_output_image_path": "..."}` | image done. `aux_output_image_path` only when a ControlNet preprocessor ran |
| `sdbk errr <text>` | job failed (one line) |
| `utds <key>___U_P_D_A_T_E___<json>` | applet UI state (Interpolator applet) |

App → backend (stdin): `b2py t2im {json}`, `b2py rapp <applet> {json}`, `b2py t2im __stop__`.
Stop is checked every step; the current image is dropped and the rest of the job is skipped.
Lines other than `__stop__` that arrive while a job runs are dropped.

Images are saved as PNG in `~/.diffusionbee/images/`.

### Job JSON (`b2py t2im`)

Model fields (replace all `*_tdict_path` fields):

| Field | Meaning |
|---|---|
| `model_path` | Local diffusers folder (has `model_index.json`) or single-file `.safetensors` checkpoint |
| `model_repo` | HF repo id, used when `model_path` is absent (resolved from the HF cache, downloaded if missing). The app downloads first and sends `model_path` |
| `model_family` | optional hint: `sd15`, `sdxl`, `sd3`, `flux`, `flux2`, `zimage`, `qwenimage`. Autodetected otherwise |
| `inpaint_model_path` / `inpaint_model_repo` | optional dedicated inpainting checkpoint, used for inpaint jobs. May equal `model_path` (loaded once). An inpainting checkpoint always runs its inpaint pipeline (full mask / blank image when none is given) |
| `controlnet_path` / `controlnet_repo` | ControlNet: diffusers folder or single `.safetensors`. SD 1.5 / SDXL only |
| `lora_paths` | list of LoRA `.safetensors` files (the app sends one); `lora_weights` optional parallel list, default 1.0 |

Rejected: `.tdict` paths (DiffusionBee 1.x/2.x, "re-import the original .safetensors") and pickle files
(`.ckpt .pt .pth .bin .pkl .pickle`). The one exception is a ControlNet folder that ships no `.safetensors`
(`lllyasviel/control_v11f1e_sd15_tile`): diffusers reads its `.bin` with `torch.load(weights_only=True)`.
Single-file checkpoints are supported for sd15 and sdxl. diffusers fetches their small config files from the Hub the first time.
sd3, flux and zimage single files usually hold only the transformer, and diffusers then downloads the text encoders and VAE from the base repo, which is large and sometimes gated.
Use diffusers folders for those and for flux2 and qwenimage.

Generation fields:

| Field | Meaning |
|---|---|
| `prompt`, `negative_prompt` | the negative prompt is used only when `guidance_scale` > 1 |
| `num_steps`, `guidance_scale` | family defaults (below) when missing. `guidance_scale` ≤ 1 means no CFG. qwenimage gets it as `true_cfg_scale` (with a `" "` negative prompt if empty) |
| `num_imgs` | images per job |
| `seed` | < 1 or missing: random. Image *i* uses `seed + 1234·i` |
| `small_mod_seed` | ≥ 0: the seed stays fixed and image *i* slerps 10% of the noise from seed `small_mod_seed + 1234·i` into it |
| `scheduler` | `karras` (DPM++ 2M Karras, default), `ddim`, `lmsd` (runs Euler: LMS needs scipy), `pndm`, `k_euler_ancestral`, `k_euler`, `lcm` (for LCM-LoRA, 4–8 steps, cfg 1–2). SD 1.5 / SDXL only; flow-matching families keep their scheduler |
| `do_v_prediction` | sets `prediction_type="v_prediction"` on the scheduler (SD 2.x-style v models) |
| `is_clip_skip_2` | sd15 only (A1111 "clip skip 2") |
| `img_width`, `img_height` | txt2img size, family default when missing. Rounded down to multiples of 16, area capped at 2048² |
| `input_img` / `input_image` | input image. Without a mask this means img2img, unless `sd_mode_override` is `"txt2img"` |
| `force_use_given_size` | with an input image: use `img_width`/`img_height`. Otherwise the output keeps the input aspect at the family's native area (512² for sd15, 1024² for most others) |
| `input_image_strength` | 2–100 (or 0–1), how much of the input to keep. diffusers `strength = 1 − s`. Inpaint uses 1.0 when `sd_mode_override` is `"txt2img"` (Inpainting page) |
| `mask_image` / `mask_image_path` | white = repaint. Makes the job inpaint |
| `get_mask_from_image_alpha` | transparent input pixels are repainted too (dilated 20 px). Makes the job inpaint |
| `blur_mask` | soft edge when pasting the original pixels back |
| `do_masking_diffusion` / `inp_only_update_masked` | paste the original pixels back outside the mask (default on) |
| `infill_alpha` | fill transparent input pixels (OpenCV Telea) before diffusion |
| `controlnet_model` | `Depth`, `BodyPose`, `LineArt`, `Scribble`, `Tile`, or `Inpaint`. Only used together with `controlnet_path` |
| `controlnet_input_image_path` | control image (ignored for `Inpaint`) |
| `do_controlnet_preprocess` + `controlnet_inp_img_preprocesser_model_path` | make the control image with the ONNX preprocessor (Depth, BodyPose, LineArt only). The result is saved as `<input>.controlnet_processed_<name>.jpg`, reused, and returned as `aux_output_image_path` |
| `control_weight` | ControlNet conditioning scale (default 1.0) |
| `controlnet_guess_mode` | diffusers `guess_mode` |

The mode is **inpaint** when there is an input image and a mask (`mask_image*` or `get_mask_from_image_alpha`). If the mask comes out empty, the job runs as img2img, or fails with "The mask is empty" when `sd_mode_override` is `"txt2img"`.
Otherwise the mode is **img2img** when there is an input image, else **txt2img**.

A mode the model can't do returns an error such as `inpaint is not supported for flux2`.
With `controlnet_model: "Inpaint"` (Generative Fill on SD 1.5) the job runs the ControlNet-inpaint pipeline, and its control image is the input with masked pixels set to −1.
Ignored legacy fields: `prompt_tokens`, `negative_prompt_tokens`, `is_control_net`, `controlnet_inp_img_preprocesser` (except `"Inpaint"`), `batch_size`.

Models stay loaded between jobs. A different model, inpaint checkpoint or ControlNet reloads, and LoRAs reload only when `lora_paths` changes.

### Applets (`b2py rapp <applet> {json}`)

`frame_interpolate` (Interpolator): `num_frames`, `seed1`, `prompt1`, `seed2`, `prompt2` plus the generation fields above.
It renders a GIF by mixing the two seeds' noise and lerping the prompt embeddings, and reports progress through `utds` lines. SD 1.5 / SDXL only.

## One-shot subcommands

Each prints plain lines on stdout. On failure it exits non-zero and the last stderr line is the error.

- `diffusionbee_backend download_model <repo_id> [--variant fp16]`
  prints `progress <0-100>` lines, then `done <HF_HOME>/hub/models--<org>--<name>/snapshots/<revision>`.
  - Downloads only what the pipeline loads: configs and tokenizers, plus one set of `.safetensors` per component listed in `model_index.json`.
  - Prefers the variant, which defaults to `fp16` when the repo has it.
  - Skips root single-file checkpoints, `.bin`/`.ckpt`, `non_ema`, onnx, `safety_checker` and assets.
  - Example sizes: SD 1.5 2.0 GB (repo 47 GB), FLUX.1-schnell 33.7 GB (repo 58 GB), FLUX.2-klein-4B 16 GB (repo 24 GB).
  - Z-Image-Turbo is 32.8 GB because its transformer is stored in fp32 (it loads as bf16).
  - `HF_HOME` defaults to `~/.diffusionbee/hf`. The `HF_TOKEN` env var enables gated repos, and a gated repo without access gives a "gated: accept its license" error.
  - When the Hub can't be reached, it falls back to the cached copy.
- `diffusionbee_backend inspect_model <path>` (a `.safetensors` file or a diffusers folder)
  prints one JSON line `{"family": "sdxl", "is_inpaint": false, "type": "sd_model"|"lora"|"controlnet"}`.
  - `family` is `null` when unknown.
  - A LoRA's family comes from its cross-attention width: 768/1024 gives sd15, 2048 gives sdxl. It is `null` for DiT LoRAs, which the app shows for every model.
- `diffusionbee_backend upscale <in.png> <out.png>`
  4x Real-ESRGAN x4plus on MPS (fp16, 256 px tiles), prints `done <out.png>`. Alpha is kept.
  Weights: `Comfy-Org/Real-ESRGAN_repackaged/RealESRGAN_x4plus.safetensors` (67 MB), fetched into the HF cache on first use.

## Model families (defaults the UI should use)

Steps and cfg follow the Hugging Face model cards.

| family | size | steps | cfg | negative prompt | img2img | inpaint | controlnet |
|---|---|---|---|---|---|---|---|
| sd15 | 512 | 25 | 7.5 | yes | yes | yes | yes |
| sdxl | 1024 | 30 | 6 | yes | yes | yes | yes |
| sd3 (3.5 Medium) | 1024 | 40 | 4.5 | yes | yes | yes | no |
| flux (FLUX.1) | 1024 | 4 schnell / 28 dev | 0 schnell / 3.5 dev (distilled guidance) | no | yes | yes | no |
| flux2 (FLUX.2 klein) | 1024 | 4 | 1 | no | yes (reference image) | no | no |
| zimage (Z-Image-Turbo) | 1024 | 8 | 0 | no | yes | yes | no |
| qwenimage (Qwen-Image, Qwen-Image-2512) | 1328 | 50 | 4 (true CFG) | yes | yes | yes | no |

Qwen-Image-2.1 (`QwenImage21Pipeline`, 2048 px, 40 steps) needs diffusers from git main, not 0.40. The backend refuses it with "QwenImage21Pipeline needs a newer diffusers".

Apple Silicon settings:
- Device is `mps`. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set.
- DiT families load in bf16. SD 1.5/SDXL load in fp16; the SDXL VAE upcasts itself.
- Attention is SDPA. Attention slicing is on only with 16 GB of RAM or less.
- Engine for zimage and flux2 diffusers folders: the MLX engine (mflux, quantized) when the family's diffusers bf16 peak
  (Z-Image 30.5 GB, FLUX.2 23 GB at 1024²) is over 75% of RAM; 4-bit with 16 GB of RAM or less, else 8-bit. Otherwise diffusers.
  Other families and single-file checkpoints always use diffusers.

  | RAM | zimage | flux2 |
  |---|---|---|
  | ≤ 16 GB | MLX 4-bit (~10 GB peak) | MLX 4-bit (~10 GB) |
  | 24 GB | MLX 8-bit (~15 GB) | MLX 8-bit (~13 GB) |
  | 32 GB | MLX 8-bit | diffusers bf16 |
  | 48 GB | diffusers bf16 + compile | diffusers bf16 + compile |

- On diffusers, zimage and flux2 transformers get `torch.compile` when RAM is at least 1.5× that peak (Z-Image 46 GB,
  FLUX.2 35 GB): about 9–15% faster per image. The first step at each new size compiles for ~10 s. If a compiled call fails,
  the backend logs it to stderr and reruns that image eagerly, and the model stays eager. `DIFFUSIONBEE_COMPILE=0` turns it off.
- `DIFFUSIONBEE_RAM_GB=<GB>` in the backend's environment replaces the detected RAM for these choices (and attention slicing),
  to test the tiers on a bigger Mac or to force the MLX engine. The app passes its environment through to the backend.
- The MLX engine runs txt2img and img2img. Inpaint and LoRA jobs fail with
  `inpaint is not supported for zimage on the low-memory MLX engine this Mac uses` (or `LoRA …`); ControlNet is SD 1.5/SDXL only anyway.
  `small_mod_seed` is ignored, a seed gives a different image than on diffusers, and flux2 img2img starts from the noised
  input image instead of using it as a reference image. Without mlx installed (dev venv) these families use diffusers.
