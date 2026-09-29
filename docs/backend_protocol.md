# Backend protocol (v3, diffusers)

The Electron app talks to the Python backend over stdin/stdout, one line per message.
The line protocol is unchanged from v2; only the job fields changed.
Implementation: `backends/stable_diffusion/diffusionbee_backend.py` (PyTorch MPS + Hugging Face diffusers 0.40).

## Long-running backend (`diffusionbee_backend` with no args)

Backend → app (stdout). Anything else on stdout/stderr is log output.

| Line | Meaning |
|---|---|
| `sdbk mltl <text>` | loading title |
| `sdbk mlxf ["flux", "qwenimage21"]` | sent once at startup: the families this Mac runs on the MLX engine (see the tier table). That engine has no inpaint, so the UI hides it for them |
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
| `model_path` | Local diffusers folder (has `model_index.json`), single-file `.safetensors` checkpoint, or MLX transformer file (below) |
| `model_repo` | HF repo id, used when `model_path` is absent (resolved from the HF cache, downloaded if missing). The app downloads first and sends `model_path` |
| `model_family` | optional hint: `sd15`, `sdxl`, `sd3`, `flux`, `flux2`, `zimage`, `qwenimage`, `qwenimage21`. Autodetected otherwise |
| `inpaint_model_path` / `inpaint_model_repo` | optional dedicated inpainting checkpoint, used for inpaint jobs. May equal `model_path` (loaded once). An inpainting checkpoint always runs its inpaint pipeline (full mask / blank image when none is given) |
| `controlnet_path` / `controlnet_repo` | ControlNet: diffusers folder or single `.safetensors`. SD 1.5 / SDXL only |
| `base_model_path` / `base_model_repo` | the diffusers folder an MLX transformer file in `model_path` runs on: its text encoder, VAE and configs |
| `lora_paths` | list of LoRA `.safetensors` files (the app sends one); `lora_weights` optional parallel list, default 1.0 |

Rejected: `.tdict` paths (DiffusionBee 1.x/2.x, "re-import the original .safetensors") and pickle files
(`.ckpt .pt .pth .bin .pkl .pickle`). The one exception is a ControlNet folder that ships no `.safetensors`
(`lllyasviel/control_v11f1e_sd15_tile`): diffusers reads its `.bin` with `torch.load(weights_only=True)`.
Single-file checkpoints are supported for sd15 and sdxl. diffusers fetches their small config files from the Hub the first time.
sd3, flux and zimage single files usually hold only the transformer, and diffusers then downloads the text encoders and VAE from the base repo, which is large and sometimes gated.
Use diffusers folders for those and for flux2 and qwenimage.

An MLX transformer file (`inspect_model` reports `mlx_bits`) is a fine-tuned transformer already quantized in the
MLX layout, e.g. `abenzerps/Qwen-Image-2.1-Uncensored-GGUF`'s `qwen-image-2.1-UC-MLX-{4,6,8}bit.safetensors`.
It runs on the MLX engine at its own bit width, whatever the RAM tier, with everything else from `base_model_path`.
The base weights are never read: its transformer is replaced before evaluation, so loading takes about as long as the base (Qwen-Image-2.1 8-bit: 6 s, same 29 GB peak).
Every tensor must fit the family's mflux transformer, so a file for another family or bit width fails at load.
It takes LoRAs but not inpaint, as on the MLX engine generally.

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

- `diffusionbee_backend download_model <repo_id> [--variant fp16] [--file <name>]`
  prints `progress <0-100>` lines, then `done <HF_HOME>/hub/models--<org>--<name>/snapshots/<revision>`
  (with `--file`, the path of that one file, which is all it fetches).
  - Downloads only what the pipeline loads: configs and tokenizers, plus one set of `.safetensors` per component listed in `model_index.json`.
  - Prefers the variant, which defaults to `fp16` when the repo has it.
  - Skips root single-file checkpoints, `.bin`/`.ckpt`, `non_ema`, onnx, `safety_checker` and assets.
  - Example sizes: SD 1.5 2.0 GB (repo 47 GB), FLUX.1-schnell 33.7 GB (repo 58 GB), FLUX.1-dev 33.8 GB, FLUX.2-klein-4B 16 GB (repo 24 GB),
    Qwen-Image-2.1 33.1 GB.
  - Z-Image-Turbo is 32.8 GB because its transformer is stored in fp32 (it loads as bf16).
  - The files go to the standard Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`), shared with other tools.
    The HF cache holds untouched downloads only: nothing but huggingface_hub downloads writes to it, and the app never
    deletes from it (Remove only forgets the model; `hf cache rm model/<org>/<name>` deletes the files). Anything
    derived from a model, such as future pre-quantized MLX weights, goes under `~/.diffusionbee`.
  - A pipeline class with no model family (anything `family_from_class` doesn't know) is refused after `model_index.json`, before the weights.
    The Models page's "Import From Hugging Face" takes any repo id, so this keeps it from fetching gigabytes the app can't use.
  - Gated repos use the `HF_TOKEN` env var (the app sets it from Settings) or the `hf auth login` token.
    A gated repo without access gives a "gated: accept its license" error.
  - When the Hub can't be reached, it falls back to the cached copy.
- `diffusionbee_backend cached_models <repo_id[:variant]>...`
  prints `cached <repo_id> <snapshot folder>` for each repo whose files (the same set `download_model` picks, at the Hub's current revision) are all in the HF cache. It downloads nothing.
  - Repos with no cache folder are skipped without a network call. Stubs, repos that are offline, gated or unknown, and older revisions are not listed.
  - The app runs it at startup, so catalog models fetched by `hf download` or another app show as downloaded. Models the user removed are skipped; they're kept in `~/.diffusionbee/hidden_hf_models.json` until downloaded again.
- `diffusionbee_backend repo_info <repo_id>`
  prints one JSON line saying what the Models page's "Import From Hugging Face" can take from a repo.
  - `{"diffusers": true}` for a pipeline or ControlNet repo (`model_index.json` or `config.json`), imported with `download_model`.
  - Otherwise `{"diffusers": false, "base_model": "Qwen/Qwen-Image-2.1", "files": [[name, bytes, inspect_model info], ...]}`:
    the root `.safetensors` files the app can run (LoRAs, MLX transformers, sd15/sdxl/sd3/flux/zimage checkpoints),
    classified from their headers, which it reads over the network without downloading the weights.
    `base_model` comes from the model card, `null` if it names none.
    The app gives a LoRA with no family its base's family, and an MLX transformer its base as `base_model_path`.
  - It fails when there is nothing to run, e.g. GGUF-only repos (not supported) or ComfyUI fp8/int8 files,
    and for repos whose Hub `pipeline_tag` isn't image generation (`*-to-image`), e.g. an MLX LLM, whose weights look like an MLX transformer's.
  - Offline, a cached pipeline repo still gives `{"diffusers": true}`.
- `diffusionbee_backend inspect_model <path>` (a `.safetensors` file or a diffusers folder)
  prints one JSON line `{"family": "sdxl", "is_inpaint": false, "type": "sd_model"|"lora"|"controlnet"}`.
  - `family` is `null` when unknown.
  - A diffusers folder of a variant that isn't its family's catalog model adds `defaults`, which the app merges into the model's
    `model_meta_data`: Z-Image base (scheduler shift 6, not Turbo's 3) and FLUX.2 without `is_distilled` (klein base, FLUX.2-dev)
    get 50 steps, cfg 4 and a negative prompt; FLUX.1 without `guidance_embeds` (schnell-like) gets 4 steps, cfg 0.
  - An MLX transformer file (`.scales` tensors) adds `"mlx_bits": 8`, with `family` `null`: it comes from the base model.
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
| qwenimage21 (Qwen-Image-2.1) | 1328 (native 2048) | 40 | 1 (true CFG above 1) | yes | yes | no | no |

Qwen-Image-2.1 (`QwenImage21Pipeline`) is not in diffusers 0.40, so it runs only on the MLX engine (see below). Without
mlx the backend refuses it with "QwenImage21Pipeline needs a newer diffusers than 0.40.0". The default size is 1328 because
2048² takes about 45 min per image on an M4 Pro (1328² about 13 min, 1024² about 7 min).

Apple Silicon settings:
- Device is `mps`. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set.
- DiT families load in bf16. SD 1.5/SDXL load in fp16; the SDXL VAE upcasts itself.
- Attention is SDPA. Attention slicing is on only with 16 GB of RAM or less.
- Engine for zimage, flux2, flux and qwenimage21 diffusers folders: the first of diffusers bf16, MLX (mflux) 8-bit,
  MLX 4-bit whose measured peak (`DIT_PEAK_GB`, bf16 / q8 / q4) is at most 75% of RAM:
  Z-Image 30.5 / 15.1 / 10.4 GB, FLUX.2 23 / 13.3 / 9.6 GB, FLUX.1 39.6 / 22.2 / 14.4 GB (all at 1024²),
  Qwen-Image-2.1 — / 28.5 / 25.1 GB (the most at any size up to 2048²; no diffusers engine).
  When none fits, the job fails with `<model> needs about N GB of RAM` (`N` = q4 peak / 0.75, e.g. FLUX.1 20, Qwen-Image-2.1 34).
  Other families and single-file checkpoints always use diffusers.

  | RAM | zimage | flux2 | flux | qwenimage21 |
  |---|---|---|---|---|
  | 16 GB | MLX 4-bit (~10 GB peak) | MLX 4-bit (~10 GB) | needs about 20 GB | needs about 34 GB |
  | 24 GB | MLX 8-bit (~15 GB) | MLX 8-bit (~13 GB) | MLX 4-bit (~14 GB) | needs about 34 GB |
  | 32 GB | MLX 8-bit | diffusers bf16 | MLX 8-bit (~22 GB) | needs about 34 GB |
  | 48 GB | diffusers bf16 + compile | diffusers bf16 + compile | MLX 8-bit | MLX 8-bit (~28 GB) |

- On diffusers, these transformers get `torch.compile` when RAM is at least 1.5× the bf16 peak (Z-Image 46 GB,
  FLUX.2 35 GB, FLUX.1 60 GB): about 9–15% faster per image. The first step at each new size compiles for ~10 s. If a compiled call fails,
  the backend logs it to stderr and reruns that image eagerly, and the model stays eager. `DIFFUSIONBEE_COMPILE=0` turns it off.
- `DIFFUSIONBEE_RAM_GB=<GB>` in the backend's environment replaces the detected RAM for these choices (and attention slicing),
  to test the tiers on a bigger Mac or to force the MLX engine. The app passes its environment through to the backend.
- The MLX engine runs txt2img and img2img, with LoRAs. Inpaint jobs fail with
  `inpaint is not supported for zimage on the low-memory MLX engine this Mac uses` (qwenimage21 says
  `on the MLX engine`, its only engine); ControlNet is SD 1.5/SDXL only anyway.
  LoRAs are adapter layers on the quantized transformer, not baked in, so changing them takes 1-2 s instead of a reload;
  a step costs up to ~8% more (FLUX.1 with a rank-16 LoRA on every layer; Qwen-Image-2.1's attention-only one ~1%).
  They use mflux's key mappings (diffusers/PEFT, kohya, ComfyUI names); Qwen-Image-2.1 has its own mapping here.
  A LoRA whose keys don't fit the model fails with `No LoRA layers were applied from …`, and the model stays as it was.
  The mflux model config comes from the model's own configs: FLUX.1 dev vs schnell from the transformer's `guidance_embeds`,
  Z-Image base vs Turbo from the scheduler shift, FLUX.2-klein 4B vs 9B from the transformer size (other FLUX.2 models fail at load).
  Qwen-Image-2.1 runs true CFG only when cfg > 1, with the negative prompt (a blank one when empty).
  `small_mod_seed` is ignored, a seed gives a different image than on diffusers, and flux2 img2img starts from the noised
  input image instead of using it as a reference image. Without mlx installed (dev venv) these families use diffusers.
