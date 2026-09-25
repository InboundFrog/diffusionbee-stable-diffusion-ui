# Backend protocol (v3, diffusers)

The Electron app talks to the Python backend over stdin/stdout, one line per message.
The line protocol is unchanged from v2; only the job fields changed.

## Long-running backend (`diffusionbee_backend` with no args)

Backend → app (stdout):

| Line | Meaning |
|---|---|
| `sdbk mltl <text>` | loading title |
| `sdbk mdld` | backend ready |
| `sdbk inrd` | waiting for input |
| `sdbk inwk` | job accepted |
| `sdbk dnpr <0-100>` | progress of current job |
| `sdbk gnms <text>` | status text ("Loading model", "Downloading model 42%") |
| `sdbk nwim {"generated_img_path": "...", "aux_output_image_path": "..."}` | image done |
| `sdbk errr <text>` | job failed |

App → backend (stdin): `b2py t2im {json}`, `b2py rapp <applet> {json}`, `__stop__`.

### Job JSON (`b2py t2im`)

Unchanged fields: `prompt`, `negative_prompt`, `num_imgs`, `img_width`, `img_height`, `num_steps`,
`guidance_scale`, `seed`, `small_mod_seed`, `scheduler`, `input_image`/`input_img`, `mask_image`,
`input_image_strength`, `controlnet_model`, `controlnet_input_image_path`, `control_weight`,
`do_controlnet_preprocess`, `controlnet_inp_img_preprocesser`, `controlnet_inp_img_preprocesser_model_path`,
`sd_mode_override`, `is_clip_skip_2`.

Model fields (replace all `*_tdict_path` fields):

| Field | Meaning |
|---|---|
| `model_path` | Local diffusers folder (has `model_index.json`) or single-file `.safetensors`/`.ckpt`-free checkpoint |
| `model_repo` | HF repo id; used when `model_path` is absent (backend resolves from HF cache, downloads if missing) |
| `model_family` | optional hint: `sd15`, `sdxl`, `sd3`, `flux`, `flux2`, `zimage`, `qwenimage`. Backend autodetects otherwise |
| `inpaint_model_path` / `inpaint_model_repo` | optional dedicated inpainting checkpoint (SD1.5/SDXL) |
| `controlnet_path` / `controlnet_repo` | ControlNet weights (diffusers format) |
| `lora_paths` | list of `.safetensors` LoRA files; `lora_weights` optional parallel list of floats |

A `model_tdict_path` ending in `.tdict` is rejected with a clear "re-import the original .safetensors" error.

`guidance_scale` of `0` or a family without CFG means no negative prompt is used.

## One-shot subcommands

Each prints plain lines on stdout and exits non-zero on failure (last stderr lines are the error).

- `diffusionbee_backend download_model <repo_id> [--variant fp16]`
  prints `progress <0-100>` lines, then `done <local_snapshot_dir>`.
  Uses the HF cache (`HF_HOME` defaults to `~/.diffusionbee/hf`). `HF_TOKEN` env enables gated repos.
- `diffusionbee_backend inspect_model <path>`
  prints one JSON line `{"family": "sdxl", "is_inpaint": false, "type": "sd_model"|"lora"|"controlnet"}`.

## Model families (defaults the UI should use)

| family | size | steps | cfg | negative prompt | img2img | inpaint | controlnet |
|---|---|---|---|---|---|---|---|
| sd15 | 512 | 25 | 7.5 | yes | yes | yes | yes |
| sdxl | 1024 | 30 | 6 | yes | yes | yes | yes |
| sd3 | 1024 | 28 | 4.5 | yes | yes | yes | no |
| flux (FLUX.1) | 1024 | 4 schnell / 28 dev | 0 schnell / 3.5 dev | no | yes | yes | no |
| flux2 (FLUX.2 klein) | 1024 | 4 | 1 | no | yes (reference image) | no | no |
| zimage (Z-Image-Turbo) | 1024 | 8 | 0 | no | yes | no | no |
| qwenimage | 1328 | 30 | 4 | yes | yes | no | no |
