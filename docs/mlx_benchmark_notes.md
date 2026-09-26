# MLX vs diffusers-MPS: benchmark results and recommendation (final, 2026-09-26)

Question: is an MLX engine (mflux) worth adding to DiffusionBee for M4-class Macs, and for which model families?
Machine: M4 Pro (20-core GPU), 48 GB, macOS 27. HF cache: `HF_HOME=~/.diffusionbee/hf`. Backend stack: torch 2.14.0,
diffusers 0.40.0 (`backends/.venv`). MLX stack: mlx 0.32.2, mflux 0.20.0. No file under `backends/`, `electron_app/`,
`packaging/` or `build_mac.sh` was changed. Every number below comes from an uncontended run (a GPU lock was shared
with the packaged-app agent, and a watcher checked for foreign backend load); older contended runs are listed as superseded.

## Recommendation (high confidence for Z-Image and FLUX.2, medium for the unmeasured families)

1. **Add mflux as a second engine, but only for the flow-matching DiT families on Macs with ≤24 GB RAM:**
   - `zimage` and `flux2`: **q4 on ≤16 GB Macs** and **q8 on 24 GB Macs**, always with MLX VAE tiling, `mx.set_cache_limit(2 << 30)`, and an eager `mx.eval(model.parameters())` after load.
   - Qwen-Image-2.1 (future `qwenimage21` family), on any RAM size: mflux is the only engine that runs it; diffusers 0.40 can't.
2. **On ≥32 GB Macs, keep diffusers bf16 for these families and add `pipe.transformer.compile()`:**
   - Z-Image goes from 74.3 to 65.1 s per image (1.14x).
   - FLUX.2 goes from 23.2 to 21.8 s (1.06x).
   - This beats every MLX configuration on speed.
3. **SDXL, SD1.5, SD3.5-medium and FLUX.1-schnell stay on diffusers MPS.**
   - MLX SDXL (mlx-examples) ran at parity (56.6–60.1 s vs 59.5 s), used more memory (20.3 vs 17.6 GB), and is unmaintained example code.
   - mflux has no SD1.5, SDXL or SD3 support.
4. **Speed is not the reason to add MLX.**
   - At bf16, both engines are within ±10% of each other: Z-Image is faster on diffusers, and FLUX.2 is roughly equal.
   - The reason is **memory**. mflux q4 is the only path measured that runs both Z-Image-Turbo and FLUX.2-klein in ≤12 GB with headroom.
   - Load time is a side benefit: mflux loads in 2–9 s, diffusers in 4–26 s (warm file cache).

## The 16 GB question

**A 16 GB Mac should run Z-Image-Turbo and FLUX.2-klein-4B with mflux q4 plus MLX VAE tiling.**

| | Z-Image-Turbo (8 steps) | FLUX.2-klein-4B (4 steps) |
|---|---|---|
| Lifetime peak footprint | **10.4 GB** | **9.6 GB** |
| Time per image, M4 Pro 20-core GPU | **86.5 s** | **24.3 s** |
| Time per image, 16 GB base M4 (estimate) | ~170–200 s | ~50–60 s |
| Load time (from the bf16 diffusers snapshot) | 8.7 s | 2.8 s |

The base-M4 estimate assumes about 2x from half the GPU cores; it was not measured. On that machine, weights read cold from the SSD add ~5–10 s of load time.

**Quality** was judged at 1:1 against bf16 from the same engine and same seed.
- The composition, pose and lighting are the same. Fine detail differs (fur strands, eye tone).
- There are no artifacts, and the VAE tiles leave no seams.
- PSNR vs mflux bf16 is 18.9 dB (Z-Image) and 20.0 dB (FLUX.2). That is the drift of a q4 model, not damage.
- The GGUF Q4_K_M files (k-quants) stay closer to bf16 (23–24 dB) but don't fit (see the table below).

All candidates, measured at 1024² on the M4 Pro. "Peak" is the process's lifetime peak `phys_footprint` (load + generation):

| Path | Z-Image s/img | Z-Image peak | FLUX.2 s/img | FLUX.2 peak | PSNR vs bf16, same engine (ZI / F2) | ≤12 GB? |
|---|---|---|---|---|---|---|
| **mflux q4 + VAE tiling** | **86.5** | **10.4** | **24.3** | **9.6** | 18.9 / 20.0 dB | **yes** |
| mflux q4 from a saved q4 dir (5.5 GB on disk) | 88.3 (load 4.0 s) | 10.5 | – | – | 18.9 | yes |
| mflux q8 + VAE tiling | 89.6 | 15.1 | 24.2 | 13.3 | 36.3 / 24.7 dB | no (fits 24 GB) |
| diffusers GGUF Q4_K_M + quanto int8 TE + VAE tiling + per-step `empty_cache` | 92.9 | 17.75 | 29.5 | 15.25 | 23.7 / 23.2 dB | no |
| … same, + `torch.compile` | – | – | 22.7 | 14.8 | – / 23.2 dB | no |
| … same, + text encoder freed after encoding (FLUX.2 also compiled) | 96.5 | 14.5 | 22.7 | 11.6 (9.5 while generating) | 23.7 / 23.2 dB | Z-Image no. FLUX.2 only just, with no headroom, and the 8 GB bf16 TE must be reloaded and re-quantized for every new prompt |
| diffusers quanto int8 (transformer + TE) | 78.3 | 19.9 | 35.4 | 18.7 | 30.5 / 26.9 dB | no |
| … same, + compile + tiling + `empty_cache` (Z-Image only) | 69.6 | 18.8 | – | – | 30.8 dB | no |
| diffusers torchao int8 weight-only (transformer + TE) | 83.7 | 19.9 | not run (same footprint class as quanto, slower) | – | 27.8 dB | no |
| reference: diffusers bf16 | 74.3 | 30.5 | 23.2 | 22.6 | – | no |
| reference: mflux bf16 | 82.7 | 28.1 | 22.1 | 25.5 | – | no |

**Why diffusers can't get there.** With the text encoder resident, every diffusers configuration adds a +5–9 GB transient during denoising on top of its
post-load footprint. The table shows the post-load footprint, then the denoise peak.

| Configuration | Post-load | Denoise peak |
|---|---|---|
| Z-Image bf16 | 22.9 | 30.5 |
| Z-Image GGUF | 11.5 | 17.6 |
| FLUX.2 GGUF | 10.1 | 15.2 |
| SDXL UNet fp16 | 7.5 | 16.6 |

The transient is not caused by the following:
- **Attention:** MPS SDPA in torch 2.14 is fused and memory-efficient. A probe at the Z-Image shape (1×30×4128×128 bf16) used +0.00 GB without a mask and +0.01 GB with a bool mask.
- **Allocator cache:** `torch.mps.empty_cache()` after every step lowers the peak by at most 1 GB, and `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3` changed nothing.
- **Quantization format:** it is the same for bf16, int8 and GGUF.

The root cause wasn't pinned down. With a quantized transformer and a resident TE, diffusers bottoms out at ~15 GB for FLUX.2 and ~18 GB for Z-Image.

Freeing the text encoder is the one lever that moves the transient:
- **FLUX.2 GGUF + compile**, with the prompt encoded before the timed runs:
  - TE kept resident: 15.5 GB peak.
  - TE freed: 9.5 GB peak.
  - So the transient is tied to the TE being resident during denoising, not to the encode pass itself.
- **Z-Image**, with the TE freed, still peaks at 14.4 GB.
- The GGUF + freed-TE route also has two practical costs:
  - Loading the TE (bf16 from disk, then quanto int8) pushes the lifetime peak to 11.6 GB, and that happens for every new prompt.
  - It needs an extra GGUF download.

**Is the second engine justified? Yes.**
- A diffusers-only quantized path doesn't reach the 12 GB budget for Z-Image (best: 14.5 GB).
- For FLUX.2 it reaches 11.6 GB, but only by stacking GGUF, compile, per-step `empty_cache`, VAE tiling and a TE reload per prompt, and that leaves no headroom.
- Either way it costs an extra 2.6–5 GB GGUF download.
- mflux reaches it from the same HF snapshot the backend already downloads.
- The cost is:
  - +212 MB of bundle (mlx + mlx-metal + mflux);
  - one ~60-line adapter class;
  - three small branches in `Engine` (sketch below).
- Known limits of the MLX path:
  - img2img at 1024² was not measured. The mflux VAE *encoder* may not be tiled, so it could add memory.
  - A given seed produces a different image than diffusers does, because the RNG differs.
  - `small_mod_seed` is a no-op.
  - Inpainting and ControlNet stay on diffusers.
  - LoRAs are supported by mflux only when passed in at load time (they are baked in). The sketch rejects them for now.

## Per-family table

| Family | Measured | ≥32 GB | 24 GB | ≤16 GB | MLX vs diffusers speed (bf16) | Notes |
|---|---|---|---|---|---|---|
| Z-Image-Turbo (6B DiT + Qwen3-4B TE), 8 steps | yes | diffusers bf16 + compile: 65.1 s, 30.4 GB | mflux q8 + tiling: 89.6 s, 15.1 GB | **mflux q4 + tiling: 86.5 s, 10.4 GB** | MLX 11% slower (82.7 vs 74.3 s); 27% slower than compiled | Must be bf16: fp16 gives a black (NaN) image |
| FLUX.2-klein-4B (+ Qwen3-4B TE), 4 steps | yes | diffusers bf16 + compile: 21.8 s, 22.8 GB | mflux q8 + tiling: 24.2 s, 13.3 GB | **mflux q4 + tiling: 24.3 s, 9.6 GB** | parity (22.1 vs 23.2 s eager, 21.8 s compiled) | Quantization costs almost no speed on either engine |
| SDXL base, 30 steps, cfg 6 | yes | diffusers fp16: 59.5 s, 17.6 GB | same | same (the backend already slices attention for UNets when RAM ≤16 GB) | parity: mlx-examples 56.6–60.1 s (1.83–1.94 s/step) vs 59.5 s | MLX uses more memory: 20.3 GB with the cache capped (35.8 GB uncapped) vs 17.6. Not worth vendoring example code with a different (Euler-ancestral) sampler |
| SD1.5 | no | diffusers | diffusers | diffusers | – | 1 GB-class UNet at 512²: memory isn't a problem. No mflux support. Inferred from SDXL: MLX would bring ≤5% |
| FLUX.1 schnell / dev (12B + T5-XXL) | yes (below) | 32–48 GB: mflux q8: 57.6 s schnell, 420 s dev, 22.2 GB; diffusers bf16 (39.6 GB) only from 53 GB | mflux q4: 56.0 s, 14.4 GB | not a 16 GB model (needs about 20 GB) | MLX q8 57.6 s vs eager diffusers 62.4 s (which was swapping) | diffusers bf16 swaps on 48 GB (39.6 GB peak, +3 GB swap) |
| SD3.5-medium (2.5B MMDiT + T5) | no (gated, not cached) | diffusers | diffusers | diffusers | – | mflux has no SD3. Memory is manageable in diffusers (T5 can be dropped) |
| Qwen-Image-2.1 (7.1B MMDiT + 8.8B Qwen3-VL TE) | yes (below) | 36–48 GB: **mflux q8**, 28.5 GB; below 34 GB it doesn't fit | not a 24 GB model (q4 25.1 GB) | not a 16 GB model | – | diffusers 0.40 has no pipeline for it, so mflux 0.20 is the only engine (`qwenimage21` family). mflux never quantizes the 17.5 GB bf16 text encoder, so q4 saves only 3.4 GB |

## Measurements (all clean)

Every run is 1024×1024, batch 1, seed 42, with the prompt "A red fox sitting in a snowy forest at golden hour, detailed fur, photorealistic".
Each run had one warm-up and 2–3 timed runs; the best is shown. The runs were within 1% of each other unless noted.

Column meanings:
- **load:** model load to ready. The OS file cache was warm.
- **pre:** text encoding and setup before the first step.
- **post-load:** the footprint after load and `empty_cache`.
- **gen peak:** the footprint peak during the timed runs (sampled every 20 ms).
- **lifetime:** the `/usr/bin/time -l` peak memory footprint.
- **PSNR:** the same-seed PSNR vs that engine's bf16 image.

### Z-Image-Turbo, 8 steps, cfg 0

| Run | Engine | Config | Load s | s/step | Total s | Pre s | Decode s | Post-load GB | Gen peak GB | Lifetime GB | PSNR dB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zi_d_bf16_fp | diffusers | bf16 | 19.3 | 9.03 | 74.3 | 0.7 | 1.3 | 22.9 | 30.5 | 30.5 | ref |
| zi_d_compile | diffusers | bf16 + compile (warm-up 75.9 s) | 25.1 | 7.89 | **65.1** | 0.7 | 1.3 | 22.9 | 30.4 | 30.4 | 37.8 |
| zi_d_quanto8 | diffusers | quanto int8 transformer + TE | 24.1 | 9.51 | 78.3 | 0.9 | 1.3 | 12.65 | 19.9 | 19.9 | 30.5 |
| zi_d_quanto8_c | diffusers | quanto int8 + compile + tiling + empty_cache | 21.7 | 8.40 | 69.6 | 0.9 | 1.4 | 12.64 | 18.8 | 18.8 | 30.8 |
| zi_d_torchao8 | diffusers | torchao int8wo transformer + TE | 26.4 | 10.17 | 83.7 | 1.1 | 1.3 | 12.65 | 19.9 | 19.9 | 27.8 |
| zi_d_gguf4 | diffusers | GGUF Q4_K_M + quanto int8 TE | 13.1 | 11.30 | 92.6 | 1.0 | 1.3 | 11.54 | 18.75 | 18.75 | 23.7 |
| zi_d_gguf4_opt | diffusers | … + VAE tiling + empty_cache | 12.5 | 11.32 | 92.9 | 0.9 | 1.4 | 11.54 | 17.75 | 17.75 | 23.7 |
| zi_d_gguf4_wm | diffusers | … + `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3` | 12.4 | 11.30 | 92.7 | 0.9 | 1.4 | 11.54 | 17.64 | 17.64 | 23.7 |
| zi_d_gguf4_dropte | diffusers | … + TE freed after encoding once | 12.5 | 11.89 | 96.5 | – | 1.4 | 10.9 | 14.4 | 14.5 | 23.7 (identical to gguf4_opt) |
| zi_mflux_bf16_r | mflux | bf16 | 7.3 | 10.12 | 82.7 | 0.7 | 1.0 | 21.7 | 28.0 | 28.1 | ref |
| zi_mflux_q8_r | mflux | q8 | 7.3 | 10.94 | 89.2 | 0.7 | 1.0 | 12.7 | 19.2 | 19.5 | 37.5 |
| zi_mflux_q8_tile | mflux | q8 + VAE tiling | 6.0 | 10.96 | 89.6 | 0.6 | 1.3 | 12.6 | 15.1 | 15.1 | 36.3 |
| zi_mflux_q4_r | mflux | q4 | 5.4 | 10.78 | 87.9 | 0.7 | 1.0 | 8.0 | 14.7 | 14.8 | 18.9 |
| **zi_mflux_q4_tile** | mflux | **q4 + VAE tiling** | 8.7 | 10.58 | **86.5** | 0.7 | 1.3 | 7.96 | **10.2** | **10.4** | 18.9 |
| zi_mflux_save_q4 | mflux | `save_model` q4: 2.3 s, 5.5 GB on disk (TE 2.1, transformer 3.2, VAE 0.16) | 7.0 | – | – | – | – | 7.9 | – | 8.9 | – |
| zi_mflux_q4_saved | mflux | load the saved q4 dir + tiling | **4.0** | 10.80 | 88.3 | 0.6 | 1.3 | 6.1 | 10.4 | 10.5 | 18.9 |
| zi_d_fp16 | diffusers | fp16 | 26.7 | 9.52 | 80.0 | 1.4 | 2.4 | – | – | 29.3 | **black image** |
| zi_d_slicing / zi_d_vaetile | diffusers | bf16 + attention slicing / VAE tiling (session 1) | 22.6 / 23.9 | 9.07 / 9.60 | 77.1 / 81.5 | 2.2 | 2.4 | – | – | 30.4 / 29.6 | slicing bit-identical (no-op) |

Superseded (session 1: contended or before the MLX cache cap): zi_diffusers_bf16 (144 s, swap-bound), zi_diffusers_bf16_b (92.5 s,
text encoder paged out), zi_d_bf16_c (82.5 s), zi_mflux_bf16_c (86.4 s), zi_mflux_q8 (102.8 s), zi_mflux_q4 (98.7 s). These ran with 12–18 GB of
swap in use from other agents' processes. The *_r reruns ran with 0.5–3 GB of swap.

### FLUX.2-klein-4B, 4 steps, guidance 1.0

| Run | Engine | Config | Load s | s/step | Total s | Pre s | Decode s | Post-load GB | Gen peak GB | Lifetime GB | PSNR dB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f2_d_bf16_fp | diffusers | bf16 | 9.3 | 5.30 | 23.2 | 0.7 | 1.3 | 16.3 | 22.6 | 22.6 | ref |
| f2_d_compile | diffusers | bf16 + compile (warm-up 30.7 s) | 4.0 | 4.94 | **21.8** | 0.7 | 1.3 | 16.3 | 22.8 | 22.8 | 39.2 |
| f2_d_quanto8 | diffusers | quanto int8 transformer + TE | 12.9 | 7.48 | 35.4 [44.4, 35.4] | 4.3 | 1.3 | 12.2 | 18.7 | 18.7 | 26.9 |
| f2_d_gguf4 | diffusers | GGUF Q4_K_M + quanto int8 TE | 6.8 | 12.71 | 48.5 [48.5, 55.2] | – | 2.5 | 10.1 | 16.6 | 16.6 | 23.2 |
| f2_d_gguf4_opt | diffusers | … + VAE tiling + empty_cache | 9.1 | 6.77 | 29.5 | 1.0 | 1.4 | 10.1 | 15.25 | 15.25 | 23.2 |
| f2_d_gguf4_wm | diffusers | … + low watermark 0.3 | 9.1 | 6.77 | 29.4 | 1.0 | 1.4 | 10.1 | 15.15 | 15.26 | – |
| f2_d_gguf4_c | diffusers | … + compile | 9.7 | 5.08 | 22.7 | 1.0 | 1.4 | 10.1 | 14.8 | 14.8 | 23.2 |
| f2_d_gguf4_preenc | diffusers | … + compile, prompt encoded once before the runs, TE kept | 10.1 | 5.30 | 22.6 | – | 1.5 | 10.9 | 15.5 | 15.5 | 23.2 |
| f2_d_gguf4_dropte | diffusers | … + compile, TE freed after encoding once | 10.1 | 5.32 | 22.7 | – | 1.4 | 10.2 | **9.5** | 11.6 (load) | 23.2 (identical to gguf4_c) |
| f2_mflux_bf16_r | mflux | bf16 | 4.1 | 5.06 | 22.1 | 0.5 | 1.5 | 15.2 | 25.4 | 25.5 | ref |
| f2_mflux_q8_r | mflux | q8 | 2.5 | 5.66 | 24.4 | 0.3 | 1.4 | 10.7 | 18.5 | 18.5 | 26.5 |
| f2_mflux_q8_tile | mflux | q8 + VAE tiling | 3.8 | 5.55 | 24.2 | – | – | 10.3 | 13.1 | 13.3 | 24.7 |
| f2_mflux_q4_r | mflux | q4 | 1.8 | 5.52 | 23.9 | 0.4 | 1.4 | 7.0 | 15.4 | 15.4 | 20.4 |
| **f2_mflux_q4_tile_r** | mflux | **q4 + VAE tiling** | 2.8 | 5.59 | **24.3** | 0.3 | 1.6 | 7.05 | **9.4** | **9.6** | 20.0 |

Superseded (session 1): f2_diffusers_bf16 / _b (25–30 s, contended), f2_mflux_bf16 / _b (80–163 s, from the uncapped MLX cache),
f2_mflux_bf16_c (25.3 s), f2_mflux_q8 (27.0 s, contended), f2_mflux_q4 (26.8 s, 13.0 GB).

### SDXL base 1.0, 30 steps, cfg 6

| Run | Engine | Config | Load s | s/step | Total s | Decode s | Post-load GB | Peak GB |
|---|---|---|---|---|---|---|---|---|
| sdxl_d_fp16 | diffusers | fp16 UNet/TE, backend defaults | 5.4 | 1.93 | 59.5 | 1.6 | 7.5 | 17.6 (denoise 16.6, decode 17.6) |
| sdxl_mlx | mlx-examples | fp16 UNet/TE, fp32 VAE, Euler-ancestral, no MLX cache cap | 1.2 | 1.83 | 56.6 | 1.6 | – | 35.8 (MLX active peak 19.9) |
| sdxl_mlx_c2 | mlx-examples | same, `mx.set_cache_limit(2 GB)` | 1.1 | 1.94 | 60.1 | 1.8 | – | 20.3 |

Both SDXL images show a similar fox. PSNR across engines is 10 dB, which is expected because the samplers and RNGs differ. Capping the cache left the output bit-identical.

### FLUX.1 and Qwen-Image-2.1 (2026-09-26, mflux 0.20.0, the backend's own Engine / MfluxPipe)

Same prompt and seed. Measured with a script that drives `Engine.prepare` / `render` (mflux runs with VAE tiling, as
in the backend). Peak is the lifetime peak memory footprint from `/usr/bin/time -l` (it includes load). Total is per image
and includes text encoding and decode. The OS file cache was warm. Swap in use was 2–8 GB from other apps.

| Run | Engine | Steps / cfg / size | Load s | Total s | Lifetime peak GB | Notes |
|---|---|---|---|---|---|---|
| f1s_bf16 | diffusers bf16 eager | 4 / 0 / 1024 | 28.8 | 62.4 [62.4, 64.4] | 39.6 | swap grew 5.3 → 8.3 GB: over 75% of 48 GB |
| f1s_q8 | mflux q8 | 4 / 0 / 1024 | 6.5 | **57.6** [57.6, 58.1] | 22.2 | |
| f1s_q4 | mflux q4 | 4 / 0 / 1024 | 6.3 | 56.0 [56.0, 57.1] | **14.4** | |
| f1dev_q8 | mflux q8 | 28 / 3.5 / 1024 | 6.1 | 419.7 (14.9 s/step) | 22.1 | |
| q21_q8_1024 | mflux q8 | 40 / 1 / 1024 | 7.6 | 472.4 (11.8 s/step) | 28.4 | |
| q21_q4_1024 | mflux q4 | 40 / 1 / 1024 | 7.0 | 423.6 (10.6 s/step) | 25.1 | |
| q21_q8_1328 | mflux q8 | 40 / 1 / 1328 | 6.7 | 775.5 (19.4 s/step) | 28.3 | the catalog default size |
| q21_q8_2048 | mflux q8 | 40 / 1 / 2048 | 6.7 | 2639.8 (66 s/step) | 28.3 | native size |
| q21_q4_2048 | mflux q4 | 40 / 1 / 2048 | 7.2 | 2716.2 | 24.8 | |

- FLUX.1: q8 and q4 give clean images of the same quality (PSNR q4 vs q8 20.3 dB). Against diffusers it is ~10 dB because
  mflux draws different noise, so the composition differs. dev at q8 gives a good golden-hour fox.
  Compile doesn't apply on 48 GB (it needs 1.5 × 39.6 = 60 GB).
- Qwen-Image-2.1: q8 at 2048² is excellent (sharp fur, no artifacts). q4 is the same quality with slightly less golden light.
  At 1024² both are clean, q4 slightly more saturated. The peak barely depends on size or bits because the
  17.5 GB bf16 Qwen3-VL text encoder (mflux `skip_quantization`) dominates. bf16 was not run: 33 GB of weights plus
  activations is about 40–46 GB, and diffusers 0.40 can't run it either, so `DIT_PEAK_GB` records it as `None`.

## Findings that matter for the backend today (with or without MLX)

1. **Offline fallback bug in `fetch_repo`.** `download_model` downloads with `revision=info.sha`, so the HF cache never gets a `refs/main`.
   - The offline branch `snapshot_download(repo, local_files_only=True)` then raises `LocalEntryNotFoundError`.
   - Confirmed for FLUX.2-klein-4B and SDXL base; Z-Image happens to have a `refs/main`.
   - So with no network, a model that has already been downloaded fails to load.
   - Suggested fix: fall back to the single snapshot dir, e.g. `glob(hub/models--org--name/snapshots/*)`, or record the sha next to the model. Fixed in 7f99760: offline, fetch_repo uses the newest cached snapshot.
2. **`torch.compile` works on MPS in torch 2.14.**
   - Inductor emits only Metal kernels (`async_compile.metal`), so no C++ toolchain is needed.
   - It costs ~8–11 s on the first image and recompiles per resolution.
   - Output drift vs eager is 37.8–39.2 dB, which is invisible.
   - Gains: Z-Image 1.14x, FLUX.2 1.06x, int8 Z-Image 1.12x, GGUF FLUX.2 1.30x.
   - In the PyInstaller bundle it needed two fixes (packaging/): the `torch/include/c10/metal` headers, which Inductor
     inlines into its kernels ("failed to compile #include <c10/metal/utils.h>"), and absolute `co_filename`s via a
     runtime hook. PyInstaller stores relative ones, so Dynamo can't tell torch internals from model code and fails
     with "maximum recursion depth exceeded". Bundle, 1024², cold Inductor cache, shared machine: Z-Image 81.6 → 74.6 s
     per image (first image +3.6 s), FLUX.2 25.9 → 21.9 s.
3. **Z-Image fp16 gives an all-black (NaN) image.** It must stay bf16, which `DIT_FAMILIES` already does. Keep it that way.
4. **Attention slicing is a no-op for the DiT families**: the Z-Image and FLUX.2 transformers have no `set_attention_slice`. The backend already limits it to UNets.
5. **VAE tiling in diffusers doesn't help at 1024²** (29.6 vs 29.9 GB). MLX VAE tiling (`TilingConfig`) does: –4.5 GB (Z-Image) and –6 GB (FLUX.2), at the same speed, with no seams.
6. **Per-step `torch.mps.empty_cache()` fixes the diffusers GGUF slowdown on FLUX.2** (12.7 → 6.8 s/step). Without it, GGUF on FLUX.2 was 2.4x slower than bf16.
7. **The MLX buffer cache must be capped:** `mx.set_cache_limit(2 << 30)`. Uncapped, it held ~20 GB extra after one 1024² image and thrashed swap (25–60 s/step).
   Also evaluate the mflux parameters eagerly after load (`mx.eval(model.parameters())`). Left lazy, the first image holds bf16 and q4 together: Z-Image peaked at 29.9 GB instead of 11.2 GB.
8. **The text encoder gets paged out under memory pressure** (pre-step went from 1 s to 11–60 s in session-1 contended runs). Never hold two pipelines, and on a switch unload one engine before loading the other.
9. **Z-Image steps:** the default of 8 (commit c35af5a) is correct. In diffusers 0.40, n steps means n transformer passes.

## Libraries (Sept 2026)

| Library | Version / status | License | Weights |
|---|---|---|---|
| mflux | 0.20.0 (2026-09-21), active | MIT | Reads the diffusers-layout HF snapshot directly (`model_path=<snapshot dir>`) and quantizes at load, so there is no separate download. Optional `save_model()` writes a pre-quantized dir (Z-Image q4 = 5.5 GB). Models: Z-Image(-Turbo), FLUX.2 klein 4B/9B, FLUX.1, Qwen-Image, **Qwen-Image-2.1**, Krea 2, ERNIE-Image, FIBO, SeedVR2… No SD1.5/SDXL/SD3 |
| mlx / mlx-metal | 0.32.2 (2026-08-25) | MIT | – |
| mlx-examples `stable_diffusion` | Example code, last touched 2024-11 | MIT | Reads HF SD2.1/SDXL safetensors. Needs a patch for SDXL-base 1024 (`text_time`) and vendoring (no pip package) |
| DiffusionKit (argmaxinc) | 0.5.2 (2024-12), **archived 2025-04** | MIT | Own MLX format. Avoid |
| apple ml-stable-diffusion (Core ML) | 1.1.1 (2024-05) | MIT (+ coremltools BSD-3) | Separately compiled Core ML weights (SDXL ~7 GB, SD1.5 ~9 GB per variant), not shared with the HF cache. Skipped |
| optimum-quanto | 0.2.7 (2025-03, maintenance) | Apache-2.0 | Quantizes diffusers modules in place |
| torchao | 0.18.0 | BSD-3 | Quantizes in place |
| gguf | 0.19.0 | MIT | unsloth Z-Image-Turbo-GGUF (Q4_K_M 5.0 GB) and FLUX.2-klein-4B-GGUF (Q4_K_M 2.6 GB), Apache-2.0: an extra download on top of the snapshot |

All are permissive and compatible with the repo's AGPL-3.0.

**Bundle cost of mflux: +212 MB.**
- Install with `pip install --no-deps mlx==0.32.2 mlx-metal==0.32.2 mflux==0.20.0`, plus the small `piexif`, `toml` and `platformdirs` packages. cv2, transformers and safetensors are already bundled.
- mlx + mlx-metal is 206 MB: `mlx/lib` holds libmlx.dylib and mlx.metallib at 199 MB.
- mflux is 6.4 MB.
- In `packaging/diffusionbee_backend.spec`, add `'mlx', 'mflux'` to `LAZY_PKGS`. Its `collect_all` picks up the dylib, the metallib and the package data.

## Integration sketch (winner: mflux adapter, against `backends/stable_diffusion/diffusionbee_backend.py`)

The adapter puts an mflux model behind the part of the diffusers pipeline interface that `Engine.render()` uses, so `render()` and `run_job()` stay unchanged. It was
**verified end to end** by the scratch self-check `mflux_pipe.py` below, which calls it exactly the way `render()` does:
- the `callback_on_step_end` progress callback;
- `pipe._interrupt` to stop;
- img2img via `image`/`strength`;
- a torch `Generator` for the seed.

Result: all assertions pass for both FLUX.2 and Z-Image, loading from the diffusers snapshot dir. I checked the outputs by eye: txt2img, and img2img that keeps the composition.

**One trap found this way:** call `mx.eval(model.parameters())` right after constructing the mflux model.
- Otherwise MLX quantizes lazily during the first image, holding the bf16 and q4 weights together.
- Lifetime peak with the lazy load: 29.9 GB (Z-Image) and 14.7 GB (FLUX.2).
- With the eager eval: 11.2 GB and 10.0 GB. That run included the 512² img2img VAE encode.
- The benchmarks above already evaluated eagerly.

New file `backends/stable_diffusion/mflux_pipe.py` (the verified scratch copy is inlined in the appendix):
- `load(family, src, bits)` builds `ZImageTurbo` / `Flux2Klein` from the same snapshot dir `load_base()` receives, evaluates the parameters eagerly, and turns on VAE tiling.
- `MfluxPipe.__call__(prompt, num_inference_steps, guidance_scale, width, height, generator, callback_on_step_end, image, strength)`:
  - seed = `generator.initial_seed()`;
  - the mflux in-loop callback evaluates the latents (MLX is lazy), forwards the step to `callback_on_step_end`, and raises `KeyboardInterrupt` when `_interrupt` is set; mflux turns that into `StopImageGenerationException`;
  - img2img writes the init image to a temp file with `image_strength = 1 - strength`;
  - it returns `SimpleNamespace(images=[img])` and calls `mx.clear_cache()` after every image.

Changes in `diffusionbee_backend.py`:

```python
MLX_FAMILIES = {"zimage", "flux2"}  # + "qwenimage21" once added; mflux is its only engine

# Engine.__init__
ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
self.low_ram = ram <= 16 * 1024 ** 3
self.mlx_bits = 4 if self.low_ram else 8 if ram <= 24 * 1024 ** 3 else None  # None: diffusers bf16 (+ compile)
# ponytail: RAM-tier switch only; add a UI "engine" override if users ask

# Engine.load_base: in the os.path.isdir(src) branch, right after `family = family_hint or family_from_class(cls_name)`
if family in MLX_FAMILIES and self.mlx_bits:
    try:
        import mflux_pipe
    except ImportError:
        pass  # dev venv without mlx: stay on diffusers
    else:
        self.base, self.base_key, self.family = mflux_pipe.load(family, src, self.mlx_bits), src, family
        return
# (the diffusers path continues; on ≥32 GB add `pipe.transformer.compile()` for DIT_FAMILIES after pipe.to())

# Engine.prepare: replace the set_loras / mode_pipe / set_scheduler lines
if type(self.base).__name__ == "MfluxPipe":
    if mode == "inpaint" or d.get("lora_paths"):
        raise ValueError("Inpainting and LoRAs need more memory than this Mac has for this model")  # ponytail: mflux takes lora_paths at load; wire it up when asked
    pipe = self.base
else:
    self.set_loras(d.get("lora_paths") or [], d.get("lora_weights"))
    pipe = self.mode_pipe(mode, controlnet)
    self.set_scheduler(pipe, d)
# (ControlNet already raises for non-sd15/sdxl families; call_params(pipe) filters kw against MfluxPipe.__call__ as before)

# Engine.unload: after gc.collect()
if "mlx.core" in sys.modules:
    sys.modules["mlx.core"].clear_cache()
```

Notes:
- `render()` needs no change.
  - `on_step`'s `torch.mps.synchronize()` is harmless: MPS is idle.
  - `noise_variation()` finds no `randn_tensor` in the adapter's module, so `small_mod_seed` is a no-op on MLX.
  - Stop returns `images=[None]` with `stopped=True`, so `render()` returns None.
- `test_backend.py` could get one MLX case: a 512², 2-step FLUX.2 q4 job. It would be skipped when mlx isn't importable.
- Qwen-Image-2.1 needs a family entry (`qwenimage21`), class detection from its `model_index.json`, and an mflux `qwen21` branch in `load()`. None of that was tested: the weights weren't downloaded.
- Optional: after the first q4 load, `model.save_model(dir)` (2.3 s, 5.5 GB) cuts later loads to 4 s. The bf16 originals (~20 GB for Z-Image) could then be deleted on a 16 GB machine, which typically also has a small SSD.

## Method and reproduction

Scratch dir: `/private/tmp/claude-501/-Users-mike-work-InboundFrog-diffusionbee-stable-diffusion-ui/60e1a826-bf96-4f4d-aa84-45fdb645e50b/scratchpad/mlx`. It may be gone; every script is inlined below.

```sh
mkdir -p $DIR && cd $DIR && mkdir -p logs out saved
uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r <repo>/backends/stable_diffusion/requirements.txt
uv pip install --python .venv/bin/python mflux==0.20.0 optimum-quanto==0.2.7 torchao==0.18.0 gguf==0.19.0   # mlx 0.32.2
# mlx-examples SDXL, vendored as package mlx_sd and patched for 1024px micro-conditioning:
mkdir -p mlx_sd && for f in __init__.py clip.py config.py model_io.py sampler.py tokenizer.py unet.py vae.py; do
  gh api repos/ml-explore/mlx-examples/contents/stable_diffusion/stable_diffusion/$f --jq .content | base64 -d > mlx_sd/$f; done
sed -i '' 's/\[\[512, 512, 0, 0, 512, 512.0\]\]/[[1024, 1024, 0, 0, 1024, 1024.0]]/g' mlx_sd/__init__.py
```

- Run everything with **bash**, not zsh: zsh doesn't word-split `$ZI`.
- Every run goes through `run.sh <name> <script> <args>`. It:
  - waits until no other backend or driver process is busy;
  - takes the shared GPU lock;
  - flags contention;
  - appends one JSON line per run to `results.jsonl`.
- diffusers baselines use `PY=<repo>/backends/.venv/bin/python`. Quantized diffusers runs and all MLX runs use the scratch venv.
- Quality: `python quality.py ref.png img...` prints the PSNR vs a reference image. Only compare images from the same engine and the same seed.
- Measured commands (the GGUF paths are the unsloth snapshots in the HF cache):

```sh
ZI="bench_diffusers.py Tongyi-MAI/Z-Image-Turbo --steps 8 --runs 2"
F2="bench_diffusers.py black-forest-labs/FLUX.2-klein-4B --steps 4 --cfg 1.0 --runs 3"
GZ=$(ls ~/.diffusionbee/hf/hub/models--unsloth--Z-Image-Turbo-GGUF/snapshots/*/z-image-turbo-Q4_K_M.gguf)
GF=$(ls ~/.diffusionbee/hf/hub/models--unsloth--FLUX.2-klein-4B-GGUF/snapshots/*/flux-2-klein-4b-Q4_K_M.gguf)
BK=<repo>/backends/.venv/bin/python
PY=$BK ./run.sh zi_d_bf16_fp $ZI --out out/zi_d_bf16_fp.png
PY=$BK ./run.sh zi_d_compile $ZI --compile --out out/zi_d_compile.png
./run.sh zi_d_quanto8 $ZI --quant quanto-int8 --quant-te --out out/zi_d_quanto8.png
./run.sh zi_d_quanto8_c $ZI --quant quanto-int8 --quant-te --vae-tiling --empty-cache --compile --out out/zi_d_quanto8_c.png
./run.sh zi_d_torchao8 $ZI --quant torchao-int8wo --quant-te --out out/zi_d_torchao8.png
./run.sh zi_d_gguf4_opt $ZI --quant gguf:$GZ --quant-te --vae-tiling --empty-cache --out out/zi_d_gguf4_opt.png
./run.sh zi_d_gguf4_dropte $ZI --quant gguf:$GZ --quant-te --vae-tiling --empty-cache --drop-te --out out/zi_d_gguf4_dropte.png
./run.sh zi_mflux_bf16_r bench_mflux.py zimage --steps 8 --runs 2 --out out/zi_mflux_bf16_r.png
./run.sh zi_mflux_q8_tile bench_mflux.py zimage --steps 8 -q 8 --vae-tiling --runs 2 --out out/zi_mflux_q8_tile.png
./run.sh zi_mflux_q4_tile bench_mflux.py zimage --steps 8 -q 4 --vae-tiling --runs 2 --out out/zi_mflux_q4_tile.png
./run.sh zi_mflux_save_q4 bench_mflux.py zimage --steps 8 -q 4 --save saved/zi_q4 --out x
./run.sh zi_mflux_q4_saved bench_mflux.py zimage --steps 8 --model-path saved/zi_q4 --vae-tiling --runs 2 --out out/zi_mflux_q4_saved.png
PY=$BK ./run.sh f2_d_bf16_fp $F2 --out out/f2_d_bf16_fp.png
PY=$BK ./run.sh f2_d_compile $F2 --compile --out out/f2_d_compile.png
./run.sh f2_d_quanto8 $F2 --quant quanto-int8 --quant-te --out out/f2_d_quanto8.png
./run.sh f2_d_gguf4_c $F2 --quant gguf:$GF --quant-te --vae-tiling --empty-cache --compile --out out/f2_d_gguf4_c.png
./run.sh f2_d_gguf4_dropte $F2 --quant gguf:$GF --quant-te --vae-tiling --empty-cache --compile --drop-te --out out/f2_d_gguf4_dropte.png
./run.sh f2_d_gguf4_preenc $F2 --quant gguf:$GF --quant-te --vae-tiling --empty-cache --compile --pre-encode --out out/f2_d_gguf4_preenc.png
./run.sh f2_mflux_q8_tile bench_mflux.py flux2 --steps 4 -q 8 --vae-tiling --runs 3 --out out/f2_mflux_q8_tile.png
./run.sh f2_mflux_q4_tile_r bench_mflux.py flux2 --steps 4 -q 4 --vae-tiling --runs 3 --out out/f2_mflux_q4_tile_r.png
PY=$BK ./run.sh sdxl_d_fp16 bench_diffusers.py stabilityai/stable-diffusion-xl-base-1.0 --steps 30 --cfg 6 --dtype fp16 --out out/sdxl_d_fp16.png
./run.sh sdxl_mlx_c2 bench_mlx_sdxl.py --out out/sdxl_mlx_c2.png          # sdxl_mlx: same, before the mx.set_cache_limit line was added
./run.sh adapter_flux2_eval mflux_pipe.py flux2; ./run.sh adapter_zimage_eval mflux_pipe.py zimage
./run.sh diag_zi_snap diag_load.py zimage snap    # mflux load peak: HF repo id vs snapshot path, with/without torch imported (all ~9.7 GB)
```


## Appendix: scripts (verbatim from the scratch dir)

### `run.sh`

```bash
#!/bin/bash
# usage: [PY=python] run.sh <name> <script args...>  -> logs/<name>.log, appends RESULT + peak footprint to results.jsonl
# Waits for the other agent's GPU jobs (backend generations) to finish, and flags a run that overlapped one.
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
name=$1; shift
# busy = another agent's backend/driver process using >15% CPU (an idle Electron-owned backend is fine)
foreign() { pgrep -f "diffusionbee_backend|drive.py" | while read p; do ps -o %cpu=,command= -p $p; done | grep -v -E "download_model|inspect_model|zsh" | awk '$1 > 15' | grep -q . ; }
quiet=0; while [ $quiet -lt 3 ]; do if foreign; then quiet=0; else quiet=$((quiet+1)); fi; sleep 10; done
LOCK=/private/tmp/claude-501/-Users-mike-work-InboundFrog-diffusionbee-stable-diffusion-ui/60e1a826-bf96-4f4d-aa84-45fdb645e50b/scratchpad/gpu.lock
until mkdir "$LOCK" 2>/dev/null; do
  # stale: older than 45 min and owner pid dead
  o=$(awk '{print $2}' "$LOCK/owner" 2>/dev/null); ts=$(awk '{print $3}' "$LOCK/owner" 2>/dev/null)
  [ -n "$ts" ] && [ $(( $(date +%s) - ts )) -gt 2700 ] && ! kill -0 "$o" 2>/dev/null && rm -rf "$LOCK" && continue
  sleep 10
done
echo "bench $$ $(date +%s)" > "$LOCK/owner"; trap 'rm -rf "$LOCK"' EXIT
rm -f logs/$name.contended; /usr/bin/time -l $PY "$@" > logs/$name.log 2>&1 &
pid=$!; contended=0
while kill -0 $pid 2>/dev/null; do
  foreign && contended=1 && echo 1 > logs/$name.contended
  sleep 5
done
wait $pid; rc=$?
[ -f logs/$name.contended ] && contended=1
res=$(grep '^RESULT ' logs/$name.log | sed 's/^RESULT //')
fp=$(grep 'peak memory footprint' logs/$name.log | awk '{print $1}')
rss=$(grep 'maximum resident set size' logs/$name.log | awk '{print $1}')
swap=$(sysctl -n vm.swapusage | awk '{print $6}')
python3 -c "import json,sys; r=json.loads(sys.argv[2] or '{}'); r.update(name=sys.argv[1], rc=int(sys.argv[5]), contended=bool(int(sys.argv[6])), swap_used_after=sys.argv[7], peak_footprint_gb=round(int(sys.argv[3] or 0)/2**30,2), max_rss_gb=round(int(sys.argv[4] or 0)/2**30,2)); print(json.dumps(r))" "$name" "$res" "$fp" "$rss" "$rc" "$contended" "$swap" | tee -a results.jsonl
[ -z "$res" ] && grep -v '^ ' logs/$name.log | tail -5
exit 0
```

### `fp.py`

```python
"""Process phys_footprint (what Activity Monitor calls Memory) via proc_pid_rusage, plus a peak sampler thread."""
import ctypes, os, threading, time
_proc = ctypes.CDLL("/usr/lib/libproc.dylib")

def footprint():
    buf = (ctypes.c_uint64 * 64)()
    _proc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf))  # RUSAGE_INFO_V2: uuid(2 u64), user, sys, idle, intr, pageins, wired, resident, phys_footprint
    return buf[9] / 2**30

class Peak:  # ponytail: 20 ms polling, misses sub-20ms spikes
    def __init__(self):
        self.max, self._on, self.samples = footprint(), True, []
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while self._on:
            f = footprint(); self.samples.append((time.perf_counter(), f)); self.max = max(self.max, f); time.sleep(0.02)
    def window(self, a, b):
        return round(max([f for t, f in self.samples if a <= t <= b] or [0]), 2)
    def stop(self):
        self._on = False; return round(self.max, 2)

if __name__ == "__main__":
    import resource
    a = footprint(); x = bytearray(1 << 30); x[::4096] = b"x" * len(x[::4096]); b = footprint()
    assert 0.9 < b - a < 1.2, (a, b); print("ok", round(a, 2), round(b, 2))
```

### `bench_diffusers.py`

```python
"""diffusers-on-MPS benchmark: load, 1 warm-up, N timed runs. Prints one JSON line (prefix RESULT)."""
import argparse, glob, json, os, time
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.diffusionbee/hf"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch, diffusers
import fp

p = argparse.ArgumentParser()
p.add_argument("repo")
p.add_argument("--steps", type=int, required=True)
p.add_argument("--cfg", type=float, default=0.0)
p.add_argument("--size", type=int, default=1024)
p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
p.add_argument("--slicing", action="store_true")
p.add_argument("--vae-tiling", action="store_true")
p.add_argument("--empty-cache", action="store_true", help="torch.mps.empty_cache() after every step")
p.add_argument("--compile", action="store_true", help="torch.compile the transformer/unet")
p.add_argument("--quant", default=None, help="quanto-int8 | quanto-int4 | torchao-int8wo | gguf:<file> ; applied to transformer (+te with --quant-te)")
p.add_argument("--quant-te", action="store_true")
p.add_argument("--drop-te", action="store_true", help="encode the prompt once, then free the text encoder before denoising")
p.add_argument("--pre-encode", action="store_true", help="encode the prompt once before the timed runs, keep the text encoder loaded")
p.add_argument("--runs", type=int, default=1)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--prompt", default="A red fox sitting in a snowy forest at golden hour, detailed fur, photorealistic")
p.add_argument("--out", required=True)
a = p.parse_args()

dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
# download_model pins revision=sha, so there is no refs/main: take the snapshot dir directly
path = a.repo if os.path.isdir(a.repo) else glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "models--" + a.repo.replace("/", "--"), "snapshots", "*"))[0]
kw = dict(torch_dtype=dtype, use_safetensors=True)
if glob.glob(os.path.join(path, "*", "*.fp16*.safetensors")):
    kw["variant"] = "fp16"

t0 = time.perf_counter()
if a.quant and a.quant.startswith("gguf:"):
    idx = json.load(open(os.path.join(path, "model_index.json")))
    cls = getattr(diffusers, idx["transformer"][1])
    kw["transformer"] = cls.from_single_file(
        a.quant[5:], quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=dtype),
        config=path, subfolder="transformer", torch_dtype=dtype)
pipe = diffusers.DiffusionPipeline.from_pretrained(path, **kw)
den = getattr(pipe, "transformer", None) or pipe.unet
if a.quant and a.quant.startswith("quanto"):
    from optimum.quanto import quantize, freeze, qint8, qint4
    w = qint8 if a.quant.endswith("int8") else qint4
    for m in [den] + ([pipe.text_encoder] if a.quant_te else []):
        quantize(m, weights=w); freeze(m)
elif a.quant and a.quant.startswith("gguf:") and a.quant_te:
    from optimum.quanto import quantize, freeze, qint8
    quantize(pipe.text_encoder, weights=qint8); freeze(pipe.text_encoder)
elif a.quant == "torchao-int8wo":
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    for m in [den] + ([pipe.text_encoder] if a.quant_te else []):
        quantize_(m, Int8WeightOnlyConfig())
pipe.to("mps")
pipe.set_progress_bar_config(disable=True)
if a.slicing:
    pipe.enable_attention_slicing()
if a.vae_tiling:
    pipe.vae.enable_tiling()
if a.compile:
    if hasattr(pipe, "transformer"):
        pipe.transformer.compile()
    else:
        pipe.unet.compile()
torch.mps.synchronize()
t_load = time.perf_counter() - t0
pe = None
if a.drop_te or a.pre_encode:  # ponytail: one fixed prompt; a real backend would re-load the TE per new prompt
    import inspect
    ekw = {"do_classifier_free_guidance": False} if "do_classifier_free_guidance" in inspect.signature(pipe.encode_prompt).parameters else {}
    with torch.no_grad():
        pe = pipe.encode_prompt(prompt=a.prompt, device="mps", **ekw)[0]
    if a.drop_te:
        pipe.text_encoder = None
import gc; gc.collect(); torch.mps.empty_cache()
fp_load = round(fp.footprint(), 2)

peak = [0]
stamps = []
def cb(pipe_, i, t, kwargs):
    torch.mps.synchronize()
    stamps.append(time.perf_counter())
    peak[0] = max(peak[0], torch.mps.driver_allocated_memory())
    if a.empty_cache:
        torch.mps.empty_cache()
    return {}

def run():
    stamps.clear()
    g = torch.Generator("cpu").manual_seed(a.seed)
    call = dict(**({"prompt_embeds": pe} if pe is not None else {"prompt": a.prompt}), num_inference_steps=a.steps, guidance_scale=a.cfg, width=a.size, height=a.size,
                generator=g, callback_on_step_end=cb)
    t = time.perf_counter()
    img = pipe(**call).images[0]
    torch.mps.synchronize()
    return img, t, time.perf_counter()

t_w0 = time.perf_counter(); run(); t_warm = time.perf_counter() - t_w0
res = []
pk = fp.Peak(); phase = []
for _ in range(a.runs):
    img, t_start, t_end = run()
    steps = [b - c for b, c in zip(stamps[1:], stamps[:-1])]
    phase.append((pk.window(t_start, stamps[-1]), pk.window(stamps[-1], t_end)))
    res.append(dict(total=t_end - t_start, first_step_at=stamps[0] - t_start,
                    s_per_step=sum(steps) / len(steps) if steps else stamps[0] - t_start,
                    decode=t_end - stamps[-1], n_steps=len(stamps)))
fp_gen = pk.stop()
img.save(a.out)
best = min(res, key=lambda r: r["total"])
print("RESULT " + json.dumps(dict(
    repo=a.repo, steps=a.steps, empty_cache=a.empty_cache, dtype=a.dtype, slicing=a.slicing, vae_tiling=a.vae_tiling, compile=a.compile,
    quant=a.quant, quant_te=a.quant_te, drop_te=a.drop_te, pre_encode=a.pre_encode, load_s=round(t_load, 2), warmup_s=round(t_warm, 2),
    total_s=round(best["total"], 2), s_per_step=round(best["s_per_step"], 3),
    prestep_s=round(best["first_step_at"] - best["s_per_step"], 2), decode_s=round(best["decode"], 2),
    n_steps=best["n_steps"], fp_load_gb=fp_load, fp_gen_peak_gb=fp_gen, fp_denoise_peak_gb=max(p[0] for p in phase), fp_decode_peak_gb=max(p[1] for p in phase), mps_peak_driver_gb=round(peak[0] / 2**30, 2),
    runs=[{k: round(v, 2) for k, v in r.items()} for r in res], all_totals=[round(r["total"], 2) for r in res], out=a.out)))
```

### `bench_mflux.py`

```python
"""mflux (MLX) benchmark: load, 1 warm-up, N timed runs. Prints one JSON line (prefix RESULT)."""
import argparse, json, os, time
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.diffusionbee/hf"))
import mlx.core as mx
import fp

p = argparse.ArgumentParser()
p.add_argument("family", choices=["zimage", "flux2"])
p.add_argument("--steps", type=int, required=True)
p.add_argument("--cfg", type=float, default=None)
p.add_argument("--size", type=int, default=1024)
p.add_argument("-q", "--quantize", type=int, default=None)
p.add_argument("--model-path", default=None, help="local snapshot dir (default: resolve repo from HF cache)")
p.add_argument("--runs", type=int, default=1)
p.add_argument("--cache-limit-gb", type=float, default=2.0, help="MLX buffer-cache cap; mflux leaves it unbounded (~20GB extra after one 1024px image)")
p.add_argument("--vae-tiling", action="store_true")
p.add_argument("--save", default=None, help="write the loaded (quantized) model in mflux format and exit")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--prompt", default="A red fox sitting in a snowy forest at golden hour, detailed fur, photorealistic")
p.add_argument("--out", required=True)
a = p.parse_args()
if a.cache_limit_gb >= 0:
    mx.set_cache_limit(int(a.cache_limit_gb * 2**30))

t0 = time.perf_counter()
if a.family == "zimage":
    from mflux.models.z_image import ZImageTurbo
    model = ZImageTurbo(quantize=a.quantize, model_path=a.model_path)
    gen_kw = {}
else:
    from mflux.models.flux2 import Flux2Klein
    from mflux.models.common.config.model_config import ModelConfig
    model = Flux2Klein(quantize=a.quantize, model_path=a.model_path, model_config=ModelConfig.flux2_klein_4b())
    gen_kw = {"guidance": a.cfg if a.cfg is not None else 1.0}
mx.eval(model.parameters())
import gc; gc.collect(); mx.clear_cache()  # drop the raw HF tensors the loader converted from
t_load = time.perf_counter() - t0
fp_load = round(fp.footprint(), 2)
load_peak = mx.get_peak_memory()
if a.save:
    t = time.perf_counter(); model.save_model(a.save)
    print("RESULT " + json.dumps(dict(family=a.family, quantize=a.quantize, load_s=round(t_load, 2), save_s=round(time.perf_counter() - t, 2),
          mlx_peak_load_gb=round(load_peak / 2**30, 2), mlx_active_gb=round(mx.get_active_memory() / 2**30, 2), fp_load_gb=fp_load, saved=a.save)))
    raise SystemExit

if a.vae_tiling:
    from mflux.models.common.vae.tiling_config import TilingConfig
    model.tiling_config = TilingConfig()

stamps = []
class Timer:  # mflux in-loop callback: one call per denoising step
    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        mx.eval(latents)
        stamps.append(time.perf_counter())
model.callbacks.register(Timer())

def run():
    stamps.clear()
    t = time.perf_counter()
    img = model.generate_image(seed=a.seed, prompt=a.prompt, num_inference_steps=a.steps,
                               width=a.size, height=a.size, **gen_kw)
    return img, t, time.perf_counter()

t_w0 = time.perf_counter(); run(); t_warm = time.perf_counter() - t_w0
mx.reset_peak_memory()
res = []
pk = fp.Peak()
for _ in range(a.runs):
    img, t_start, t_end = run()
    steps = [b - c for b, c in zip(stamps[1:], stamps[:-1])]
    res.append(dict(total=t_end - t_start, first_step_at=stamps[0] - t_start,
                    s_per_step=sum(steps) / len(steps) if steps else stamps[0] - t_start,
                    decode=t_end - stamps[-1], n_steps=len(stamps)))
fp_gen = pk.stop()
img.image.save(a.out) if hasattr(img, "image") else img.save(a.out)
best = min(res, key=lambda r: r["total"])
print("RESULT " + json.dumps(dict(
    family=a.family, steps=a.steps, quantize=a.quantize, vae_tiling=a.vae_tiling, load_s=round(t_load, 2), warmup_s=round(t_warm, 2),
    total_s=round(best["total"], 2), s_per_step=round(best["s_per_step"], 3),
    prestep_s=round(best["first_step_at"] - best["s_per_step"], 2), decode_s=round(best["decode"], 2),
    n_steps=best["n_steps"], fp_load_gb=fp_load, fp_gen_peak_gb=fp_gen, mlx_peak_gen_gb=round(mx.get_peak_memory() / 2**30, 2),
    mlx_peak_load_gb=round(load_peak / 2**30, 2), mlx_active_gb=round(mx.get_active_memory() / 2**30, 2), cache_limit_gb=a.cache_limit_gb,
    runs=[{k: round(v, 2) for k, v in r.items()} for r in res],
    all_totals=[round(r["total"], 2) for r in res], out=a.out)))
```

### `bench_mlx_sdxl.py`

```python
"""SDXL base on MLX via ml-explore/mlx-examples stable_diffusion (scratch copy in mlx_sd/, fp16 UNet/TE, fp32 VAE).
Research-only: Euler-ancestral sampler, reads the fp16-variant files the backend downloads."""
import argparse, json, os, time
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.diffusionbee/hf"))
os.environ["HF_HUB_OFFLINE"] = "1"
import mlx.core as mx
import mlx.nn as nn
import fp
import numpy as np
from PIL import Image
import mlx_sd
mx.set_cache_limit(2 << 30)  # same cap as the mflux runs
from mlx_sd import model_io

REPO = "stabilityai/stable-diffusion-xl-base-1.0"
import glob
SNAP = glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "models--" + REPO.replace("/", "--"), "snapshots", "*"))[0]
model_io.hf_hub_download = lambda repo, f: os.path.join(SNAP, f)  # download_model pins the sha: no refs/main to resolve
model_io._MODELS[REPO] = dict(model_io._MODELS["stabilityai/sdxl-turbo"],
    unet="unet/diffusion_pytorch_model.fp16.safetensors", text_encoder="text_encoder/model.fp16.safetensors",
    text_encoder_2="text_encoder_2/model.fp16.safetensors", vae="vae/diffusion_pytorch_model.fp16.safetensors")

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=30)
p.add_argument("--cfg", type=float, default=6.0)
p.add_argument("-q", "--quantize", action="store_true", help="8-bit UNet + text encoders")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--prompt", default="A red fox sitting in a snowy forest at golden hour, detailed fur, photorealistic")
p.add_argument("--out", required=True)
a = p.parse_args()

t0 = time.perf_counter()
sd = mlx_sd.StableDiffusionXL(REPO, float16=True)
sd.autoencoder = model_io.load_autoencoder(REPO, False)  # SDXL VAE overflows in fp16
if a.quantize:
    for m in (sd.text_encoder_1, sd.text_encoder_2, sd.unet):
        nn.quantize(m, bits=8, class_predicate=lambda _, m: isinstance(m, nn.Linear))
sd.ensure_models_are_loaded()
t_load = time.perf_counter() - t0

def run():
    stamps = []
    t = time.perf_counter()
    for x_t in sd.generate_latents(a.prompt, n_images=1, num_steps=a.steps, cfg_weight=a.cfg,
                                   latent_size=(128, 128), seed=a.seed):
        mx.eval(x_t)
        stamps.append(time.perf_counter())
    x = sd.decode(x_t.astype(mx.float32))
    mx.eval(x)
    return x, t, stamps, time.perf_counter()

t_w = time.perf_counter(); run(); t_warm = time.perf_counter() - t_w
mx.reset_peak_memory()
pk = fp.Peak()
x, t_start, stamps, t_end = run()
fp_gen = pk.stop()
Image.fromarray((np.array(x[0]) * 255).astype(np.uint8)).save(a.out)
steps = [b - c for b, c in zip(stamps[1:], stamps[:-1])]
print("RESULT " + json.dumps(dict(engine="mlx-examples", repo=REPO, steps=a.steps, quantize=a.quantize,
    load_s=round(t_load, 2), warmup_s=round(t_warm, 2), total_s=round(t_end - t_start, 2),
    s_per_step=round(sum(steps) / len(steps), 3), decode_s=round(t_end - stamps[-1], 2), n_steps=len(stamps), fp_gen_peak_gb=fp_gen,
    mlx_peak_gen_gb=round(mx.get_peak_memory() / 2**30, 2), out=a.out)))
```

### `quality.py`

```python
"""PSNR (dB) of each image vs a reference; same seed + same engine => measures quantization drift. usage: quality.py ref.png img...  """
import sys, numpy as np
from PIL import Image
def load(p): return np.asarray(Image.open(p).convert("RGB"), np.float64)
def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
ref = load(sys.argv[1])
for p in sys.argv[2:]:
    x = load(p); print(f"{p}: psnr {psnr(ref, x):.1f} dB  mean {x.mean():.1f} std {x.std():.1f}")
```

### `sdpa_probe2.py`

```python
"""Peak process footprint for one Z-Image-sized SDPA call on MPS: no mask vs boolean padding mask."""
import json, sys, torch, torch.nn.functional as F
import fp
q = torch.randn(1, 30, 4128, 128, device="mps", dtype=torch.bfloat16); k, v = q.clone(), q.clone()
mask = torch.ones(1, 1, 1, 4128, dtype=torch.bool, device="mps"); mask[..., -40:] = False
torch.mps.synchronize()
res = {}
for name, kw in [("nomask", {}), ("boolmask", {"attn_mask": mask})]:
    torch.mps.empty_cache(); b = fp.footprint(); pk = fp.Peak()
    for _ in range(3):
        o = F.scaled_dot_product_attention(q, k, v, **kw); torch.mps.synchronize(); del o
    res[name + "_extra_gb"] = round(pk.stop() - b, 2)
print("RESULT " + json.dumps(res))
```

### `diag_load.py`

```python
"""Where does the mflux load peak come from? usage: diag_load.py zimage|flux2 none|snap [torch]"""
import glob, json, os, sys, time
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.diffusionbee/hf"))
if "torch" in sys.argv: import torch
import mlx.core as mx
import fp
mx.set_cache_limit(2 << 30)
fam, how = sys.argv[1], sys.argv[2]
repo = {"flux2": "black-forest-labs--FLUX.2-klein-4B", "zimage": "Tongyi-MAI--Z-Image-Turbo"}[fam]
src = glob.glob(os.path.expanduser(f"~/.diffusionbee/hf/hub/models--{repo}/snapshots/*"))[0] if how == "snap" else None
pk = fp.Peak(); r = {"fam": fam, "how": how, "torch": "torch" in sys.argv}
if fam == "zimage":
    from mflux.models.z_image import ZImageTurbo
    m = ZImageTurbo(quantize=4, model_path=src)
else:
    from mflux.models.flux2 import Flux2Klein
    from mflux.models.common.config.model_config import ModelConfig
    m = Flux2Klein(quantize=4, model_path=src, model_config=ModelConfig.flux2_klein_4b())
r["fp_construct"] = round(fp.footprint(), 2); r["peak_construct"] = round(pk.max, 2)
mx.eval(m.parameters()); r["fp_eval"] = round(fp.footprint(), 2); r["peak_eval"] = round(pk.max, 2)
from mflux.models.common.vae.tiling_config import TilingConfig
m.tiling_config = TilingConfig()
m.generate_image(seed=1, prompt="a fox", num_inference_steps=1, width=512, height=512)
r["peak_gen"] = pk.stop(); r["mlx_peak"] = round(mx.get_peak_memory() / 2**30, 2)
print("RESULT " + json.dumps(r))
```

### `mflux_pipe.py`

```python
"""Integration sketch: an mflux model behind the diffusers call surface that diffusionbee_backend.Engine.render() uses.
render() passes prompt/num_inference_steps/guidance_scale/width/height/generator/callback_on_step_end (+ image/strength
for img2img), reads .images[0], and stops a run by setting pipe._interrupt from the callback."""
import os, tempfile
from types import SimpleNamespace

import mlx.core as mx
from mflux.utils.exceptions import StopImageGenerationException

mx.set_cache_limit(2 << 30)  # MLX keeps freed buffers by default: +20 GB after one 1024² image, then swap


def load(family, src, bits=4):
    """src = the same diffusers snapshot dir load_base() gets; mflux converts + quantizes it at load (~3-9 s)."""
    from mflux.models.common.vae.tiling_config import TilingConfig
    if family == "zimage":
        from mflux.models.z_image import ZImageTurbo
        model = ZImageTurbo(quantize=bits, model_path=src)
    elif family == "flux2":
        from mflux.models.flux2 import Flux2Klein
        from mflux.models.common.config.model_config import ModelConfig
        model = Flux2Klein(quantize=bits, model_path=src, model_config=ModelConfig.flux2_klein_4b())
    else:
        raise ValueError(f"no MLX engine for {family}")
    mx.eval(model.parameters())  # quantize now, one layer at a time; left lazy, the first image holds bf16 + q4 (30 GB peak)
    mx.clear_cache()
    model.tiling_config = TilingConfig()  # VAE tiling: -4.5 GB peak at 1024², same speed
    return MfluxPipe(model)


class MfluxPipe:
    def __init__(self, model):
        self.model, self._interrupt, self._num_timesteps, self._cb = model, False, None, None
        model.callbacks.register(self)

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):  # mflux in-loop callback, once per step
        mx.eval(latents)  # MLX is lazy: without this, progress/stop run one step ahead of the GPU
        if self._cb:
            self._cb(self, t, None, {})
        if self._interrupt:
            raise KeyboardInterrupt  # mflux turns this into StopImageGenerationException

    def set_progress_bar_config(self, **kw):
        pass

    def __call__(self, prompt, num_inference_steps, guidance_scale, width, height, generator,
                 callback_on_step_end=None, image=None, strength=None, output_type="pil"):
        self._interrupt, self._num_timesteps, self._cb = False, num_inference_steps, callback_on_step_end
        kw = dict(seed=generator.initial_seed(), prompt=prompt, num_inference_steps=num_inference_steps,
                  width=width, height=height, guidance=guidance_scale)
        with tempfile.TemporaryDirectory() as d:
            if image is not None:  # mflux img2img takes a file and the fraction of steps to skip (= 1 - diffusers strength)
                kw.update(image_path=os.path.join(d, "init.png"), image_strength=1 - strength)
                image.save(kw["image_path"])
            try:
                r = self.model.generate_image(**kw)
            except StopImageGenerationException:
                return SimpleNamespace(images=[None])  # render() sees stopped=True and returns None
            finally:
                mx.clear_cache()
        return SimpleNamespace(images=[getattr(r, "image", r)])


if __name__ == "__main__":  # self-check: FLUX.2 q4 straight from the diffusers snapshot, called the way render() calls it
    import glob, sys, torch
    fam = sys.argv[1] if len(sys.argv) > 1 else "flux2"
    repo = {"flux2": "black-forest-labs--FLUX.2-klein-4B", "zimage": "Tongyi-MAI--Z-Image-Turbo"}[fam]
    snap = glob.glob(os.path.expanduser(f"~/.diffusionbee/hf/hub/models--{repo}/snapshots/*"))[0]
    pipe = load(fam, snap)
    seen = []
    def on_step(p, i, t, kw):
        seen.append(i)
        return {}
    call = dict(prompt="a red fox", num_inference_steps=4, guidance_scale=1.0 if fam == "flux2" else 0.0, width=512, height=512)
    img = pipe(**call, generator=torch.Generator("cpu").manual_seed(7), callback_on_step_end=on_step).images[0]
    assert img.size == (512, 512) and seen == [0, 1, 2, 3], (img.size, seen)
    img.save(f"out/adapter_{fam}_txt2img.png")

    def stop_at_1(p, i, t, kw):
        if i == 1:
            p._interrupt = True
        return {}
    assert pipe(**call, generator=torch.Generator("cpu").manual_seed(7), callback_on_step_end=stop_at_1).images[0] is None

    seen.clear()  # img2img, diffusers strength 0.5 of 4 steps -> the last 2 steps run
    img2 = pipe(**call, generator=torch.Generator("cpu").manual_seed(8), callback_on_step_end=on_step, image=img, strength=0.5).images[0]
    assert img2.size == (512, 512) and seen == [2, 3], seen
    img2.save(f"out/adapter_{fam}_img2img.png")
    print('RESULT {"adapter_selfcheck": "ok", "family": "%s"}' % fam)
```
