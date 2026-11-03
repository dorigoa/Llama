# Qwen3.8-27B Q8_0 on a PC (RTX 4060 Ti + RTX 5060 Ti, 16 GB each) — llama.cpp

Date: 2026-11-03 — llama.cpp build b12069

## Final configuration (~35 tok/s generation, 80K context)

```bash
/opt/llama.cpp/llama-server -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 66 -ts 1,1 -sm tensor -c 81920 -ub 256 -fa on --jinja \
  -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 -np 1 \
  --no-mmproj-offload
```

Steady-state VRAM: 15,790 / 16,380 MiB (4060 Ti) and 15,770 / 16,311 MiB (5060 Ti) → ~550–590 MiB free per card.

Latest measurement (~1000 generated tokens): 28.1 ms/token = **35.6 tok/s**; drafts accepted 615/1152 (53%), ~2.6 tokens per cycle.

Variant with more VRAM headroom: `-c 65536` without `-ub 256` (~800 MiB free per card, 33 tok/s).

## Measured progression (generation, short prompt)

| Configuration | tok/s |
|---|---|
| `-sm layer`, all on GPU, 32K, f16 KV, no MTP | 11.6 |
| + MTP (`--spec-type draft-mtp`), q8_0 KV | 23.0 |
| same with `-ngl 64` (block 0 + output head on CPU) | 16.5 |
| same with `-ngl 65` (only block 0 on CPU) | 19 |
| `-sm tensor`, `-ngl 65`, 48K | 26 |
| `-sm tensor`, all on GPU, 64K, mmproj on CPU | 33 |
| `-sm tensor`, all on GPU, 80K, `-ub 256` | **35.6** |

Note: the 33 → 35.6 jump is most likely due to a different prompt (53% vs 48% acceptance), not to `-ub 256`, which does not affect generation.

## Lessons learned

- The unsloth Q8_0 GGUF already contains the MTP head (`qwen35.nextn_predict_layers = 1`, tensors `blk.64.nextn.*`): 65 blocks + output = 66 "layers" for `-ngl`.
- `-ngl N` keeps the *first* layers on the CPU; with N < n_layer+1 the output head (~1.3 GB, also read by the MTP drafts) ends up on the CPU too → heavy penalty.
- MTP: acceptance ~48–53%, mean length ~2.4–2.6 with 3 drafted tokens per cycle; drafting costs ~13% of the time. The MTP context allocates its own compute buffer.
- `-hf` also loads the mmproj (vision projector) automatically: with the whole model on GPU it crashed in `clip_model_loader::load_tensors` due to lack of VRAM → use `--no-mmproj-offload` (vision on CPU) or `--no-mmproj`.
- At 80K with the default `-ub` (512): OOM crash (148 MiB on device 0) while allocating the MTP context's compute buffer (`common_speculative_init` → `sched_reserve`). `-ub 256` halves the compute buffers (including the logits buffer, 248K vocabulary) and fixes it; cost: somewhat slower prefill on long prompts.
- `-sm tensor` (experimental) with quantized KV cache requires a recent build; it splits weights and KV across the GPUs → near-perfect VRAM balance.
- With `-sm layer`, the 4060 Ti (288 GB/s) + 5060 Ti (448 GB/s) give a theoretical limit of ~12 tok/s without speculative decoding; measured 11.6.

## To do

- Measure prefill with long prompts (tens of thousands of tokens), comparing `-ub 256` vs 512.
- Drop `-lv 4` / `--log-file` for normal use.
