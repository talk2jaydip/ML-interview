## LLaMA-Factory Scaling & Configuration Guide

This guide helps you choose and configure training/finetuning topologies (single GPU, multi‑GPU, single node, multi node), understand memory/CPU trade‑offs, and select attention/KV/RoPE options. It includes decision tables, formulas, and copy‑paste commands.

### Scope
- Methods: LoRA, QLoRA, Freeze, Full fine‑tuning (and notes on OFT).
- Topologies: single GPU, multi‑GPU single node (DDP), multi‑node, Ray, DeepSpeed ZeRO, FSDP.
- Performance features: mixed precision, gradient checkpointing, attention backends (SDPA/FlashAttn‑2), KV cache, RoPE scaling, LongLoRA shift attention.

---

## Quick Decision Matrix

| Scenario | VRAM per GPU | Recommended Method | Topology | Notes |
| --- | --- | --- | --- | --- |
| SFT up to 7B on 1 GPU | 16–24 GB | LoRA | Single GPU | Fastest iteration; minimal memory. |
| SFT 7–13B on 1 GPU | 16–24 GB | QLoRA (4‑bit) | Single GPU | Use `quantization_bit: 4`, `bf16: true`. |
| SFT 13–34B on 1–2 GPUs | 24–48 GB | QLoRA | Single or 2x GPUs | Consider gradient accumulation and checkpointing. |
| Full FT ≤ 13B | 48–80 GB | Full | Single node DDP + ZeRO‑3 | Requires DeepSpeed ZeRO‑3 or FSDP. |
| Full FT ≥ 34B | 80–160+ GB | Full | Multi‑GPU + ZeRO‑3/FSDP | Prefer multi‑node; bf16 + checkpointing. |
| RL (DPO/KTO/RM/PPO) with adapters | 16–24 GB | LoRA/QLoRA | Single or Multi‑GPU | PPO needs reward/ref models; mind extra memory. |

Best defaults:
- Precision: `bf16` (if supported) else `fp16`.
- Attention: `flash_attn=fa2` if available, else `sdpa` (PyTorch ≥ 2.1.1), else `eager`.
- Checkpointing: enabled (default) for training; disable only if memory is plentiful.

---

## Finetuning Techniques: Pros/Cons & When to Use

### LoRA
- Concept: Train low‑rank adapters on top of frozen base weights.
- Pros: Very low VRAM/CPU; fast; easy to swap/merge; excellent for SFT/RLHF.
- Cons: Limited capacity vs full FT; target modules selection matters.
- Use when: Single‑GPU or limited VRAM; rapid iteration; deploying via adapter or merge.

Key args (FinetuningArguments): `finetuning_type: lora`, `lora_rank`, `lora_alpha`, `lora_target`, `lora_dropout`, `use_rslora`, `use_dora`.

### QLoRA
- Concept: Base model quantized (e.g., 4‑bit) + LoRA adapters trained in higher precision.
- Pros: Enables larger models on a single GPU; very memory‑efficient.
- Cons: Slight quality drop vs full precision; some features limited in 4‑bit.
- Use when: 4‑bit/8‑bit training needed to fit VRAM.

Key args (ModelArguments): `quantization_bit: 4|8`, `quantization_type: nf4|fp4`, `double_quantization: true`, `upcast_layernorm: true` (recommended).

### Freeze
- Concept: Freeze most layers; train top N layers/modules.
- Pros: More capacity than LoRA in some cases; still efficient.
- Cons: Less efficient than LoRA; careful layer selection needed.
- Use when: Need more capacity than LoRA but cannot afford full FT.

Key args: `finetuning_type: freeze`, `freeze_trainable_layers`, `freeze_trainable_modules`, `freeze_extra_modules`.

### Full Fine‑Tuning
- Concept: Update all parameters.
- Pros: Maximum capacity/performance.
- Cons: Very high VRAM/CPU/IO; slow; requires ZeRO‑3/FSDP for bigger models.
- Use when: You have substantial compute/memory and need peak quality.

Key args: `finetuning_type: full`; recommend `deepspeed: ds_z3_config.json` or FSDP.

### OFT (Orthogonal FT) (optional)
- Niche low‑rank alternative to LoRA (see `oft_*` args); similar trade‑offs.

---

## Memory & CPU Estimation Cheatsheet

Let:
- P = number of model parameters (e.g., 7B ≈ 7e9).
- b = bytes/param for weights (fp32=4, bf16/fp16=2, int8=1, int4≈0.5 effective).
- O = optimizer states multiplier (AdamW ≈ 2 for m,v; grads ≈ 1; params ≈ 1 → total ≈ 4× in full FT).
- A ≈ activation memory factor (depends on batch_size, seq_len, hidden_size, layers); with checkpointing, A reduces 30–60%.

Approx GPU memory (very rough):
- Inference: Weights ≈ P × b.
- LoRA train: Base weights (frozen) + LoRA params (~<1% of P) + activations.
- QLoRA train: Quantized base (P × 0.5 for int4) + LoRA params (fp16/bf16) + activations.
- Full FT: (Params + Grads + Opt states) × b + activations ≈ 4 × P × b + A.

Ballpark (bf16, activations included, with checkpointing; highly workload‑dependent):

| Model | LoRA (bf16) | QLoRA (4‑bit) | Full FT (ZeRO‑3/FSDP) |
| --- | ---: | ---: | ---: |
| 7B | 12–18 GB | 10–14 GB | 60–80+ GB (sharded) |
| 13B | 18–26 GB | 14–20 GB | 100–140+ GB (sharded) |
| 34B | 32–48 GB | 26–38 GB | 250–400+ GB (multi‑node) |

CPU considerations:
- DeepSpeed ZeRO‑Offload and FSDP CPU offload shift optimizer/params to CPU RAM; ensure fast NVMe/PCIe and sufficient host RAM (≥ weights × 2–4).
- Dataloading: set `preprocessing_num_workers` and `dataloader_num_workers` per disk/CPU.

Use `bf16: true` if supported to reduce activation memory with minimal quality loss.

---

## Attention, KV Cache, RoPE, Long Context

### Attention backend (ModelArguments.flash_attn)
- Values: `auto`, `disabled` (eager), `sdpa`, `fa2` (FlashAttention‑2).
- Recommendations:
  - If FlashAttention‑2 installed: `fa2` (best throughput/latency).
  - Else PyTorch ≥ 2.1.1: `sdpa`.
  - Else: `disabled` (vanilla eager).
- Gemma‑2 is auto‑forced to FA2 if available; SDPA warns about soft‑capping.

Example YAML:
```yaml
flash_attn: fa2   # choices: auto|disabled|sdpa|fa2
```

### KV cache (ModelArguments.use_cache)
- Training: disabled automatically to save memory.
- Inference: enable for faster generation.

```yaml
use_cache: true   # inference only
```

### RoPE scaling (ModelArguments.rope_scaling)
- Options (repo constants): dynamic, yarn, llama3.
- Training: scaling factor computed from `cutoff_len`; inference default factor ≈ 2.
- Caveats: dynamic NTK may underperform for FT; vLLM backend disallows RoPE scaling.

```yaml
rope_scaling: llama3   # or dynamic|yarn
cutoff_len: 8192       # enlarges max position embeddings accordingly
```

### LongLoRA / Shift Short Attention (S^2‑Attn)
- Enable with `shift_attn: true` (not compatible with PPO stage).

```yaml
shift_attn: true
```

---

## Topologies & Commands

### Single‑GPU (recommended for LoRA/QLoRA)

```bash
llamafactory-cli train \
  examples/train_lora/llama3_lora_sft.yaml
```

Key YAML (LoRA + bf16):
```yaml
stage: sft
do_train: true
finetuning_type: lora
bf16: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1e-4
num_train_epochs: 3
```

### Multi‑GPU, Single Node (DDP via torchrun)

```bash
FORCE_TORCHRUN=1 llamafactory-cli train \
  examples/train_lora/llama3_lora_sft.yaml
```

Tips:
- Increase `per_device_train_batch_size` when possible.
- For LoRA DDP: `ddp_find_unused_parameters` is auto‑set False when applicable.

### Multi‑Node DDP

```bash
# Node 0
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# Node 1
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

Elastic (fault‑tolerant) multi‑node:
```bash
FORCE_TORCHRUN=1 MIN_NNODES=1 MAX_NNODES=3 MAX_RESTARTS=3 RDZV_ID=llamafactory \
  MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  llamafactory-cli train examples/train_full/llama3_full_sft.yaml
```

### DeepSpeed ZeRO‑3 (weight sharding)

YAML:
```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
bf16: true
```

Launch:
```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft.yaml
```

Offload variant:
```yaml
deepspeed: examples/deepspeed/ds_z3_offload_config.json
```

### Ray distributed

```bash
USE_RAY=1 llamafactory-cli train examples/train_lora/llama3_lora_sft_ray.yaml
```

Key args (RayArguments): `ray_num_workers`, `resources_per_worker` (default {GPU:1}), `placement_strategy`.

### FSDP (via Accelerate)

Use template configs in `examples/accelerate/*.yaml` (e.g., `fsdp_config.yaml`).

---

## Configuration Snippets by Technique

### LoRA (SFT)
```yaml
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
bf16: true
output_dir: saves/llama3-8b/lora/sft
```

### QLoRA (4‑bit)
```yaml
stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
quantization_type: nf4
upcast_layernorm: true
bf16: true
```

### Freeze (train last N layers)
```yaml
stage: sft
do_train: true
finetuning_type: freeze
freeze_trainable_layers: 8
freeze_trainable_modules: all
bf16: true
```

### Full Fine‑Tuning + ZeRO‑3
```yaml
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json
bf16: true
```

---

## Best‑Practice Playbook

- Start with LoRA; switch to QLoRA if VRAM is tight.
- Use `bf16: true`; enable FlashAttention‑2 if available (`flash_attn: fa2`).
- Keep gradient checkpointing enabled for larger seq/batch lengths.
- For >13B full FT, plan ZeRO‑3/FSDP and consider multi‑node.
- For RL (PPO), allocate memory for ref/reward models (can be LoRA/quantized). PPO forbids some options (e.g., `shift_attn`).
- When adding tokens, set `resize_vocab: true` and add embeddings to `additional_target` for LoRA.

---

## Verification Checklist (before launch)

- Model/Tokenization
  - `model_name_or_path` reachable and `trust_remote_code` set appropriately.
  - Template matches model family (e.g., `llama3`, `qwen2_vl`).
- Data
  - `dataset_info.json` entries correct; `eval_dataset` or `val_size` set.
  - `cutoff_len` matches desired context; consider RoPE scaling.
- Compute
  - `bf16`/`fp16` set; FlashAttn/SDPA chosen; checkpointing enabled.
  - For ZeRO‑3/FSDP: configs in place; storage offload directories exist.
- Logging/Export
  - `output_dir` unique; `save_steps`/`logging_steps` configured.
  - For export/merge/quantize: use `llamafactory-cli export` with proper `export_*` args.

---

## References (Source Code Hooks)

- Attention backend: `src/llamafactory/model/model_utils/attention.py`.
- KV cache: `src/llamafactory/model/model_utils/kv_cache.py`.
- RoPE scaling: `src/llamafactory/model/model_utils/rope.py`.
- Training dispatcher: `src/llamafactory/train/tuner.py`.
- Arguments: `src/llamafactory/hparams/*.py` (model/data/training/finetuning/generating).


