# vLLM inference optimization notebook pack

This pack turns the major vLLM inference optimizations into separate notebook-style Markdown guides.
Each topic is self-contained, uses diagrams you can read in plain text, and ends with practical tuning advice.
To make the pack easier to teach from and skim quickly, each topic now follows the same learning pattern:
what the optimization is, why it exists, how it works, which knobs matter first, and what confusion to avoid.
The newest notebook is an adjacent systems guide on how to evaluate agentic multi-agent systems end to end.

## What is included

- `01_continuous_batching.md` — how iteration-level scheduling keeps GPUs full.
- `02_quantization.md` — how lower precision reduces memory and bandwidth pressure.
- `03_kv_cache.md` — what the KV cache is, why it matters, and how to size it.
- `04_pagedattention.md` — the memory paging idea behind vLLM’s KV management.
- `05_cuda_hip_graphs.md` — graph capture and replay to reduce launch overhead.
- `06_optimized_cuda_kernels.md` — attention/GEMM/MoE backend selection and why kernels matter.
- `07_speculative_decoding.md` — draft-and-verify generation to reduce expensive target-model steps.
- `08_chunked_prefill.md` — splitting long prompts so prefills do not block decodes.
- `09_agentic_multi_agent_system_evaluation.md` — end-to-end evaluation frameworks, benchmarks, protocols, and practical scoring ideas for agentic multi-agent systems.
- `99_all_topics_combined.md` — all topics in one long notebook-style file.

## At a glance

| Topic | Main question it answers | Main bottleneck it targets | First knobs or signals to watch |
| --- | --- | --- | --- |
| Continuous batching | How do we keep mixed-size requests moving? | Idle slots and head-of-line blocking | `max_num_seqs`, `max_num_batched_tokens` |
| Quantization | How do we shrink model cost per layer? | Weight memory and bandwidth pressure | `quantization`, `dtype`, hardware/kernel support |
| KV cache | Why does memory dominate concurrency? | Live token state growth over time | `gpu_memory_utilization`, `kv_cache_*` |
| PagedAttention | How does vLLM avoid brittle KV allocation? | Fragmentation and waste in KV memory | Usually automatic; tune adjacent KV settings |
| CUDA/HIP graphs | How do repeated decode steps get cheaper? | CPU launch overhead on stable shapes | `compilation_config`, backend compatibility |
| Optimized CUDA kernels | Why can the same model run very differently? | Backend fit to hardware and layout | attention/backend selection, dtype compatibility |
| Speculative decoding | How do we advance multiple tokens per target step? | Expensive one-token-at-a-time decode | `speculative_config`, acceptance behavior |
| Chunked prefill | How do long prompts stop blocking short decodes? | Long prefill bursts dominating token budget | `max_num_batched_tokens`, scheduler behavior |
| Agentic multi-agent evaluation | How do we know an agent team really works end to end? | Hidden failures across planning, delegation, tool use, safety, and reliability | dataset slices, trace quality, task success, trajectory quality, latency/cost budgets |

## Suggested reading order

1. Continuous batching
2. KV cache
3. PagedAttention
4. Chunked prefill
5. Quantization
6. CUDA/HIP graphs
7. Optimized CUDA kernels
8. Speculative decoding
9. Agentic multi-agent system evaluation

## Pack concept map

```text
Incoming requests
    |
    v
[Continuous batching scheduler] <----- [Chunked prefill protects decode latency]
    |
    +---- decides how much work fits this iteration
    |
    v
[Prefill] ---------------------> [KV cache] <----- managed by ----- [PagedAttention]
    |                                |
    |                                +----- footprint can shrink via ----- [Quantization / FP8 KV]
    v
[Decode loop]
    +----- fewer expensive target steps via ----- [Speculative decoding]
    +----- lower CPU launch overhead via -------- [CUDA/HIP graphs]
    +----- faster math/backends via ------------- [Optimized CUDA kernels]
```

## How to read each notebook

- `What is it?` Start with the short idea and concept snapshot.
- `Why does it exist?` Look for the bottleneck or failure mode it is addressing.
- `How does it work?` Read the serving-path diagram and the worked example together.
- `What do I tune first?` Use the knobs/checklist sections before trying advanced flags.
- `What can it not fix?` Read the common-confusion row and the common-mistakes section before benchmarking.

## Quick start

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

That one command already gives you vLLM’s core scheduling and memory-management stack. The rest of this pack explains what those optimizations are actually doing under the hood.

## Notes on examples

- The CLI examples use current vLLM-style `vllm serve` commands.
- The Python examples use `from vllm import LLM, SamplingParams` where it helps.
- Some examples use model names taken from official docs pages because they are known to illustrate the feature.
- Real performance depends on your workload mix, prompt lengths, hardware generation, and concurrency.

## Source basis
This notebook was written from the official vLLM docs and blog/paper set, including the vLLM docs homepage, Optimization and Tuning, Quantization, Quantized KV Cache, Paged Attention, CUDA Graphs, Attention Backend Feature Support, Batch LLM Inference example, and the original vLLM blog post and paper.
