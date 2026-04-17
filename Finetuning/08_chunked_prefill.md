# Chunked prefill in vLLM

## 1. The short idea

Chunked prefill splits a large prompt prefill into smaller chunks so those chunks can be scheduled more flexibly, often alongside decode work.

This matters because long prefills can otherwise block ongoing decodes and hurt latency.

## Concept snapshot

| Lens | Answer |
| --- | --- |
| What | A scheduling approach that breaks large prompt prefills into smaller chunks instead of forcing the entire prefill to run as one slab. |
| Why | Long prefills are compute-heavy and can temporarily crowd out short decode steps that are more latency-sensitive. |
| How | If a prefill would exceed the current token budget, vLLM slices it into chunks and interleaves those chunks with decode work. |
| Main knobs | `enable_chunked_prefill`, `max_num_batched_tokens`, and the scheduler knobs for partial or long prefills. |
| Common confusion | It is not just "prompt splitting"; it is a scheduler policy for balancing prefill and decode in the same engine loop. |
| What it cannot fix alone | A token budget that is wildly mis-sized, insufficient KV capacity, or a workload with almost no long prompts. |

## Where it sits in the serving path

```text
Long prompt arrives -> scheduler checks token budget -> prompt is chunked if needed -> decode and prefill chunks interleave -> KV cache grows gradually
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       chunked prefill changes this admission step
```

## 2. Why prefill becomes a problem

Remember the two main phases:

- **prefill**: process the prompt and build the initial KV cache
- **decode**: generate new tokens step by step

Long prefills are usually compute-heavy.
Decodes are usually smaller but latency-sensitive.
If a giant prefill monopolizes the iteration budget, active decodes can stall.

## 3. Visual explanation

### Without chunked prefill

```text
Queue:
[decode][decode][decode][LONG PREFILL 16k TOKENS]

Scheduler runs long prefill in one big slab
------------------------------------------>
During that time, decode requests wait longer than they should.
```

### With chunked prefill

```text
LONG PREFILL 16k TOKENS
-> split into 4k + 4k + 4k + 4k

Iterations might look like:
[decode][decode][prefill chunk 1]
[decode][decode][prefill chunk 2]
[decode][prefill chunk 3]
[decode][decode][prefill chunk 4]
```

Now decode work gets regular chances to run.

### Mermaid view: split long prefills, then interleave them with decode work

```mermaid
flowchart LR
    L["Long prompt arrives"] --> C["Split into prefill chunks"]
    D["Active decode requests"] --> S{"Scheduler token budget"}
    C --> S
    S -- "serve decode first" --> R1["Run decode step"]
    S -- "use remaining budget" --> R2["Run one prefill chunk"]
    R1 --> S
    R2 --> S
    R2 --> K["KV cache grows chunk by chunk"]
    S --> B["Better TTFT-ITL balance on mixed workloads"]
```

This is why chunked prefill is best understood as a scheduler tool, not just a prompt-splitting trick. The split matters only because it gives the scheduler more chances to keep decode latency under control.

## 4. What vLLM does

Current vLLM docs describe chunked prefill this way:

- large prefills are processed in smaller chunks
- those chunks can be batched with decode requests
- this helps balance compute-bound prefill and memory-bound decode

In vLLM V1, the docs say chunked prefill is enabled by default whenever possible.

## 5. Scheduling intuition

With chunked prefill enabled, the scheduler prioritizes decode requests first.
Then, if there is remaining token budget in `max_num_batched_tokens`, it schedules prefills.
If a prefill does not fit, it is chunked automatically.

That means chunked prefill is not just “split prompts.”
It is really a **scheduling policy plus prompt-splitting behavior**.

## 6. Why it improves both latency and utilization

This seems counterintuitive at first.
How can doing extra splitting help both?

Because it lets vLLM combine two different kinds of work more effectively:

```text
Prefill = more compute-heavy
Decode  = more memory-sensitive and latency-sensitive
```

Interleaving them can keep the GPU busier while also preventing long-prefill domination.

## 7. The key tuning knob: `max_num_batched_tokens`

This is the most important mental model:

```text
Smaller max_num_batched_tokens  -> better decode latency / ITL
Larger max_num_batched_tokens   -> better TTFT and often better throughput
```

So chunked prefill is not just on/off.
Its behavior is shaped heavily by the iteration token budget.

## Token-budget intuition

| `max_num_batched_tokens` posture | What you usually get | Best fit |
| --- | --- | --- |
| Smaller / latency-biased | Decode steps get faster turns, but long prefills make slower forward progress | Interactive serving where ITL matters most |
| Balanced | Reasonable decode latency with steady prefill progress | Mixed workloads with both short chats and longer documents |
| Larger / throughput-biased | Long prefills finish sooner, but decodes can wait longer behind heavier iterations | Batchier jobs or workloads where TTFT/throughput matter more than ITL |

## 8. Minimal vLLM examples

### CLI

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct   --enable-chunked-prefill   --max-num-batched-tokens 4096   --max-num-seqs 32
```

### Python

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_chunked_prefill=True,
    max_num_batched_tokens=4096,
    max_model_len=16384,
)

sampling_params = SamplingParams(max_tokens=128, temperature=0)
outputs = llm.generate([
    "Summarize the following long report...",
], sampling_params)
```

### Example from a batch pipeline configuration

```python
engine_kwargs = {
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 4096,
    "max_model_len": 16384,
}
```

## 9. Worked example

Suppose one user sends a 20k-token context window prompt while ten other users are already decoding short answers.

Without chunking, the long prompt may occupy a large scheduling window and push the short decodes back.
With chunking:

- the long prompt enters in chunks
- the short decodes keep getting service
- the system avoids turning one giant prefill into a temporary traffic jam

## 10. Chunked prefill and long prompts

Long prompts are exactly where chunked prefill matters most.
If most prompts are short, you may notice less difference.
If your workload is mixed, especially with RAG or long documents, chunked prefill becomes much more important.

## 11. Related scheduler knobs

These are worth knowing:

- `--max-num-batched-tokens`
- `--max-num-partial-prefills`
- `--max-long-partial-prefills`
- `--long-prefill-token-threshold`
- `--scheduler-reserve-full-isl`

A good beginner approach is to tune `--max-num-batched-tokens` first and only touch the other ones after you understand the workload.

## 12. Multimodal nuance

Current docs also mention a control for chunked multimodal input scheduling.
The reason is simple: some multimodal items should not be half-admitted in awkward ways.
That is a more advanced case, but it is worth remembering if you serve image/text models.

## 13. Practical tuning checklist

```text
1. Turn chunked prefill on if long prompts hurt decode latency.
2. Start around 2048–4096 max_num_batched_tokens for latency-sensitive serving.
3. Increase the token budget if TTFT is too slow or throughput is too low.
4. Benchmark with a realistic long-prompt workload, not only short prompts.
5. Watch both TTFT and ITL; chunked prefill is all about their balance.
```

## 14. Common mistakes

### Mistake: evaluating chunked prefill on only short prompts
You may conclude it does nothing because the workload never needed it.

### Mistake: chasing throughput only
If you raise the token budget too far, decodes may slow down even while total throughput improves.

### Mistake: treating prefill and decode as the same kind of work
They stress the system differently. Chunked prefill exists because their scheduling needs are different.

## 15. One-line summary

Chunked prefill prevents long prompts from monopolizing the scheduler by splitting prefills into smaller pieces that can be interleaved with decode work, improving the TTFT–ITL balance on mixed workloads.

## 16. Visual references

- vLLM Optimization and Tuning: https://docs.vllm.ai/en/stable/configuration/optimization/
- vLLM Batch LLM Inference example: https://docs.vllm.ai/en/stable/examples/offline_inference/batch_llm_inference/
- vLLM Engine Arguments for scheduler knobs: https://docs.vllm.ai/en/stable/configuration/engine_args/

## Source basis
This notebook was written from the official vLLM docs and blog/paper set, including the vLLM docs homepage, Optimization and Tuning, Quantization, Quantized KV Cache, Paged Attention, CUDA Graphs, Attention Backend Feature Support, Batch LLM Inference example, and the original vLLM blog post and paper.
