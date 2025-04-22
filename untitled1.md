# 📘 DeepSpeed ZeRO: Comprehensive Preparation Guide

## 🧠 Overview

**DeepSpeed** is an open-source deep learning optimization library developed by Microsoft, designed to reduce computational demands and improve training efficiency. A standout feature is the **Zero Redundancy Optimizer (ZeRO)**, which optimizes memory usage and scales training to unprecedented model sizes.

As models grow, the memory required for parameters, gradients, and optimizer states can exceed GPU capacities. ZeRO addresses this by partitioning model states across devices, reducing memory footprint per device and enabling the training of larger models.

---

## ⚙️ ZeRO Optimization Stages

ZeRO operates in stages, each progressively partitioning more components of the model state:

### 🔹 Stage 1: Sharding Optimizer States

- Partitions optimizer states across GPUs instead of replicating them.
- Reduces memory redundancy without impacting gradient or parameter storage.
- Each GPU holds a shard of the optimizer states.

```json
{
  "zero_optimization": {
    "stage": 1
  },
  "gradient_accumulation_steps": 1,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true
  }
}
```

- You can offload optimizer states to CPU (works in stages 1–3):

```json
{
  "zero_optimization": {
    "stage": 1
  },
  "offload_optimizer": {
    "device": "cpu"
  },
  "gradient_accumulation_steps": 1,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true
  }
}
```

### 🔹 Stage 2: Partitioning Gradients

- Partitions gradients across GPUs instead of storing full gradients on each.
- Enables larger batch sizes and models.

```json
{
  "zero_optimization": {
    "stage": 2
  },
  "gradient_accumulation_steps": 1,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true
  }
}
```

### 🔹 Stage 3: Partitioning Model Parameters

- Partitions model parameters across GPUs.
- Minimizes GPU memory usage by sharding full model state: parameters, gradients, and optimizer states.
- Communication between GPUs ensures each device gets parameters it needs for forward/backward passes.

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "gradient_accumulation_steps": 1,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true
  }
}
```

Or simplified version:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu"
    },
    "offload_optimizer": {
      "device": "cpu"
    }
  },
  "gradient_accumulation_steps": 1,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true
  }
}
```

---

## 🖥️ Single GPU Optimization Techniques

While DeepSpeed excels in multi-GPU settings, it also offers strategies for single GPU environments:

### 🔹 CPU Offloading

- Transfers network parameters and optimizer states to CPU RAM, reducing GPU memory usage.
- Requires sufficient CPU memory.

### 🔹 Gradient Checkpointing

- Saves a subset of intermediate activations during the forward pass, recomputes during backward pass.
- Reduces memory at the cost of compute.

### 🔹 Low Precision Data Types

- **FP16:** Faster training, reduced memory; requires hardware support.
- **BF16:** Near FP32 accuracy, wider range; supported on newer CPUs/GPUs.

---

## ⚡ Memory-Efficient Optimizers

DeepSpeed supports various optimizers for performance and memory savings:

- **Adam (CPU)**
- **AdamW (CPU)**
- **FusedAdam (GPU)**
- **FusedLamb (GPU)**
- **OnebitAdam (GPU)**
- **ZeroOneAdam (GPU)**
- **OnebitLamb (GPU)**

---

## 🛠️ Sample DeepSpeed Config (Single GPU + Offloading)

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu" },
    "offload_param": { "device": "cpu" }
  },
  "gradient_checkpointing": true,
  "fp16": { "enabled": true }
}
```

---

## 📚 Additional Resources

- [DeepSpeed ZeRO Documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html)
- [ZeRO-Offload Tutorial](https://www.deepspeed.ai/tutorials/zero-offload/)
- [ZeRO-Infinity Overview](https://www.deepspeed.ai/2021/03/07/zero3-offload.html)
- [DeepSpeed Config JSON Guide](https://www.deepspeed.ai/docs/config-json/)
- [Hugging Face DeepSpeed Integration](https://huggingface.co/docs/transformers/en/deepspeed)

