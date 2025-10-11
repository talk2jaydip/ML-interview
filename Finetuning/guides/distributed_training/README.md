# Distributed Training Guide

This comprehensive guide covers distributed training techniques for scaling large language model training across multiple GPUs and nodes:

1. **DeepSpeed** - Microsoft DeepSpeed with ZeRO stages
2. **FSDP (Fully Sharded Data Parallel)** - PyTorch's distributed training
3. **Ray** - Distributed computing framework

## Table of Contents

- [Overview](#overview)
- [DeepSpeed](#deepspeed)
- [FSDP (Fully Sharded Data Parallel)](#fsdp-fully-sharded-data-parallel)
- [Ray](#ray)
- [Hardware Requirements](#hardware-requirements)
- [Performance Comparison](#performance-comparison)
- [Best Practices](#best-practices)

## Overview

Distributed training enables scaling model training across multiple GPUs and nodes, allowing for:

- **Larger Models**: Train models that don't fit in single GPU memory
- **Larger Batch Sizes**: Use larger effective batch sizes for better training
- **Faster Training**: Parallelize computation across multiple devices
- **Better Utilization**: Make use of multiple GPUs in a system

### Types of Parallelism

1. **Data Parallelism**: Split data across devices
2. **Model Parallelism**: Split model across devices
3. **Pipeline Parallelism**: Split model into stages
4. **Tensor Parallelism**: Split individual operations

## DeepSpeed

DeepSpeed is a deep learning optimization library that provides distributed training capabilities with advanced memory optimizations.

### ZeRO Stages

- **ZeRO-0**: Standard data parallelism
- **ZeRO-1**: Optimizer state partitioning
- **ZeRO-2**: Optimizer + gradient partitioning
- **ZeRO-3**: Optimizer + gradient + parameter partitioning

### Configuration

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "gradient_clipping": "auto",
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

### Usage with LLaMA-Factory

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/deepspeed/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### deepspeed
deepspeed: examples/deepspeed/ds_z3_config.json
```

## FSDP (Fully Sharded Data Parallel)

FSDP is PyTorch's native distributed training framework that shards model parameters, gradients, and optimizer states across devices.

### Configuration

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/fsdp/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### fsdp
fsdp: examples/accelerate/fsdp_config.yaml
```

### FSDP Configuration File

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
```

## Ray

Ray is a distributed computing framework that provides simple APIs for scaling training across multiple nodes.

### Configuration

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/ray/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### ray
ray: examples/train_lora/llama3_lora_sft_ray.yaml
```

## Hardware Requirements

### Requirements by Framework

| Framework | Min GPUs | GPU Memory | CPU Memory | Network |
|-----------|----------|------------|------------|---------|
| DeepSpeed ZeRO-3 | 2 | 8GB+ | 32GB+ | 10Gbps+ |
| FSDP | 2 | 8GB+ | 32GB+ | 10Gbps+ |
| Ray | 2 | 8GB+ | 16GB+ | 1Gbps+ |

### Multi-Node Setup

For multi-node distributed training, ensure:
- **Network**: High-speed interconnect (Infiniband, 100Gbps Ethernet)
- **Storage**: Shared filesystem (NFS, Lustre, etc.)
- **Configuration**: Consistent hardware across nodes
- **Software**: Same CUDA, PyTorch versions across nodes

## Performance Comparison

### Memory Efficiency

| Framework | 7B Model | 13B Model | 70B Model | Max Model Size |
|-----------|----------|-----------|-----------|----------------|
| Single GPU | 16GB | 32GB | ❌ | 13B |
| DeepSpeed ZeRO-3 | 4GB | 8GB | 24GB | 100B+ |
| FSDP | 6GB | 12GB | 32GB | 100B+ |
| Ray | 8GB | 16GB | 48GB | 50B+ |

### Training Speed

| Framework | Setup Time | Communication | Memory Overhead | Scalability |
|-----------|------------|---------------|-----------------|-------------|
| DeepSpeed ZeRO-3 | Medium | Low | Low | Excellent |
| FSDP | Low | Medium | Medium | Very Good |
| Ray | High | High | High | Good |

## Best Practices

### 1. Framework Selection

```python
def select_distributed_framework(hardware, model_size, expertise):
    if hardware["multi_node"]:
        return "deepspeed"  # Best multi-node support
    elif model_size > 30e9:
        return "deepspeed"  # Best memory efficiency
    elif expertise == "pytorch":
        return "fsdp"  # Native PyTorch
    else:
        return "ray"  # Simple API
```

### 2. Configuration Tuning

```yaml
# DeepSpeed ZeRO-3
deepspeed_stage: 3
offload_optimizer: true
offload_param: true

# FSDP
fsdp_sharding_strategy: FULL_SHARD
fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
```

### 3. Monitoring

```python
def monitor_distributed_training():
    # Track GPU utilization across devices
    # Monitor communication overhead
    # Check memory usage per process
    # Validate gradient synchronization
    pass
```

## Next Steps

After setting up distributed training:
- Scale to larger models
- Use multiple nodes
- Optimize communication
- Monitor distributed training metrics
- Handle distributed training failures

For hands-on examples, see the [notebooks](../notebooks/distributed_training/) directory.
