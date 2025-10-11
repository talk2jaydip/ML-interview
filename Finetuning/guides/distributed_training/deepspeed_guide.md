# DeepSpeed Guide

## Overview

DeepSpeed is a deep learning optimization library developed by Microsoft that provides advanced distributed training capabilities with memory optimizations through ZeRO (Zero Redundancy Optimizer) technology.

## DeepSpeed ZeRO Stages

### ZeRO-0: Standard Data Parallelism
- **Memory**: Full model replication
- **Communication**: All-reduce gradients
- **Use Case**: Small models, baseline comparison

### ZeRO-1: Optimizer State Partitioning
- **Memory**: Partitions optimizer states
- **Communication**: All-reduce gradients
- **Use Case**: Medium models with optimizer memory issues

### ZeRO-2: Gradient and Optimizer Partitioning
- **Memory**: Partitions gradients and optimizer states
- **Communication**: Reduce-scatter gradients
- **Use Case**: Large models with gradient memory issues

### ZeRO-3: Full Model Sharding
- **Memory**: Partitions model, gradients, and optimizer states
- **Communication**: Reduce-scatter gradients, gather parameters
- **Use Case**: Very large models that don't fit in memory

## Configuration Files

### Basic ZeRO-3 Configuration

```json
{
  "fp16": {
    "enabled": "auto",
    "auto_cast": false,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": false
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 10,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

### ZeRO-2 Configuration

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  }
}
```

## Usage with LLaMA-Factory

### Single GPU/ZeRO-0

```bash
python src/train.py examples/train_lora/llama3_lora_sft.yaml
```

### Multi-GPU ZeRO-3

```bash
deepspeed src/train.py examples/train_lora/llama3_lora_sft_ds3.yaml
```

### Multi-Node ZeRO-3

```bash
# On master node
deepspeed --hostfile hostfile --master_addr $MASTER_ADDR --master_port $MASTER_PORT src/train.py config.yaml

# hostfile format:
# node1 slots=8
# node2 slots=8
```

## Hardware Requirements

### Single Node Requirements

| GPUs | Model Size | GPU Memory | CPU Memory | Network |
|------|------------|------------|------------|---------|
| 2 | 7B-13B | 16GB+ | 64GB+ | N/A |
| 4 | 7B-30B | 16GB+ | 128GB+ | N/A |
| 8 | 7B-70B | 24GB+ | 256GB+ | N/A |

### Multi-Node Requirements

| Nodes | GPUs/Node | Model Size | Network | Storage |
|-------|-----------|------------|---------|---------|
| 2 | 4 | 70B+ | 100Gbps+ | Shared |
| 4 | 8 | 100B+ | Infiniband | Shared |
| 8 | 8 | 200B+ | Infiniband | Shared |

## Performance Optimization

### Memory Optimization

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "cpu_offload": true,
    "cpu_offload_use_pin_memory": true
  }
}
```

### Communication Optimization

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 2e8,
    "allgather_bucket_size": 2e8
  }
}
```

### Gradient Accumulation

```yaml
gradient_accumulation_steps: 8  # Adjust based on GPU count
per_device_train_batch_size: 1
```

## Monitoring and Debugging

### DeepSpeed Monitoring

```python
def monitor_deepspeed():
    # Monitor memory usage per GPU
    # Track communication overhead
    # Check ZeRO stage effectiveness
    # Validate parameter gathering
    pass
```

### Common Issues

1. **Out of Memory**
   - Increase offloading
   - Reduce batch size
   - Use gradient accumulation

2. **Communication Bottlenecks**
   - Check network bandwidth
   - Optimize bucket sizes
   - Use overlap_comm

3. **Training Instability**
   - Use mixed precision
   - Adjust learning rate
   - Check gradient synchronization

## Best Practices

### 1. Configuration Selection

```python
def select_zeRO_stage(model_size, gpu_memory, num_gpus):
    if model_size <= 7e9 and gpu_memory >= 16:
        return 2  # ZeRO-2 for smaller models
    else:
        return 3  # ZeRO-3 for large models
```

### 2. Batch Size Optimization

```python
def optimize_batch_size(num_gpus, gpu_memory, model_size):
    # Calculate optimal batch size
    # Consider gradient accumulation
    # Balance memory and speed
    pass
```

### 3. Learning Rate Scaling

```yaml
# Scale learning rate with batch size
learning_rate: 1.0e-4  # Base learning rate
# Effective batch size = per_device_batch * accumulation_steps * num_gpus
```

## Advanced Features

### Activation Checkpointing

```json
{
  "activation_checkpointing": {
    "partition_activations": false,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false
  }
}
```

### Model Parallelism

```json
{
  "model_parallelism": {
    "model_parallel_size": 2,
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1
  }
}
```

### Custom ZeRO Configuration

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "custom_partition": true,
    "custom_partition_schedule": "schedule.json"
  }
}
```

## Multi-Node Setup

### Network Configuration

```bash
# Check network connectivity
ping -c 3 node1
ping -c 3 node2

# Test bandwidth
ibstat  # For Infiniband
ethtool eth0  # For Ethernet
```

### Environment Setup

```bash
# Set environment variables
export MASTER_ADDR=node1
export MASTER_PORT=29500
export CUDA_VISIBLE_DEVICES=0,1,2,3

# On all nodes
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
```

### Hostfile Configuration

```
node1 slots=8
node2 slots=8
node3 slots=8
node4 slots=8
```

## Performance Benchmarks

### Memory Scaling

| GPUs | Model Size | ZeRO Stage | Memory/GPU | Speedup |
|------|------------|------------|------------|---------|
| 1 | 7B | N/A | 16GB | 1x |
| 4 | 7B | ZeRO-3 | 4GB | 3.5x |
| 8 | 7B | ZeRO-3 | 2GB | 6.8x |
| 4 | 30B | ZeRO-3 | 8GB | 3.2x |
| 8 | 70B | ZeRO-3 | 12GB | 5.5x |

### Communication Overhead

| Operation | Time (ms) | Percentage | Optimization |
|-----------|-----------|------------|--------------|
| Forward | 45 | 60% | Activation checkpointing |
| Backward | 20 | 27% | Gradient accumulation |
| Communication | 10 | 13% | Overlap comm |

## Next Steps

After mastering DeepSpeed:
- Scale to 100B+ models
- Use model parallelism for very large models
- Optimize communication patterns
- Implement custom ZeRO configurations
- Monitor and debug distributed training

For hands-on examples, see the [notebooks](../../notebooks/distributed_training/) directory.
