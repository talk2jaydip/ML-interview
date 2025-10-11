# Resource Optimization and Configuration Guide

This comprehensive guide covers resource optimization strategies and configuration recommendations for efficient fine-tuning with LLaMA-Factory:

1. **Hardware Requirements**: GPU, CPU, and memory specifications
2. **Memory Optimization**: Strategies for reducing memory usage
3. **Training Speed Optimization**: Techniques for faster training
4. **Configuration Recommendations**: Best settings for different scenarios
5. **Cost Optimization**: Balancing performance and resource usage

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Memory Optimization](#memory-optimization)
- [Training Speed Optimization](#training-speed-optimization)
- [Configuration Recommendations](#configuration-recommendations)
- [Cost Optimization](#cost-optimization)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Best Practices](#best-practices)

## Hardware Requirements

### GPU Requirements by Method

| Method | Model Size | GPU Memory | GPU Type | Training Time |
|--------|------------|------------|----------|---------------|
| **Full Fine-tuning** | 7B | 24GB+ | RTX 3090/4090 | 2-4 hours |
| **Full Fine-tuning** | 13B | 48GB+ | RTX 4090/A100 | 4-8 hours |
| **Full Fine-tuning** | 70B | 80GB+ | A100/H100 | 8-16 hours |
| **LoRA** | 7B | 8-12GB | RTX 3080+ | 1-2 hours |
| **LoRA** | 13B | 16-24GB | RTX 3090+ | 2-4 hours |
| **LoRA** | 70B | 40-80GB | A100/H100 | 4-8 hours |
| **QLoRA** | 7B | 4-8GB | RTX 3060+ | 1-3 hours |
| **QLoRA** | 13B | 8-12GB | RTX 3080+ | 2-4 hours |
| **QLoRA** | 70B | 24-48GB | RTX 4090/A100 | 6-12 hours |

### CPU and System Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **CPU Cores** | 8 | 16 | 32+ |
| **System RAM** | 32GB | 64GB | 128GB+ |
| **Storage (SSD)** | 100GB | 500GB | 1TB+ |
| **Network** | 1Gbps | 10Gbps | 40Gbps+ |

### Multi-GPU Setup

| GPUs | Model Size | Memory/GPU | Communication | Scalability |
|------|------------|------------|---------------|-------------|
| 2 | 13B-30B | 16GB+ | NVLink | 1.8x |
| 4 | 30B-70B | 16GB+ | NVLink/InfiniBand | 3.5x |
| 8 | 70B-100B | 24GB+ | InfiniBand | 6.8x |
| 16 | 100B+ | 40GB+ | InfiniBand | 12x+ |

## Memory Optimization

### 1. Model Loading Optimization

```python
# Optimize model loading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16
    device_map="auto",  # Automatic device placement
    low_cpu_mem_usage=True,  # Reduce CPU memory usage
    trust_remote_code=True
)
```

### 2. Gradient Checkpointing

```yaml
# Enable gradient checkpointing
gradient_checkpointing: true
use_reentrant: false  # More stable
```

### 3. Mixed Precision Training

```yaml
# Use mixed precision
bf16: true  # bfloat16
fp16: false  # Avoid float16 unless necessary
tf32: true  # Tensor Float 32
```

### 4. Batch Size Optimization

```python
def optimize_batch_size(gpu_memory, model_size, method):
    """Calculate optimal batch size based on available memory"""
    # Base memory per sample
    base_memory = {
        "full": 2.0,  # GB per sample for full fine-tuning
        "lora": 0.5,  # GB per sample for LoRA
        "qlora": 0.25  # GB per sample for QLoRA
    }

    available_memory = gpu_memory * 0.9  # Leave 10% headroom
    memory_per_sample = base_memory.get(method, 0.5)

    # Calculate batch size
    batch_size = int(available_memory / memory_per_sample)

    # Use gradient accumulation if batch size is too small
    if batch_size < 1:
        batch_size = 1
        gradient_accumulation = int(4 / batch_size)
    else:
        gradient_accumulation = 1

    return batch_size, gradient_accumulation
```

### 5. Parameter Offloading

```yaml
# CPU offloading for large models
cpu_offload: true
cpu_offload_use_pin_memory: true
```

## Training Speed Optimization

### 1. Data Loading Optimization

```yaml
# Optimize data loading
dataloader_num_workers: 4  # Number of data loading workers
preprocessing_num_workers: 16  # Parallel preprocessing
pin_memory: true  # Pin memory for faster transfer
```

### 2. Compilation Optimization

```python
# Use torch.compile for faster training
model = torch.compile(model, mode="max-autotune")
```

### 3. Flash Attention

```yaml
# Use Flash Attention 2 for faster attention computation
use_flash_attn: true  # Requires flash-attn package
```

### 4. Gradient Accumulation Strategy

```python
def optimize_gradient_accumulation(batch_size, target_batch_size, gpu_memory):
    """Optimize gradient accumulation for memory constraints"""

    # Calculate required accumulation steps
    accumulation_steps = max(1, target_batch_size // batch_size)

    # Adjust based on memory constraints
    if accumulation_steps > 32:
        accumulation_steps = 32  # Maximum recommended
        effective_batch = batch_size * accumulation_steps
    else:
        effective_batch = target_batch_size

    return accumulation_steps, effective_batch
```

### 5. Learning Rate Scheduling

```yaml
# Optimized learning rate schedule
lr_scheduler_type: cosine  # Better than linear
warmup_ratio: 0.1  # 10% warmup
learning_rate: 1.0e-4  # Base learning rate
```

## Configuration Recommendations

### 1. Method Selection by Hardware

```python
def select_optimal_method(gpu_memory, model_size, quality_requirement):
    """Select the best fine-tuning method based on hardware"""

    if gpu_memory >= 48 and model_size <= 13e9:
        return "full"  # Full fine-tuning for best quality
    elif gpu_memory >= 16 and model_size <= 70e9:
        return "lora"  # LoRA for balanced performance
    elif gpu_memory >= 8:
        return "qlora"  # QLoRA for memory efficiency
    else:
        return "qlora"  # Always fallback to QLoRA
```

### 2. Hyperparameter Optimization

| Method | Learning Rate | Batch Size | LoRA Rank | Quantization |
|--------|--------------|------------|-----------|--------------|
| **Full** | 5.0e-6 - 1.0e-5 | 1-4 | N/A | N/A |
| **LoRA** | 1.0e-4 - 3.0e-4 | 2-8 | 8-64 | N/A |
| **QLoRA** | 1.0e-4 - 2.0e-4 | 1-4 | 32-128 | 4-bit |

### 3. Memory-Efficient Configuration

```yaml
# For limited GPU memory (8-16GB)
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
lora_rank: 8  # Smaller rank
bf16: true
```

### 4. Speed-Optimized Configuration

```yaml
# For maximum training speed
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
use_flash_attn: true
dataloader_num_workers: 8
bf16: true
tf32: true
```

## Cost Optimization

### 1. Cloud GPU Selection

| GPU Type | Memory | Cost/Hour | Performance | Cost Efficiency |
|----------|--------|-----------|-------------|-----------------|
| **RTX 4090** | 24GB | $0.50-1.00 | Excellent | Very High |
| **A100** | 40GB | $3.00-4.00 | Excellent | High |
| **H100** | 80GB | $5.00-8.00 | Outstanding | Medium |
| **V100** | 32GB | $2.00-3.00 | Good | Medium |

### 2. Training Cost Calculation

```python
def calculate_training_cost(training_time_hours, cost_per_hour, num_gpus):
    """Calculate total training cost"""

    total_cost = training_time_hours * cost_per_hour * num_gpus

    # Add storage and data transfer costs (10-20%)
    additional_costs = total_cost * 0.15

    return total_cost + additional_costs
```

### 3. Cost-Effective Strategies

1. **Use QLoRA** for large models to reduce GPU requirements
2. **Spot instances** can reduce costs by 50-70%
3. **Gradient accumulation** to simulate larger batch sizes
4. **Early stopping** to avoid unnecessary training
5. **Model compression** for deployment efficiency

## Monitoring and Debugging

### 1. Resource Monitoring

```python
import psutil
import GPUtil

def monitor_resources():
    """Monitor system resources during training"""

    # GPU utilization
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.load*100:.1f}% | {gpu.memoryUsed}/{gpu.memoryTotal} MB")

    # CPU and memory
    print(f"CPU: {psutil.cpu_percent()}%")
    print(f"Memory: {psutil.virtual_memory().percent}%")
```

### 2. Training Metrics

```python
def log_training_metrics():
    """Log comprehensive training metrics"""

    metrics = {
        "gpu_memory_used": torch.cuda.memory_allocated() / 1024**3,
        "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
        "training_speed": "samples/sec",
        "loss": "current_loss",
        "learning_rate": "current_lr"
    }

    return metrics
```

### 3. Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **CUDA OOM** | Training crashes with OOM error | Reduce batch size, enable gradient checkpointing, use QLoRA |
| **Slow Training** | Low GPU utilization | Increase batch size, use Flash Attention, optimize data loading |
| **Training Instability** | Loss spikes, poor convergence | Reduce learning rate, increase warmup, use gradient clipping |
| **Memory Leaks** | Memory usage increases over time | Clear cache regularly, use gradient checkpointing, monitor memory |

## Best Practices

### 1. Configuration Templates

```yaml
# Template for different scenarios
templates = {
    "high_quality": {
        "method": "full",
        "learning_rate": 5.0e-6,
        "batch_size": 2,
        "epochs": 5
    },
    "balanced": {
        "method": "lora",
        "learning_rate": 1.0e-4,
        "batch_size": 4,
        "epochs": 3
    },
    "fast_training": {
        "method": "qlora",
        "learning_rate": 2.0e-4,
        "batch_size": 2,
        "epochs": 2
    }
}
```

### 2. Automated Optimization

```python
def auto_optimize_config(base_config, hardware_constraints):
    """Automatically optimize configuration based on hardware"""

    # Detect available resources
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    num_gpus = torch.cuda.device_count()

    # Select optimal method
    method = select_optimal_method(gpu_memory, base_config["model_size"])

    # Optimize batch size and accumulation
    batch_size, accumulation = optimize_batch_size(
        gpu_memory, base_config["model_size"], method
    )

    # Update configuration
    config = base_config.copy()
    config.update({
        "finetuning_type": method,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "bf16": True,
        "gradient_checkpointing": method == "qlora"
    })

    return config
```

### 3. Resource-Efficient Training

1. **Use QLoRA** for large models when possible
2. **Enable gradient checkpointing** for memory savings
3. **Use mixed precision** training (bf16)
4. **Optimize data loading** with appropriate number of workers
5. **Monitor resource usage** and adjust accordingly
6. **Use early stopping** to avoid unnecessary training
7. **Save checkpoints strategically** to save storage

## Summary

This guide provides comprehensive strategies for optimizing resource usage in LLaMA-Factory:

- **Hardware Selection**: Choose appropriate GPUs and systems for your needs
- **Memory Optimization**: Use gradient checkpointing, mixed precision, and QLoRA
- **Speed Optimization**: Optimize batch sizes, use Flash Attention, and efficient data loading
- **Cost Optimization**: Balance performance with resource costs
- **Monitoring**: Track resource usage and training metrics
- **Best Practices**: Use proven configurations and optimization strategies

For hands-on examples, see the [notebooks](../notebooks/) directory.
