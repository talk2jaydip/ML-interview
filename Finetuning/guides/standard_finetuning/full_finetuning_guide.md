# Full Fine-tuning Guide

## Overview

Full fine-tuning updates all parameters of the pre-trained model, providing the most comprehensive adaptation but requiring significant computational resources.

## When to Use Full Fine-tuning

- **Small to medium models** (< 13B parameters)
- **Sufficient hardware resources** (24GB+ GPU memory)
- **Maximum performance requirements**
- **Domain-specific fine-tuning** where every parameter matters
- **Research and experimentation** with model capabilities

## Hardware Requirements

### Minimum Requirements
- **GPU Memory**: 24GB+ for 7B models, 48GB+ for 13B models
- **System RAM**: 64GB+
- **Storage**: 100GB+ for models and datasets
- **GPU**: RTX 3090/4090, A100, V100 (24GB+)

### Recommended Setup
- **GPU**: RTX 4090 (24GB) or RTX 3090 (24GB)
- **CPU**: Intel Core i9 or AMD Ryzen 9 with 16+ cores
- **RAM**: 128GB DDR4/DDR5
- **Storage**: 1TB NVMe SSD
- **Power Supply**: 1000W+ with appropriate GPU power connectors

## Configuration Parameters

### Model Configuration
```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true
```

### Training Method
```yaml
### method
stage: sft  # Supervised Fine-tuning
do_train: true
finetuning_type: full  # Full parameter updates
```

### Dataset Configuration
```yaml
### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
```

### Training Parameters
```yaml
### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5  # Lower learning rate for full fine-tuning
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true  # Use mixed precision
tf32: true  # Use tensor float 32 if available
```

### Output Configuration
```yaml
### output
output_dir: saves/llama3-8b/full/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
```

## Memory Optimization Strategies

### 1. Gradient Checkpointing
```yaml
gradient_checkpointing: true
use_reentrant: false
```

### 2. Mixed Precision Training
```yaml
bf16: true  # Use bfloat16
fp16: false  # Don't use float16 unless necessary
```

### 3. Model Loading Optimization
```yaml
torch_dtype: bfloat16  # Load model in bfloat16
device_map: auto  # Automatic device placement
low_cpu_mem_usage: true  # Reduce CPU memory usage
```

## Batch Size and Memory Trade-offs

| Model Size | GPU Memory | Batch Size | Gradient Accumulation | Effective Batch |
|------------|------------|------------|----------------------|-----------------|
| 7B | 24GB | 1 | 8 | 8 |
| 7B | 48GB | 2 | 4 | 8 |
| 13B | 48GB | 1 | 4 | 4 |
| 13B | 80GB | 1 | 8 | 8 |

## Training Scripts

### Basic Training Command
```bash
python src/train.py examples/train_full/llama3_full_sft.yaml
```

### Training with Custom Configuration
```bash
python src/train.py --config_path my_full_sft_config.yaml
```

### Resume Training
```bash
python src/train.py examples/train_full/llama3_full_sft.yaml \
  --resume_from_checkpoint saves/llama3-8b/full/sft/checkpoint-500
```

## Model Loading and Inference

### Loading Full Fine-tuned Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "saves/llama3-8b/full/sft"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate text
inputs = tokenizer("Explain quantum computing", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Using LLaMA-Factory Chat Interface
```python
from llamafactory import ChatModel

model = ChatModel(dict(
    model_name_or_path="saves/llama3-8b/full/sft",
    finetuning_type="full",
    template="llama3"
))

response = model.chat("What are the benefits of renewable energy?")
print(response)
```

## Performance Optimization

### 1. Flash Attention 2
```yaml
use_flash_attn: true  # Requires flash-attn package
```

### 2. Tensor Parallelism (if multiple GPUs)
```yaml
tensor_parallel_size: 2  # Split model across 2 GPUs
```

### 3. CPU Offloading
```yaml
cpu_offload: true  # Offload parameters to CPU when not in use
```

## Monitoring and Logging

### Enable Detailed Logging
```yaml
report_to: tensorboard  # or wandb, swanlab, mlflow
logging_steps: 10
save_steps: 100
eval_steps: 200
```

### Training Metrics to Monitor
- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should follow training loss closely
- **Learning Rate**: Monitor the learning rate schedule
- **GPU Memory Usage**: Ensure sufficient memory headroom
- **Training Speed**: Tokens per second

## Troubleshooting Common Issues

### 1. CUDA Out of Memory
```yaml
# Reduce memory usage
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
```

### 2. Training Instability
```yaml
# Stabilize training
learning_rate: 5.0e-6  # Lower learning rate
warmup_ratio: 0.2  # Longer warmup
lr_scheduler_type: linear  # More stable than cosine
```

### 3. Slow Training
```yaml
# Optimize for speed
bf16: true
tf32: true
dataloader_num_workers: 8
preprocessing_num_workers: 16
```

## Best Practices

### 1. Data Preparation
- Use high-quality, diverse datasets
- Ensure proper formatting and chat templates
- Validate data before training

### 2. Hyperparameter Tuning
- Start with conservative hyperparameters
- Monitor training dynamics
- Adjust based on validation performance

### 3. Model Selection
- Choose appropriate model size for your hardware
- Consider the trade-off between model size and performance
- Use model parallelism for very large models

### 4. Regular Checkpointing
```yaml
save_steps: 500
save_total_limit: 3  # Keep only last 3 checkpoints
```

## Comparison with Other Methods

| Aspect | Full Fine-tuning | LoRA | QLoRA |
|--------|------------------|------|-------|
| Memory Usage | High | Low | Very Low |
| Training Speed | Slow | Fast | Fast |
| Model Quality | High | High | Good |
| Deployment Size | Large | Small | Small |
| Hardware Required | 24GB+ GPU | 8GB+ GPU | 4GB+ GPU |

## Advanced Configuration Examples

### Multi-GPU Training
```yaml
# For multiple GPUs
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
# Use torchrun for multi-GPU
# torchrun --nproc_per_node=2 src/train.py config.yaml
```

### Long Context Fine-tuning
```yaml
cutoff_len: 4096  # Increase for longer sequences
max_position_embeddings: 4096
# May require more memory and longer training
```

### Domain-Specific Fine-tuning
```yaml
# Use domain-specific datasets
dataset: alpaca_en_demo,mathinstruct,codealpaca
learning_rate: 5.0e-6  # Lower for specialized domains
num_train_epochs: 5.0  # More epochs for better adaptation
```

## Next Steps

After mastering full fine-tuning, consider:
- Exploring LoRA for more efficient training
- Using QLoRA for large model fine-tuning
- Implementing distributed training for very large models
- Fine-tuning for specific tasks like tool calling or multimodal

For hands-on examples, see the [notebooks](../../notebooks/sft/) directory.
