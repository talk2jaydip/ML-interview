# Standard Fine-tuning Methods Guide

This comprehensive guide covers the three main standard fine-tuning approaches available in LLaMA-Factory:

1. **Full Fine-tuning** - Complete model parameter updates
2. **LoRA Fine-tuning** - Parameter-efficient fine-tuning using Low-Rank Adaptation
3. **QLoRA Fine-tuning** - Quantized LoRA for memory-efficient training

## Table of Contents

- [Full Fine-tuning](#full-fine-tuning)
- [LoRA Fine-tuning](#lora-fine-tuning)
- [QLoRA Fine-tuning](#qlora-fine-tuning)
- [Hardware Requirements](#hardware-requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training Configuration](#training-configuration)
- [Evaluation and Benchmarking](#evaluation-and-benchmarking)

## Full Fine-tuning

Full fine-tuning updates all parameters of the pre-trained model, making it the most comprehensive but also the most resource-intensive approach.

### Key Characteristics

- **Parameter Updates**: All model parameters are updated during training
- **Memory Requirements**: Highest memory consumption (requires full model precision)
- **Training Speed**: Slower due to updating all parameters
- **Model Quality**: Potentially highest quality results
- **Storage**: Largest model size after training

### When to Use

- When you have sufficient computational resources
- When maximum model performance is required
- For smaller models (< 7B parameters)
- When fine-tuning for specific domains where every parameter matters

### Configuration Example

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
lora_rank: 8  # Not used in full fine-tuning
lora_target: all  # Not used in full fine-tuning

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/full/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

### Resource Requirements

| Model Size | GPU Memory | Training Time | Batch Size |
|------------|------------|---------------|------------|
| 7B | 24GB | 2-4 hours | 1-2 |
| 13B | 48GB | 4-8 hours | 1 |
| 70B | 120GB+ | 12-24 hours | 1 |

## LoRA Fine-tuning

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that adds small, trainable adapters to the model while keeping the original parameters frozen.

### Key Characteristics

- **Parameter Updates**: Only LoRA adapters are trained (~0.1-1% of parameters)
- **Memory Requirements**: Significantly reduced memory usage
- **Training Speed**: Faster training due to fewer parameters
- **Model Quality**: Comparable to full fine-tuning for many tasks
- **Storage**: Smaller adapter files for deployment

### When to Use

- For models larger than 7B parameters
- When memory is limited
- For quick experimentation and iteration
- When you need to fine-tune multiple models
- For deployment with multiple task-specific adapters

### LoRA Configuration Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `lora_rank` | Rank of LoRA matrices | 8, 16, 32, 64 |
| `lora_alpha` | Scaling parameter | 16, 32, 64 |
| `lora_dropout` | Dropout probability | 0.0, 0.1, 0.2 |
| `lora_target` | Target modules for LoRA | all, q_proj, k_proj, v_proj, o_proj |

### Configuration Example

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 32
lora_dropout: 0.1
lora_target: all

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

### Resource Requirements

| Model Size | GPU Memory | Training Time | Batch Size |
|------------|------------|---------------|------------|
| 7B | 8-12GB | 1-2 hours | 2-4 |
| 13B | 16-24GB | 2-4 hours | 1-2 |
| 70B | 48-80GB | 4-8 hours | 1 |

## QLoRA Fine-tuning

QLoRA (Quantized Low-Rank Adaptation) combines 4-bit quantization with LoRA to enable fine-tuning of large models on consumer hardware.

### Key Characteristics

- **Quantization**: 4-bit NormalFloat quantization of model weights
- **Memory Requirements**: Dramatically reduced memory usage
- **Training Speed**: Efficient training on limited hardware
- **Model Quality**: Maintains quality while reducing memory
- **Compatibility**: Works with various quantization formats

### When to Use

- When training very large models (70B+) on limited hardware
- For consumer GPUs with limited VRAM
- When you need to fine-tune multiple large models
- For experimentation with different model sizes

### QLoRA Configuration Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `quantization_bit` | Quantization precision | 4, 8 |
| `quantization_type` | Quantization algorithm | nf4, fp4, int4 |
| `double_quantization` | Nested quantization | true, false |
| `quantization_method` | Quantization method | bitsandbytes, hqq, eetq |

### Configuration Example

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
quantization_type: nf4
double_quantization: true
quantization_method: bitsandbytes
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.1
lora_target: all

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/qlora/sft
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
```

### Resource Requirements

| Model Size | GPU Memory | Training Time | Batch Size |
|------------|------------|---------------|------------|
| 7B | 4-8GB | 1-3 hours | 1-2 |
| 13B | 8-12GB | 2-4 hours | 1 |
| 70B | 24-40GB | 6-12 hours | 1 |

## Hardware Requirements

### Minimum System Requirements

| Method | GPU Memory | CPU RAM | Storage |
|--------|------------|---------|---------|
| Full Fine-tuning | 24GB+ | 64GB+ | 100GB+ |
| LoRA | 8GB+ | 32GB+ | 50GB+ |
| QLoRA | 4GB+ | 16GB+ | 30GB+ |

### Recommended Hardware

- **GPU**: NVIDIA RTX 4090 (24GB) or RTX 3090 (24GB) for LoRA/QLoRA
- **CPU**: Intel Core i7/i9 or AMD Ryzen 7/9 with 16+ cores
- **RAM**: 64GB+ system RAM
- **Storage**: 500GB+ NVMe SSD for fast data loading
- **Network**: High-speed internet for downloading models and datasets

## Dataset Preparation

### Dataset Formats Supported

1. **Instruction-Response Format** (Alpaca-style)
```json
[
  {
    "instruction": "Describe a process",
    "input": "optional context",
    "output": "expected response"
  }
]
```

2. **Conversation Format** (ShareGPT-style)
```json
[
  {
    "conversations": [
      {"from": "human", "value": "Hello"},
      {"from": "gpt", "value": "Hi there!"}
    ]
  }
]
```

3. **Preference Format** (for DPO/KTO)
```json
[
  {
    "conversations": [...],
    "chosen": {"from": "gpt", "value": "preferred response"},
    "rejected": {"from": "gpt", "value": "dispreferred response"}
  }
]
```

### Dataset Configuration

```yaml
### dataset
dataset: alpaca_en_demo  # or your custom dataset
template: llama3  # chat template for the model
cutoff_len: 2048  # maximum sequence length
max_samples: 1000  # limit dataset size for testing
overwrite_cache: true  # rebuild dataset cache
preprocessing_num_workers: 16  # parallel processing
```

### Custom Dataset Integration

1. **Add dataset to `data/dataset_info.json`**:
```json
"my_custom_dataset": {
  "hf_hub_url": "myorg/my_custom_dataset",
  "columns": {
    "prompt": "instruction",
    "response": "output"
  }
}
```

2. **Use in training configuration**:
```yaml
dataset: my_custom_dataset
```

## Training Configuration

### Hyperparameter Tuning

#### Learning Rate Schedules
- **Cosine**: Smooth decay, good for fine-tuning
- **Linear**: Simple decay, good for experimentation
- **Constant**: Fixed learning rate, requires careful tuning

#### Batch Size Configuration
```yaml
per_device_train_batch_size: 2  # samples per GPU per step
gradient_accumulation_steps: 4  # accumulate gradients over steps
effective_batch_size: 8  # total batch size = per_device * accumulation * num_gpus
```

### Optimization Settings

```yaml
### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true  # Use bfloat16 for faster training
tf32: true  # Use tensor float 32 if available
```

### Memory Optimization

```yaml
# For memory-limited systems
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true  # Trade compute for memory
use_reentrant: false  # For gradient checkpointing stability
```

## Evaluation and Benchmarking

### Built-in Evaluation

```yaml
### eval
eval_dataset: alpaca_en_demo
val_size: 0.1  # Use 10% of data for validation
per_device_eval_batch_size: 1
eval_strategy: steps  # Evaluate during training
eval_steps: 500  # Evaluate every 500 steps
```

### Manual Evaluation

```python
from llamafactory import ChatModel

# Load fine-tuned model
model = ChatModel(dict(
    model_name_or_path="saves/llama3-8b/lora/sft",
    finetuning_type="lora",
    template="llama3"
))

# Generate response
response = model.chat("Explain quantum computing")
print(response)
```

### Performance Metrics

- **Training Loss**: Monitor convergence
- **Validation Loss**: Check for overfitting
- **Perplexity**: Measure language modeling quality
- **Task-specific Metrics**: Accuracy, F1, BLEU, ROUGE depending on task

## Best Practices

### 1. Start Small
- Begin with small datasets and models
- Use LoRA for initial experiments
- Gradually increase complexity

### 2. Monitor Training
- Track loss curves with `plot_loss: true`
- Use validation sets to prevent overfitting
- Monitor GPU utilization and memory usage

### 3. Hyperparameter Tuning
- Start with recommended hyperparameters
- Tune learning rate, batch size, and LoRA rank
- Use learning rate schedulers for stability

### 4. Dataset Quality
- Ensure data quality and diversity
- Use appropriate chat templates
- Balance dataset if needed

### 5. Resource Management
- Use gradient checkpointing for large models
- Choose appropriate quantization for your hardware
- Monitor memory usage during training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable gradient checkpointing
   - Switch to QLoRA

2. **Slow Training**
   - Increase batch size if memory allows
   - Use mixed precision (bf16)
   - Enable tensor float 32 (tf32)

3. **Poor Model Performance**
   - Increase training epochs
   - Use higher LoRA rank
   - Try full fine-tuning
   - Improve dataset quality

4. **Training Instability**
   - Reduce learning rate
   - Use gradient clipping
   - Increase warmup steps
   - Use cosine learning rate schedule

## Next Steps

After mastering standard fine-tuning methods, consider exploring:

- [Preference Optimization](preference_optimization/README.md)
- [Quantization Methods](quantization_methods/README.md)
- [Advanced Parameter-Efficient Methods](advanced_methods/README.md)
- [Distributed Training](distributed_training/README.md)

For hands-on examples, check the [notebooks](../../notebooks/sft/) directory.
