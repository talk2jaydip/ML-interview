# Advanced Parameter-Efficient Methods Guide

This comprehensive guide covers advanced parameter-efficient fine-tuning techniques and optimization methods:

1. **LoRA+ (LoRA Plus)** - Improved LoRA with different learning rates
2. **OFT (Orthogonal Fine-tuning)** - Orthogonal matrix decomposition
3. **PISSA (Principal Singular values and Singular vectors Adaptation)** - SVD-based adaptation
4. **QOFT (Quantized Orthogonal Fine-tuning)** - Quantized orthogonal adaptation
5. **Advanced Optimizers** - Adam-mini, Apollo, BAdam, DFT, GaLore, Muon

## Table of Contents

- [Overview](#overview)
- [LoRA+ (LoRA Plus)](#lora-lora-plus)
- [OFT (Orthogonal Fine-tuning)](#oft-orthogonal-fine-tuning)
- [PISSA (Principal Singular values and Singular vectors Adaptation)](#pissa-principal-singular-values-and-singular-vectors-adaptation)
- [QOFT (Quantized Orthogonal Fine-tuning)](#qoft-quantized-orthogonal-fine-tuning)
- [Advanced Optimizers](#advanced-optimizers)
- [Hardware Requirements](#hardware-requirements)
- [Performance Comparison](#performance-comparison)
- [Best Practices](#best-practices)

## Overview

Advanced parameter-efficient methods improve upon standard LoRA by using more sophisticated mathematical techniques:

- **Orthogonal Methods**: Maintain orthogonality constraints for better representation
- **SVD-based Methods**: Use singular value decomposition for efficient adaptation
- **Quantized Methods**: Combine parameter efficiency with quantization
- **Advanced Optimizers**: Use specialized optimization algorithms for better training

### Comparison with Standard LoRA

| Method | Parameters | Quality | Speed | Memory | Complexity |
|--------|------------|---------|-------|--------|------------|
| LoRA | Low | Good | Fast | Low | Low |
| LoRA+ | Low | Better | Fast | Low | Low |
| OFT | Low | Better | Medium | Low | Medium |
| PISSA | Low | Better | Medium | Low | High |
| QOFT | Very Low | Good | Fast | Very Low | Medium |

## LoRA+ (LoRA Plus)

LoRA+ improves LoRA by using different learning rates for the A and B matrices, leading to better training dynamics.

### How LoRA+ Works

1. **Asymmetric Learning Rates**: Use different learning rates for A and B matrices
2. **Matrix Decomposition**: A has learning rate η, B has learning rate η/ratio
3. **Better Convergence**: Improved convergence properties
4. **Dropout Integration**: Better regularization through asymmetric updates

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
lora_alpha: 32
lora_dropout: 0.1
lora_target: all
loraplus_lr_ratio: 16  # Different learning rates for A and B

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/loraplus/sft
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

## OFT (Orthogonal Fine-tuning)

OFT uses orthogonal matrix decomposition to maintain orthogonality constraints during fine-tuning.

### How OFT Works

1. **Orthogonal Initialization**: Initialize with orthogonal matrices
2. **Orthogonal Updates**: Maintain orthogonality during training
3. **Block-wise Decomposition**: Use block-wise orthogonal updates
4. **Stable Training**: Better numerical stability

### Configuration

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: oft
lora_rank: 8  # Used as block size for OFT
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
output_dir: saves/llama3-8b/oft/sft
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

## PISSA (Principal Singular values and Singular vectors Adaptation)

PISSA uses SVD decomposition to adapt only the most important singular values and vectors.

### How PISSA Works

1. **SVD Decomposition**: Decompose weight matrices using SVD
2. **Principal Components**: Adapt only top singular values and vectors
3. **Low-rank Update**: Update only important components
4. **Efficient Storage**: Store only adapted components

### Configuration

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: pissa
lora_rank: 8  # Number of singular values to adapt
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
output_dir: saves/llama3-8b/pissa/sft
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

## QOFT (Quantized Orthogonal Fine-tuning)

QOFT combines orthogonal fine-tuning with quantization for maximum efficiency.

### How QOFT Works

1. **Orthogonal Updates**: Use orthogonal matrix updates
2. **Quantized Storage**: Store updates in quantized format
3. **Block-wise Quantization**: Apply quantization per block
4. **Dynamic Precision**: Adapt precision based on importance

### Configuration

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: oft
quantization_bit: 4
quantization_type: qoft
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
output_dir: saves/llama3-8b/qoft/sft
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

## Advanced Optimizers

### Adam-mini

Adam-mini is a memory-efficient optimizer that reduces memory usage while maintaining performance.

```yaml
### Adam-mini configuration
optim: adam_mini
learning_rate: 1.0e-4
adam_mini_lr: 5.0e-5  # Lower learning rate for mini steps
adam_mini_clip: 10.0  # Gradient clipping
adam_mini_update_freq: 10  # Update frequency
```

### Apollo

Apollo is an optimizer that adapts learning rates based on curvature information.

```yaml
### Apollo configuration
optim: apollo
learning_rate: 1.0e-4
apollo_lr: 5.0e-5
apollo_weight_decay: 0.01
apollo_update_freq: 20
```

### BAdam

BAdam is a block-wise Adam optimizer for memory efficiency.

```yaml
### BAdam configuration
optim: badam
learning_rate: 1.0e-4
badam_block_size: 64
badam_update_freq: 10
badam_clip: 10.0
```

### DFT

DFT uses discrete Fourier transform for efficient parameter updates.

```yaml
### DFT configuration
optim: dft
learning_rate: 1.0e-4
dft_lr: 5.0e-5
dft_update_freq: 50
dft_clip: 10.0
```

### GaLore

GaLore uses gradient low-rank projection for memory-efficient training.

```yaml
### GaLore configuration
optim: galore
learning_rate: 1.0e-4
galore_rank: 128
galore_update_freq: 20
galore_clip: 10.0
```

### Muon

Muon is an optimizer inspired by muon momentum for better convergence.

```yaml
### Muon configuration
optim: muon
learning_rate: 1.0e-4
muon_lr: 5.0e-5
muon_momentum: 0.9
muon_clip: 10.0
```

## Hardware Requirements

### Requirements by Method

| Method | GPU Memory | CPU RAM | Storage | GPU Compatibility |
|--------|------------|---------|---------|-------------------|
| LoRA+ | 8-16GB | 32GB | 30GB | Universal |
| OFT | 8-16GB | 32GB | 30GB | Universal |
| PISSA | 12-24GB | 64GB | 50GB | Universal |
| QOFT | 8-16GB | 32GB | 30GB | Universal |
| Advanced Optimizers | 8-16GB | 32GB | 30GB | Universal |

## Performance Comparison

### Quality Comparison

| Method | MMLU | GSM8K | HumanEval | Memory Usage | Training Speed |
|--------|------|-------|-----------|--------------|----------------|
| LoRA | 0.62 | 0.49 | 0.25 | 8GB | 100% |
| LoRA+ | 0.64 | 0.51 | 0.26 | 8GB | 98% |
| OFT | 0.63 | 0.50 | 0.26 | 8GB | 95% |
| PISSA | 0.65 | 0.52 | 0.27 | 8GB | 90% |
| QOFT | 0.61 | 0.48 | 0.24 | 6GB | 105% |

### Efficiency Comparison

| Method | Parameters | Memory Reduction | Training Speed | Quality |
|--------|------------|------------------|---------------|---------|
| LoRA | 0.1% | 10x | Fast | Good |
| LoRA+ | 0.1% | 10x | Fast | Better |
| OFT | 0.05% | 20x | Medium | Good |
| PISSA | 0.02% | 50x | Slow | Better |
| QOFT | 0.01% | 100x | Fast | Good |

## Best Practices

### 1. Method Selection

```python
def select_advanced_method(task_type, hardware, quality_requirement):
    if quality_requirement == "maximum":
        return "pissa"
    elif hardware == "limited":
        return "qoft"
    elif task_type == "vision":
        return "oft"
    else:
        return "loraplus"
```

### 2. Hyperparameter Tuning

```yaml
# Conservative settings
learning_rate: 5.0e-5
warmup_ratio: 0.2
gradient_checkpointing: true

# Method-specific settings
lora_rank: 16  # Higher rank for advanced methods
lora_alpha: 32  # Balanced alpha
```

### 3. Training Stability

```yaml
# Use conservative settings for advanced methods
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
```

## Next Steps

After exploring advanced methods:
- Combine methods (e.g., LoRA+ with QOFT)
- Use advanced optimizers with any method
- Experiment with different rank/alpha combinations
- Try method-specific configurations
- Monitor training dynamics carefully

For hands-on examples, see the [notebooks](../notebooks/advanced_methods/) directory.
