# Advanced Optimizers Guide

## Overview

This guide covers advanced optimizers available in LLaMA-Factory that provide better training dynamics, memory efficiency, and convergence properties compared to standard optimizers.

## Available Optimizers

1. **Adam-mini** - Memory-efficient Adam with reduced memory usage
2. **Apollo** - Curvature-aware optimizer with adaptive learning rates
3. **BAdam** - Block-wise Adam for memory efficiency
4. **DFT** - Discrete Fourier Transform-based optimizer
5. **GaLore** - Gradient Low-Rank Projection optimizer
6. **Muon** - Momentum-based optimizer inspired by physics

## Adam-mini

Adam-mini is a memory-efficient variant of Adam that reduces memory usage while maintaining performance.

### Configuration

```yaml
### Adam-mini configuration
optim: adam_mini
learning_rate: 1.0e-4
adam_mini_lr: 5.0e-5  # Lower learning rate for mini steps
adam_mini_clip: 10.0  # Gradient clipping
adam_mini_update_freq: 10  # Update frequency
```

### When to Use

- **Memory-constrained** environments
- **Large models** where memory is critical
- **Long training** runs requiring memory efficiency
- **Research** with memory-efficient training

## Apollo

Apollo is a curvature-aware optimizer that adapts learning rates based on curvature information.

### Configuration

```yaml
### Apollo configuration
optim: apollo
learning_rate: 1.0e-4
apollo_lr: 5.0e-5
apollo_weight_decay: 0.01
apollo_update_freq: 20
```

### When to Use

- **Better convergence** is needed
- **Adaptive learning rates** are beneficial
- **Complex loss landscapes** require curvature awareness
- **Stable training** for difficult tasks

## BAdam

BAdam is a block-wise Adam optimizer that processes parameters in blocks for memory efficiency.

### Configuration

```yaml
### BAdam configuration
optim: badam
learning_rate: 1.0e-4
badam_block_size: 64
badam_update_freq: 10
badam_clip: 10.0
```

### When to Use

- **Very large models** that don't fit in memory
- **Memory bandwidth** is limited
- **Distributed training** scenarios
- **Batch processing** of parameters

## DFT

DFT uses discrete Fourier transform for efficient parameter updates in frequency domain.

### Configuration

```yaml
### DFT configuration
optim: dft
learning_rate: 1.0e-4
dft_lr: 5.0e-5
dft_update_freq: 50
dft_clip: 10.0
```

### When to Use

- **Frequency-domain** optimization is beneficial
- **Spectral properties** of gradients matter
- **Research** with frequency-based optimization
- **Signal processing** inspired optimization

## GaLore

GaLore uses gradient low-rank projection to reduce memory usage while maintaining gradient information.

### Configuration

```yaml
### GaLore configuration
optim: galore
learning_rate: 1.0e-4
galore_rank: 128
galore_update_freq: 20
galore_clip: 10.0
```

### When to Use

- **Memory efficiency** is critical
- **Gradient compression** is needed
- **Large batch training** scenarios
- **Low-rank structure** in gradients

## Muon

Muon is inspired by muon momentum and uses physics-inspired optimization principles.

### Configuration

```yaml
### Muon configuration
optim: muon
learning_rate: 1.0e-4
muon_lr: 5.0e-5
muon_momentum: 0.9
muon_clip: 10.0
```

### When to Use

- **Momentum-based** optimization is beneficial
- **Physics-inspired** methods are interesting
- **Stable convergence** is needed
- **Research** with novel optimizers

## Hardware Requirements

### Requirements by Optimizer

| Optimizer | GPU Memory | CPU Memory | Training Speed | Stability |
|-----------|------------|------------|----------------|-----------|
| Adam-mini | Low | Medium | Fast | High |
| Apollo | Medium | Medium | Medium | High |
| BAdam | Low | High | Medium | Medium |
| DFT | Medium | High | Slow | Medium |
| GaLore | Low | Medium | Medium | High |
| Muon | Medium | Medium | Medium | High |

## Performance Comparison

### Memory Usage

| Optimizer | 7B Model | 13B Model | 70B Model | Relative Speed |
|-----------|----------|-----------|-----------|----------------|
| Standard Adam | 16GB | 32GB | 160GB | 100% |
| Adam-mini | 12GB | 24GB | 120GB | 95% |
| Apollo | 18GB | 36GB | 180GB | 90% |
| BAdam | 10GB | 20GB | 100GB | 85% |
| GaLore | 8GB | 16GB | 80GB | 90% |

### Training Quality

| Optimizer | Convergence | Stability | Final Quality | Memory Efficiency |
|-----------|------------|-----------|---------------|-------------------|
| Adam-mini | Fast | High | Excellent | High |
| Apollo | Medium | High | Excellent | Medium |
| BAdam | Slow | Medium | Good | High |
| DFT | Slow | Medium | Good | Medium |
| GaLore | Medium | High | Very Good | High |
| Muon | Medium | High | Very Good | Medium |

## Best Practices

### 1. Optimizer Selection

```python
def select_optimizer(task, hardware, requirements):
    if requirements["memory_efficiency"] == "high":
        if hardware["memory"] < 16:
            return "adam_mini"  # Most memory efficient
        else:
            return "galore"  # Good balance
    elif requirements["stability"] == "high":
        return "apollo"  # Most stable
    elif requirements["speed"] == "high":
        return "badam"  # Fastest
    else:
        return "adam_mini"  # Default choice
```

### 2. Hyperparameter Tuning

```yaml
# Conservative settings
learning_rate: 5.0e-5
warmup_ratio: 0.2

# Optimizer-specific settings
adam_mini_lr: 2.5e-5
apollo_weight_decay: 0.01
galore_rank: 64
```

### 3. Training Configuration

```yaml
# Use appropriate batch sizes
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# Enable gradient clipping
max_grad_norm: 1.0
```

## Next Steps

After exploring advanced optimizers:
- Combine optimizers with different methods
- Use optimizer-specific learning rate schedules
- Monitor optimizer-specific metrics
- Experiment with custom optimizer configurations
- Research optimizer theoretical properties

For hands-on examples, see the [notebooks](../notebooks/advanced_methods/) directory.
