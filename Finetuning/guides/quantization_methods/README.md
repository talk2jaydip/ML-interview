# Quantization Methods Guide

This comprehensive guide covers advanced quantization techniques for efficient fine-tuning and inference of large language models:

1. **AWQ (Activation-aware Weight Quantization)** - Activation-aware quantization
2. **GPTQ (General Purpose Transformer Quantization)** - Post-training quantization
3. **AQLM (Activation-aware Quantization for Language Models)** - Advanced quantization
4. **OTFQ (On-the-fly Quantization)** - Dynamic quantization during training

## Table of Contents

- [Overview](#overview)
- [AWQ (Activation-aware Weight Quantization)](#awq-activation-aware-weight-quantization)
- [GPTQ (General Purpose Transformer Quantization)](#gptq-general-purpose-transformer-quantization)
- [AQLM (Activation-aware Quantization for Language Models)](#aqlm-activation-aware-quantization-for-language-models)
- [OTFQ (On-the-fly Quantization)](#otfq-on-the-fly-quantization)
- [Hardware Requirements](#hardware-requirements)
- [Performance Comparison](#performance-comparison)
- [Best Practices](#best-practices)

## Overview

Quantization reduces model precision from 32-bit or 16-bit floating point to lower precision (8-bit, 4-bit, or even 3-bit) to reduce memory usage and improve inference speed while maintaining model quality.

### Types of Quantization

1. **Post-Training Quantization**: Quantize after training
2. **Quantization-Aware Training**: Train with quantization
3. **Dynamic Quantization**: Quantize during inference
4. **Static Quantization**: Pre-compute quantization parameters

### Quantization Precision

| Precision | Bits | Memory Reduction | Quality Loss |
|-----------|------|------------------|--------------|
| FP32 | 32 | 0% | None |
| FP16 | 16 | 50% | Minimal |
| INT8 | 8 | 75% | Low |
| INT4 | 4 | 87.5% | Moderate |
| INT3 | 3 | 90.6% | High |

## AWQ (Activation-aware Weight Quantization)

AWQ is an activation-aware weight quantization method that protects salient weights during quantization.

### How AWQ Works

1. **Activation Analysis**: Analyze activation patterns to identify important weights
2. **Weight Protection**: Protect important weights from quantization
3. **Channel-wise Quantization**: Apply quantization per channel
4. **Scaling**: Scale remaining weights to compensate for quantization

### Key Advantages

- **Quality Preservation**: Maintains model quality better than uniform quantization
- **Activation Aware**: Uses real activation data for better quantization
- **Automatic**: No manual calibration required
- **Efficient**: Fast quantization process

### Configuration

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
quantization_type: awq  # AWQ quantization
double_quantization: false
quantization_method: awq

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/awq/lora/sft
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

### AWQ Parameters

| Parameter | Description | Values | Impact |
|-----------|-------------|--------|---------|
| `awq_block_size` | Block size for quantization | 64, 128 | Smaller = better quality, slower |
| `awq_group_size` | Group size for scaling | -1, 128 | -1 for channel-wise |
| `awq_w_bit` | Weight quantization bits | 4, 8 | Lower = more compression |
| `awq_version` | AWQ version | GEMM, GEMV | GEMM for modern GPUs |

## GPTQ (General Purpose Transformer Quantization)

GPTQ is a post-training quantization method that uses layer-wise quantization with optimal brain surgeon for weight updates.

### How GPTQ Works

1. **Layer-wise Quantization**: Quantize one layer at a time
2. **Optimal Brain Surgeon**: Use OBS to find optimal weight updates
3. **Hessian-based Updates**: Use second-order information for updates
4. **Iterative Process**: Iteratively quantize and update weights

### Key Advantages

- **Post-Training**: Can quantize any pre-trained model
- **Optimal Updates**: Uses second-order optimization
- **Layer-wise**: More stable than end-to-end quantization
- **Flexible**: Works with various quantization schemes

### Configuration

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
quantization_type: gptq  # GPTQ quantization
double_quantization: false
quantization_method: gptq

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/gptq/lora/sft
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

### GPTQ Parameters

| Parameter | Description | Values | Impact |
|-----------|-------------|--------|---------|
| `gptq_block_size` | Block size for quantization | 64, 128 | Smaller = better quality |
| `gptq_group_size` | Group size for scaling | -1, 128 | -1 for channel-wise |
| `gptq_w_bit` | Weight quantization bits | 4, 8 | Lower = more compression |
| `gptq_percdamp` | Damping for OBS | 0.01 | Higher = more conservative |

## AQLM (Activation-aware Quantization for Language Models)

AQLM is an advanced quantization method that uses multiple codebooks to achieve better compression ratios.

### How AQLM Works

1. **Multiple Codebooks**: Use multiple codebooks for better representation
2. **Vector Quantization**: Quantize weights using learned codebooks
3. **Activation Analysis**: Use activation data to optimize codebooks
4. **Layer-wise Optimization**: Optimize codebooks per layer

### Key Advantages

- **Better Compression**: Higher compression ratios than uniform quantization
- **Learned Codebooks**: Optimized for specific weight distributions
- **Activation Aware**: Uses activation data for optimization
- **Flexible**: Supports various precision levels

### Configuration

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
quantization_type: aqlm  # AQLM quantization
double_quantization: false
quantization_method: aqlm

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/aqlm/lora/sft
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

### AQLM Parameters

| Parameter | Description | Values | Impact |
|-----------|-------------|--------|---------|
| `aqlm_codebook_size` | Size of codebooks | 16, 32, 64 | Larger = better quality, more memory |
| `aqlm_block_size` | Block size for quantization | 64, 128 | Smaller = better quality |
| `aqlm_n_bits` | Bits per codebook entry | 8, 16 | Lower = more compression |
| `aqlm_scales_bits` | Bits for scales | 4, 8 | Lower = more compression |

## OTFQ (On-the-fly Quantization)

OTFQ performs quantization dynamically during training, adapting quantization parameters based on the current training state.

### How OTFQ Works

1. **Dynamic Quantization**: Quantize/dequantize weights during forward pass
2. **Adaptive Scaling**: Adjust quantization parameters based on gradients
3. **Training-time Optimization**: Optimize quantization for current training phase
4. **Mixed Precision**: Use different precision for different layers

### Key Advantages

- **Adaptive**: Adjusts to training dynamics
- **Dynamic**: No pre-computed quantization parameters
- **Training-aware**: Optimizes quantization for training
- **Flexible**: Can change precision during training

### Configuration

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
quantization_type: otfq  # OTFQ quantization
double_quantization: false
quantization_method: otfq

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/otfq/lora/sft
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

### OTFQ Parameters

| Parameter | Description | Values | Impact |
|-----------|-------------|--------|---------|
| `otfq_block_size` | Block size for quantization | 64, 128 | Smaller = better quality |
| `otfq_warmup_steps` | Warmup steps for quantization | 100, 500 | Longer = more stable |
| `otfq_decay` | Decay rate for quantization | 0.9, 0.99 | Higher = slower adaptation |
| `otfq_update_freq` | Update frequency | 10, 50 | Lower = more updates |

## Hardware Requirements

### Minimum Requirements by Method

| Method | GPU Memory | CPU RAM | Storage | GPU Compatibility |
|--------|------------|---------|---------|-------------------|
| AWQ | 8-16GB | 32GB | 30GB | Modern NVIDIA GPUs |
| GPTQ | 8-16GB | 32GB | 30GB | NVIDIA GPUs with CUDA |
| AQLM | 12-24GB | 64GB | 50GB | High-end GPUs |
| OTFQ | 8-16GB | 32GB | 30GB | Modern NVIDIA GPUs |

### Performance by Hardware

| Hardware | 7B Model | 13B Model | 70B Model | Quality |
|----------|----------|-----------|-----------|---------|
| RTX 4090 (24GB) | ✅ | ✅ | ⚠️ | Excellent |
| RTX 3090 (24GB) | ✅ | ✅ | ⚠️ | Very Good |
| RTX 3080 (12GB) | ✅ | ⚠️ | ❌ | Good |
| A100 (40GB) | ✅ | ✅ | ✅ | Excellent |
| V100 (32GB) | ✅ | ✅ | ⚠️ | Good |

## Performance Comparison

### Memory Usage

| Method | 7B Model | 13B Model | 70B Model | Quality Loss |
|--------|----------|-----------|-----------|--------------|
| FP16 | 14GB | 26GB | 140GB | None |
| AWQ (4-bit) | 4GB | 7GB | 35GB | Minimal |
| GPTQ (4-bit) | 4GB | 7GB | 35GB | Low |
| AQLM (4-bit) | 3GB | 6GB | 30GB | Low |
| OTFQ (4-bit) | 4GB | 7GB | 35GB | Minimal |

### Inference Speed

| Method | Tokens/sec | Memory Usage | Quality | Compatibility |
|--------|------------|--------------|---------|---------------|
| FP16 | 100 | High | 100% | Universal |
| AWQ (4-bit) | 150 | Low | 98% | Good |
| GPTQ (4-bit) | 140 | Low | 96% | Good |
| AQLM (4-bit) | 160 | Very Low | 97% | Limited |
| OTFQ (4-bit) | 145 | Low | 98% | Good |

### Training Speed

| Method | Training Time | GPU Memory | Stability | Quality |
|--------|---------------|------------|-----------|---------|
| Standard LoRA | 1-2 hours | 8-16GB | High | High |
| AWQ + LoRA | 1.5-3 hours | 4-8GB | High | High |
| GPTQ + LoRA | 1.5-3 hours | 4-8GB | High | High |
| AQLM + LoRA | 2-4 hours | 3-6GB | Medium | High |
| OTFQ + LoRA | 1.5-3 hours | 4-8GB | Medium | High |

## Best Practices

### 1. Method Selection

```python
def select_quantization_method(model_size, hardware, quality_requirement):
    if model_size <= 13e9 and hardware >= 16:  # 16GB GPU
        return "awq"  # Best balance
    elif model_size <= 30e9 and hardware >= 8:  # 8GB GPU
        return "gptq"  # Good compatibility
    elif model_size <= 70e9 and hardware >= 24:  # 24GB GPU
        return "aqlm"  # Best compression
    else:
        return "otfq"  # Most flexible
```

### 2. Quality vs Size Trade-off

```yaml
# Maximum quality
quantization_bit: 8
quantization_type: awq

# Balanced
quantization_bit: 4
quantization_type: awq

# Maximum compression
quantization_bit: 4
quantization_type: aqlm
```

### 3. Training Stability

```yaml
# Conservative settings for quantization
learning_rate: 5.0e-5  # Lower than standard LoRA
warmup_ratio: 0.2  # Longer warmup
gradient_checkpointing: true  # Memory efficiency
```

### 4. Evaluation

```python
def evaluate_quantized_model(model_path, eval_data):
    # Test on original precision tasks
    # Check for quantization artifacts
    # Measure accuracy drop
    pass
```

## Troubleshooting

### Common Issues

1. **Quantization Errors**
   - Check GPU compatibility
   - Update quantization libraries
   - Use supported model architectures

2. **Quality Degradation**
   - Try different quantization methods
   - Use higher precision
   - Increase training epochs

3. **Training Instability**
   - Reduce learning rate
   - Use gradient clipping
   - Increase warmup steps

4. **Memory Issues**
   - Use smaller batch sizes
   - Enable gradient checkpointing
   - Try different quantization methods

### Debugging Tips

```python
def debug_quantization():
    # Check quantization ranges
    # Monitor weight distributions
    # Validate layer-wise quantization
    # Test dequantization accuracy
    pass
```

## Advanced Techniques

### Mixed Precision Quantization

```yaml
# Different precision for different layers
quantization_config: {
  "attention": {"bits": 4, "type": "awq"},
  "mlp": {"bits": 8, "type": "awq"},
  "embedding": {"bits": 8, "type": "awq"}
}
```

### Dynamic Quantization

```yaml
# Adapt quantization during training
dynamic_quantization: true
quantization_update_freq: 100
quantization_momentum: 0.9
```

### Quantization-aware Fine-tuning

```yaml
# Train with quantization awareness
quantization_aware_training: true
quantization_noise: 0.01
quantization_temperature: 1.0
```

## Deployment Considerations

### Model Loading

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "saves/llama3-8b/awq/lora/sft")
```

### Inference Optimization

```python
# Optimize for inference speed
torch.compile(model)  # Use torch.compile if available
model.eval()  # Set to evaluation mode
```

### Batch Processing

```python
# Efficient batch processing with quantization
def batch_inference(model, texts):
    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs, max_length=200)
    return outputs
```

## Comparison with Other Methods

| Aspect | AWQ | GPTQ | AQLM | OTFQ | QLoRA |
|--------|-----|------|------|------|-------|
| Memory Usage | Low | Low | Very Low | Low | Low |
| Quality | Excellent | Very Good | Good | Excellent | Good |
| Speed | Fast | Fast | Very Fast | Fast | Fast |
| Flexibility | High | High | Medium | High | High |
| Hardware Support | Good | Good | Limited | Good | Good |

## Next Steps

- Experiment with different quantization methods
- Try mixed precision quantization
- Use quantization for model deployment
- Combine quantization with other optimizations
- Monitor quantization effects on model behavior

For hands-on examples, see the [notebooks](../notebooks/quantization/) directory.
