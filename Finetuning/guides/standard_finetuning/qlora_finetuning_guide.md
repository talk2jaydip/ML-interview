# QLoRA Fine-tuning Guide

## Overview

QLoRA (Quantized Low-Rank Adaptation) combines 4-bit quantization with LoRA to enable fine-tuning of very large language models on consumer hardware while maintaining performance.

## How QLoRA Works

QLoRA combines three key techniques:
1. **4-bit NormalFloat Quantization**: Compresses model weights to 4 bits
2. **Double Quantization**: Quantizes the quantization constants
3. **Paged Optimizers**: Manages memory spikes during training
4. **LoRA**: Parameter-efficient fine-tuning

### Quantization Details

- **NF4**: 4-bit NormalFloat quantization optimized for normally distributed weights
- **Double Quantization**: Quantizes quantization constants for additional memory savings
- **Paged Optimizers**: Prevents memory fragmentation during training

## Advantages of QLoRA

- **Memory Efficiency**: 75-90% reduction in memory usage
- **Large Model Support**: Fine-tune 70B+ models on 24GB GPUs
- **Performance Preservation**: Maintains model quality
- **Fast Training**: Efficient training on limited hardware
- **Consumer Hardware**: Works on RTX 30/40 series GPUs

## When to Use QLoRA

- **Very large models** (30B+ parameters)
- **Limited GPU memory** (4-24GB)
- **Consumer hardware** (RTX 3090/4090)
- **Multiple model experimentation**
- **Memory-constrained environments**

## QLoRA Configuration Parameters

### Quantization Parameters

| Parameter | Description | Options | Impact |
|-----------|-------------|---------|---------|
| `quantization_bit` | Quantization precision | 4, 8 | 4-bit for maximum compression |
| `quantization_type` | Quantization format | nf4, fp4, int4 | nf4 for best performance |
| `double_quantization` | Nested quantization | true, false | true for more memory savings |
| `quantization_method` | Backend implementation | bitsandbytes, hqq, eetq | bitsandbytes for compatibility |

### LoRA Parameters for QLoRA

```yaml
lora_rank: 64  # Higher rank for QLoRA
lora_alpha: 128  # Higher alpha for QLoRA
lora_dropout: 0.1
lora_target: all
```

## Basic Configuration

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

## Hardware Requirements

### Minimum Requirements
- **GPU Memory**: 4GB for 7B models, 12GB for 13B models, 24GB for 70B models
- **System RAM**: 16GB (model weights loaded on GPU)
- **Storage**: 30GB for models and datasets
- **GPU**: RTX 3060 (12GB), RTX 3090 (24GB), RTX 4090 (24GB)

### Resource Usage by Model Size

| Model | GPU Memory | CPU RAM | Trainable Parameters | Training Time |
|-------|------------|---------|---------------------|---------------|
| 7B | 4-8GB | 8GB | ~4-8M | 1-3 hours |
| 13B | 8-12GB | 12GB | ~8-16M | 2-4 hours |
| 30B | 16-24GB | 16GB | ~16-32M | 4-6 hours |
| 70B | 24-40GB | 24GB | ~40-80M | 6-12 hours |

## Training Scripts

### Basic QLoRA Training
```bash
python src/train.py examples/train_qlora/llama3_lora_sft_gptq.yaml
```

### QLoRA with Custom Parameters
```bash
python src/train.py examples/train_qlora/llama3_lora_sft_gptq.yaml \
  --lora_rank 64 \
  --lora_alpha 128 \
  --learning_rate 2.0e-4
```

### Multi-GPU QLoRA Training
```bash
torchrun --nproc_per_node=2 src/train.py examples/train_qlora/llama3_lora_sft_gptq.yaml
```

## Model Loading and Inference

### Loading QLoRA Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load base model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "saves/llama3-8b/qlora/sft")

# Generate text
inputs = tokenizer("Explain quantum computing", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Using LLaMA-Factory Chat Interface
```python
from llamafactory import ChatModel

model = ChatModel(dict(
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    adapter_path="saves/llama3-8b/qlora/sft",
    finetuning_type="lora",
    quantization_bit=4,
    quantization_type="nf4",
    template="llama3"
))

response = model.chat("What are the benefits of renewable energy?")
print(response)
```

## Advanced QLoRA Variants

### QAT-LoRA (Quantization-Aware Training)
```yaml
### QAT-LoRA configuration
finetuning_type: lora
quantization_bit: 4
quantization_type: nf4
double_quantization: true
use_qat: true  # Quantization-aware training
qat_dropout: 0.1
```

### Paged QLoRA
```yaml
# Automatic paged optimizers
optim: paged_adamw_32bit  # Prevents memory fragmentation
```

### QLoRA with Flash Attention
```yaml
use_flash_attn: true  # Requires flash-attn package
```

## Memory Optimization

### For Very Limited Memory (4-8GB GPUs)
```yaml
### Ultra-low memory configuration
quantization_bit: 4
quantization_type: nf4
double_quantization: true
lora_rank: 32
lora_target: q_proj,k_proj,v_proj,o_proj  # Only attention layers
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
```

### For Better Performance (12-24GB GPUs)
```yaml
### Balanced configuration
quantization_bit: 4
quantization_type: nf4
double_quantization: true
lora_rank: 64
lora_target: all
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
bf16: true
```

## Troubleshooting Common Issues

### 1. Installation Issues
```bash
# Install bitsandbytes for QLoRA
pip install bitsandbytes
# For CUDA 12.x
pip install bitsandbytes --upgrade
```

### 2. Memory Issues During Training
```yaml
# Reduce memory usage
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
lora_rank: 32
# Use CPU offloading
cpu_offload: true
```

### 3. Slow Training
```yaml
# Optimize for speed
bf16: true
tf32: true
dataloader_num_workers: 4
preprocessing_num_workers: 8
```

### 4. Quantization Errors
```yaml
# Use compatible settings
quantization_type: nf4  # Most stable
double_quantization: false  # If having issues
quantization_method: bitsandbytes  # Most compatible
```

## Performance Benchmarks

### Memory Usage Comparison

| Method | 7B Model | 13B Model | 70B Model |
|--------|----------|-----------|-----------|
| Full Fine-tuning | 24GB | 48GB | 120GB+ |
| LoRA | 8GB | 16GB | 48GB |
| QLoRA | 4GB | 8GB | 24GB |

### Training Speed Comparison

| Method | 7B Model | 13B Model | 70B Model |
|--------|----------|-----------|-----------|
| Full Fine-tuning | 100% | 100% | 100% |
| LoRA | 150% | 140% | 130% |
| QLoRA | 120% | 110% | 100% |

### Model Quality Comparison

| Method | MMLU Score | GSM8K Score | HumanEval Score |
|--------|------------|-------------|-----------------|
| Full Fine-tuning | 0.65 | 0.52 | 0.28 |
| LoRA | 0.63 | 0.50 | 0.26 |
| QLoRA | 0.62 | 0.49 | 0.25 |

## Best Practices

### 1. Start with Standard QLoRA
```yaml
quantization_bit: 4
quantization_type: nf4
double_quantization: true
lora_rank: 64
lora_alpha: 128
```

### 2. Use Higher LoRA Rank
```yaml
# QLoRA benefits from higher rank
lora_rank: 64  # Instead of 8-16
lora_alpha: 128  # Scale accordingly
```

### 3. Monitor Quantization Effects
- Check for performance degradation
- Use validation sets to ensure quality
- Compare with LoRA baseline

### 4. Hardware-Specific Optimization
```yaml
# For RTX 30/40 series
quantization_method: bitsandbytes
bnb_4bit_compute_dtype: bfloat16

# For A100/H100
quantization_method: bitsandbytes
bnb_4bit_compute_dtype: float16
```

## Deployment Considerations

### Merging QLoRA Adapters
```bash
# Merge QLoRA weights (quantized base + LoRA)
python scripts/merge_lora.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --lora_path saves/llama3-8b/qlora/sft \
  --output_path saves/llama3-8b/qlora_merged \
  --quantization_bit 4
```

### Quantized Model Loading
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "saves/llama3-8b/qlora_merged",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Advanced Techniques

### QLoRA with DeepSpeed
```yaml
# Use DeepSpeed ZeRO-3 for very large models
deepspeed: examples/deepspeed/ds_z3_config.json
```

### QLoRA with FSDP
```yaml
# Use FSDP for multi-GPU training
fsdp: examples/accelerate/fsdp_config.yaml
```

### QLoRA with Gradient Checkpointing
```yaml
gradient_checkpointing: true
use_reentrant: false  # More stable
```

## Comparison with Other Methods

| Aspect | QLoRA | LoRA | Full Fine-tuning |
|--------|-------|------|------------------|
| Memory Usage | Very Low | Low | High |
| Max Model Size | 70B+ | 30B+ | 13B+ |
| Training Speed | Fast | Fast | Slow |
| Hardware Required | 4GB+ GPU | 8GB+ GPU | 24GB+ GPU |
| Model Quality | Good | Very Good | Excellent |
| Deployment Size | Small | Small | Large |

## Next Steps

- Explore advanced quantization methods (HQQ, EETQ)
- Try QLoRA with different model architectures
- Experiment with QAT-LoRA for better performance
- Use QLoRA for fine-tuning vision-language models
- Deploy QLoRA models in production environments

For hands-on examples, see the [notebooks](../../notebooks/sft/) directory.
