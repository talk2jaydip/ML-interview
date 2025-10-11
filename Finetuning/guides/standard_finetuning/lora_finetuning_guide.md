# LoRA Fine-tuning Guide

## Overview

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that adds small, trainable adapter matrices to the frozen pre-trained model, significantly reducing memory requirements while maintaining performance.

## How LoRA Works

LoRA works by:
1. **Freezing** the original model parameters
2. **Adding** low-rank adapter matrices to attention layers
3. **Training** only the adapter matrices (~0.1-1% of parameters)
4. **Merging** adapters during inference for full-speed execution

### Mathematical Foundation

For a weight matrix W ∈ ℝ^(d×k), LoRA approximates:
```
W = W₀ + ΔW = W₀ + BA
```
Where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), and r is the rank (typically 8-64).

## Advantages of LoRA

- **Memory Efficient**: 70-90% reduction in trainable parameters
- **Fast Training**: Fewer parameters to optimize
- **Modular**: Multiple LoRA adapters for different tasks
- **Deployment Friendly**: Small adapter files
- **Stable Training**: Less prone to catastrophic forgetting

## When to Use LoRA

- **Large models** (13B+ parameters)
- **Limited GPU memory** (8GB+)
- **Multiple task adaptation**
- **Quick experimentation**
- **Consumer hardware**

## LoRA Configuration Parameters

### Core Parameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|---------|
| `lora_rank` | Rank of LoRA matrices | 8, 16, 32, 64 | Higher = more parameters, better quality |
| `lora_alpha` | Scaling parameter | 16, 32, 64 | Higher = stronger updates |
| `lora_dropout` | Dropout probability | 0.0, 0.1, 0.2 | Prevents overfitting |
| `lora_target` | Target modules | all, q_proj, k_proj, v_proj, o_proj | Which layers to adapt |

### Target Module Options

```yaml
# Apply LoRA to all linear layers
lora_target: all

# Apply LoRA only to attention layers
lora_target: q_proj,k_proj,v_proj,o_proj

# Apply LoRA to MLP layers
lora_target: gate_proj,up_proj,down_proj

# Custom target modules
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
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

## Advanced LoRA Variants

### LoRA+ (LoRA Plus)

LoRA+ improves LoRA by using different learning rates for different matrix components.

```yaml
### LoRA+ configuration
finetuning_type: lora
lora_rank: 8
lora_alpha: 32
lora_dropout: 0.1
lora_target: all
loraplus_lr_ratio: 16  # Different learning rate for A and B matrices
```

### AdaLoRA

Adaptive LoRA that dynamically allocates rank budget.

```yaml
### AdaLoRA configuration (requires extras/adam_mini)
finetuning_type: lora
lora_rank: 8
lora_alpha: 32
lora_dropout: 0.1
lora_target: all
use_adalora: true
adalora_init_r: 8
adalora_target_r: 64
adalora_tinit: 200
adalora_tfinal: 1000
adalora_delta_t: 10
```

## Hardware Requirements

### Minimum Requirements
- **GPU Memory**: 8GB for 7B models, 16GB for 13B models
- **System RAM**: 32GB
- **Storage**: 50GB for models and datasets

### Resource Usage by Model Size

| Model | GPU Memory | Trainable Parameters | Storage | Training Time |
|-------|------------|---------------------|---------|---------------|
| 7B | 8-12GB | ~4-8M | ~50GB | 1-2 hours |
| 13B | 16-24GB | ~8-16M | ~100GB | 2-4 hours |
| 70B | 48-80GB | ~40-80M | ~400GB | 4-8 hours |

## Training Scripts

### Basic LoRA Training
```bash
python src/train.py examples/train_lora/llama3_lora_sft.yaml
```

### LoRA with Custom Parameters
```bash
python src/train.py examples/train_lora/llama3_lora_sft.yaml \
  --lora_rank 16 \
  --lora_alpha 64 \
  --learning_rate 2.0e-4
```

### Multi-GPU LoRA Training
```bash
torchrun --nproc_per_node=2 src/train.py examples/train_lora/llama3_lora_sft.yaml
```

## Model Loading and Inference

### Loading LoRA Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
lora_path = "saves/llama3-8b/lora/sft"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_path)

# Generate text
inputs = tokenizer("Explain machine learning", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Using LLaMA-Factory Chat Interface
```python
from llamafactory import ChatModel

model = ChatModel(dict(
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    adapter_path="saves/llama3-8b/lora/sft",
    finetuning_type="lora",
    template="llama3"
))

response = model.chat("What is the capital of France?")
print(response)
```

## LoRA Hyperparameter Tuning

### Rank Selection

```yaml
# Low rank (faster, lower quality)
lora_rank: 8
lora_alpha: 16

# Medium rank (balanced)
lora_rank: 16
lora_alpha: 32

# High rank (slower, higher quality)
lora_rank: 64
lora_alpha: 128
```

### Learning Rate Tuning

```yaml
# Conservative learning rate
learning_rate: 5.0e-5
warmup_ratio: 0.2

# Standard learning rate
learning_rate: 1.0e-4
warmup_ratio: 0.1

# Aggressive learning rate
learning_rate: 3.0e-4
warmup_ratio: 0.05
```

### Target Module Selection

```yaml
# Attention only (memory efficient)
lora_target: q_proj,k_proj,v_proj,o_proj

# MLP only (different adaptation)
lora_target: gate_proj,up_proj,down_proj

# All layers (comprehensive adaptation)
lora_target: all
```

## Memory Optimization

### For Limited Memory
```yaml
# Reduce memory usage
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
lora_rank: 8
lora_target: q_proj,k_proj,v_proj,o_proj
gradient_checkpointing: true
```

### For Faster Training
```yaml
# Optimize for speed
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
lora_rank: 16
bf16: true
tf32: true
```

## LoRA Adapter Management

### Saving Multiple Adapters
```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, config)

# Train for task A
# ... training code ...

# Save adapter A
model.save_pretrained("adapters/task_a")

# Train for task B with new adapter
model = get_peft_model(model, LoraConfig(...))
# ... training code ...
model.save_pretrained("adapters/task_b")
```

### Merging LoRA Adapters
```bash
# Merge LoRA weights into base model
python scripts/merge_lora.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --lora_path saves/llama3-8b/lora/sft \
  --output_path saves/llama3-8b/merged
```

## Best Practices

### 1. Start with Standard Settings
```yaml
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target: all
learning_rate: 1.0e-4
```

### 2. Monitor Training
- Track LoRA adapter weights
- Monitor gradient norms
- Check validation performance

### 3. Regularization
```yaml
# Add regularization for better generalization
lora_dropout: 0.1
weight_decay: 0.01
```

### 4. Learning Rate Scheduling
```yaml
lr_scheduler_type: cosine  # Better than linear for LoRA
warmup_ratio: 0.1
```

### 5. Batch Size Optimization
```yaml
# Effective batch size = per_device * accumulation * num_gpus
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
# Effective batch size: 8
```

## Troubleshooting

### Common Issues

1. **Poor Performance**
   - Increase LoRA rank
   - Add more target modules
   - Use longer training

2. **Memory Issues**
   - Reduce LoRA rank
   - Use fewer target modules
   - Reduce batch size

3. **Training Instability**
   - Reduce learning rate
   - Increase warmup ratio
   - Add dropout

4. **Overfitting**
   - Increase dropout
   - Use early stopping
   - Add regularization

## Performance Comparison

| Method | Memory Usage | Training Speed | Model Quality | Adapter Size |
|--------|--------------|----------------|---------------|--------------|
| Full Fine-tuning | High | Slow | High | Full model |
| LoRA (r=8) | Low | Fast | Good | ~4-8MB |
| LoRA (r=16) | Medium | Medium | Very Good | ~8-16MB |
| LoRA (r=64) | High | Slow | Excellent | ~32-64MB |

## Advanced Techniques

### LoRA with Quantization (QLoRA)
```yaml
### QLoRA configuration
finetuning_type: lora
quantization_bit: 4
quantization_type: nf4
double_quantization: true
lora_rank: 64
lora_alpha: 128
```

### LoRA with DeepSpeed
```yaml
# Use DeepSpeed configuration
deepspeed: examples/deepspeed/ds_z3_config.json
```

### LoRA with FSDP
```yaml
# Use FSDP configuration
fsdp: examples/accelerate/fsdp_config.yaml
```

## Next Steps

- Explore QLoRA for even more memory efficiency
- Try advanced LoRA variants like LoRA+
- Experiment with different target modules
- Use LoRA for multi-task learning
- Deploy LoRA adapters in production

For hands-on examples, see the [notebooks](../../notebooks/sft/) directory.
