# OFT (Orthogonal Fine-tuning) Guide

## Overview

OFT (Orthogonal Fine-tuning) uses orthogonal matrix decomposition to maintain orthogonality constraints during fine-tuning, leading to better parameter efficiency and training stability.

## How OFT Works

### Core Algorithm

1. **Orthogonal Initialization**: Initialize adapter matrices as orthogonal
2. **Orthogonal Updates**: Maintain orthogonality during training
3. **Block-wise Decomposition**: Use block-wise orthogonal updates
4. **Stable Training**: Better numerical stability through orthogonality
5. **Efficient Representation**: More efficient parameter usage

### Mathematical Foundation

For orthogonal matrix O ∈ ℝ^(d×d):
1. Decompose as O = Q S where Q is orthogonal, S is block-diagonal
2. Update only S during training: S ← S + ΔS
3. Reconstruct O = Q (S + ΔS)
4. Maintain orthogonality: O^T O = I

## Advantages of OFT

- **Parameter Efficiency**: More efficient than LoRA for same rank
- **Training Stability**: Better numerical stability
- **Orthogonality Benefits**: Better representation learning
- **Convergence**: Faster convergence in many cases
- **Quality**: Competitive or better quality than LoRA

## When to Use OFT

- **Parameter efficiency** is critical
- **Numerical stability** is important
- **Research** with orthogonal constraints
- **Vision-language models** (works well with visual features)
- **Tasks** requiring stable training dynamics

## Configuration Parameters

### Basic Configuration

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: oft
lora_rank: 8  # Block size for OFT
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

### OFT-Specific Parameters

| Parameter | Description | Values | Impact |
|-----------|-------------|--------|---------|
| `lora_rank` | Block size for orthogonal updates | 4, 8, 16, 32 | Smaller = more efficient |
| `lora_alpha` | Scaling parameter | 16, 32, 64 | Should scale with block size |
| `oft_blocks` | Number of orthogonal blocks | 1, 2, 4 | More = better quality, less efficient |
| `oft_block_share` | Share blocks across layers | true/false | true = more efficient |

## Hardware Requirements

### Minimum Requirements
- **GPU Memory**: 8GB for 7B models, 16GB for 13B models
- **System RAM**: 32GB
- **Storage**: 30GB for models and datasets
- **GPU**: Any modern GPU with sufficient memory

### Performance by Model Size

| Model Size | GPU Memory | Training Time | Parameter Efficiency |
|------------|------------|---------------|-------------------|
| 7B | 8-12GB | 1.5-2.5 hours | 2x better than LoRA |
| 13B | 16-24GB | 2.5-4 hours | 1.8x better than LoRA |
| 30B | 32-48GB | 5-8 hours | 1.5x better than LoRA |
| 70B | 48-80GB | 8-12 hours | 1.3x better than LoRA |

## Training Scripts

### Basic OFT Training
```bash
python src/train.py examples/extras/oft/llama3_oft_sft.yaml
```

### OFT with Custom Parameters
```bash
python src/train.py examples/extras/oft/llama3_oft_sft.yaml \
  --lora_rank 16 \
  --lora_alpha 64 \
  --learning_rate 2.0e-4
```

### OFT for Vision-Language Models
```bash
python src/train.py examples/extras/oft/qwen2_5vl_oft_sft.yaml
```

## Model Loading and Inference

### Loading OFT Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
oft_model_path = "saves/llama3-8b/oft/sft"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, oft_model_path)

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
    adapter_path="saves/llama3-8b/oft/sft",
    finetuning_type="oft",
    template="llama3"
))

response = model.chat("What are the benefits of renewable energy?")
print(response)
```

## Hyperparameter Tuning

### Block Size Selection

```yaml
# Small blocks (more efficient)
lora_rank: 8
lora_alpha: 32

# Medium blocks (balanced)
lora_rank: 16
lora_alpha: 64

# Large blocks (higher quality)
lora_rank: 32
lora_alpha: 128
```

### Number of Blocks

```yaml
# Single block per matrix
oft_blocks: 1

# Multiple blocks per matrix
oft_blocks: 2
lora_rank: 16  # Total rank = blocks * lora_rank

# More blocks for better quality
oft_blocks: 4
lora_rank: 8
```

### Block Sharing

```yaml
# Share blocks across layers (more efficient)
oft_block_share: true
lora_rank: 8

# Independent blocks per layer (better quality)
oft_block_share: false
lora_rank: 16
```

## Advanced OFT Configurations

### OFT with QLoRA
```yaml
### QLoRA + OFT configuration
finetuning_type: oft
quantization_bit: 4
quantization_type: nf4
double_quantization: true
lora_rank: 16
lora_alpha: 64
```

### OFT with DeepSpeed
```yaml
# Use DeepSpeed with OFT
deepspeed: examples/deepspeed/ds_z3_config.json
finetuning_type: oft
lora_rank: 16
```

### OFT with Custom Target Modules

```yaml
# Target specific modules
lora_target: q_proj,k_proj,v_proj,o_proj
finetuning_type: oft

# Target MLP modules
lora_target: gate_proj,up_proj,down_proj
finetuning_type: oft
```

## Evaluation

### OFT Model Evaluation
```python
from llamafactory.eval import evaluate_oft_model

results = evaluate_oft_model(
    model_path="saves/llama3-8b/oft/sft",
    eval_dataset="alpaca_en_demo",
    metrics=["perplexity", "orthogonality", "efficiency"]
)
```

### Manual Evaluation
```python
def evaluate_oft_model(model, test_cases):
    results = []
    for prompt in test_cases:
        response = model.chat(prompt)
        # Evaluate response quality
        # Check orthogonality preservation
        results.append(evaluate_response(response))
    return results
```

### Metrics to Track

- **Orthogonality Error**: Deviation from perfect orthogonality
- **Perplexity**: Language modeling quality
- **Task Accuracy**: Performance on specific tasks
- **Parameter Efficiency**: Parameters vs performance trade-off
- **Training Stability**: Stability metrics

## Best Practices

### 1. Start with Conservative Settings
```yaml
lora_rank: 16
lora_alpha: 64
oft_blocks: 1
oft_block_share: false
learning_rate: 1.0e-4
```

### 2. Quality vs Efficiency Trade-off

```yaml
# Maximum quality
lora_rank: 32
lora_alpha: 128
oft_blocks: 2
oft_block_share: false

# Balanced
lora_rank: 16
lora_alpha: 64
oft_blocks: 1
oft_block_share: false

# Maximum efficiency
lora_rank: 8
lora_alpha: 32
oft_blocks: 1
oft_block_share: true
```

### 3. Training Stability

```yaml
# Conservative settings for OFT
learning_rate: 5.0e-5
warmup_ratio: 0.2
gradient_checkpointing: true
```

### 4. Hardware Optimization

```yaml
# For RTX 30/40 series
lora_rank: 16
oft_blocks: 1

# For A100/H100
lora_rank: 32
oft_blocks: 2
```

### 5. Monitoring

```python
def monitor_oft_training():
    # Track orthogonality metrics
    # Monitor block-wise updates
    # Check convergence patterns
    # Validate matrix properties
    pass
```

## Troubleshooting

### Common Issues

1. **Orthogonality Loss**
   - Reduce learning rate
   - Increase warmup steps
   - Use smaller block sizes

2. **Training Instability**
   - Use conservative learning rates
   - Add gradient clipping
   - Reduce block sizes

3. **Poor Performance**
   - Increase number of blocks
   - Use larger block sizes
   - Try longer training

4. **Memory Issues**
   - Reduce number of blocks
   - Use block sharing
   - Reduce batch sizes

### Debugging Tips

```python
def debug_oft_training():
    # Check orthogonality constraints
    # Monitor block matrix properties
    # Validate update mechanisms
    # Test numerical stability
    pass
```

## Performance Benchmarks

### Efficiency Comparison

| Method | Parameters | Memory | Quality | Orthogonality |
|--------|------------|--------|---------|---------------|
| LoRA (r=16) | 8M | 8GB | 0.62 | N/A |
| OFT (r=16) | 4M | 8GB | 0.63 | 0.98 |
| OFT (r=8) | 2M | 8GB | 0.61 | 0.99 |
| OFT (r=4) | 1M | 8GB | 0.59 | 0.995 |

### Quality Comparison

| Model | Method | MMLU | GSM8K | HumanEval | Orthogonality |
|-------|--------|------|-------|-----------|---------------|
| 7B | LoRA | 0.62 | 0.49 | 0.25 | N/A |
| 7B | OFT | 0.63 | 0.50 | 0.26 | 0.98 |
| 13B | LoRA | 0.65 | 0.52 | 0.27 | N/A |
| 13B | OFT | 0.66 | 0.53 | 0.28 | 0.97 |

## Advanced Techniques

### OFT with Adaptive Blocks

```yaml
# Dynamic block sizes based on layer
oft_adaptive_blocks: true
oft_min_block_size: 4
oft_max_block_size: 32
```

### OFT with Regularization

```yaml
# Orthogonality regularization
oft_ortho_reg: 0.1
oft_ortho_reg_type: cayley  # cayley, exp, householder
```

### OFT with Quantization

```yaml
# QOFT - Quantized OFT
finetuning_type: oft
quantization_bit: 4
quantization_type: qoft
```

## Deployment Considerations

### Model Serving
```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load OFT model
generator = pipeline(
    "text-generation",
    model="saves/llama3-8b/oft/sft",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

@app.post("/generate")
def generate_text(prompt: str):
    response = generator(prompt, max_length=200)
    return {"response": response[0]["generated_text"]}
```

### Orthogonality Validation

```python
def validate_orthogonality(model):
    # Check if adapter matrices are orthogonal
    # Validate training process
    # Ensure numerical stability
    pass
```

### Production Optimization

```python
# Optimize OFT for production
def optimize_for_production(model):
    # Validate orthogonality
    # Optimize inference
    # Compress if needed
    return optimized_model
```

## Comparison with Other Methods

| Aspect | OFT | LoRA | LoRA+ | PISSA |
|--------|-----|------|-------|-------|
| Efficiency | Excellent | Good | Good | Excellent |
| Quality | Good | Good | Better | Best |
| Stability | Good | Good | Better | Good |
| Complexity | Medium | Low | Low | High |
| Orthogonality | Yes | No | No | No |

## Next Steps

- Experiment with different OFT block sizes
- Try OFT with quantization (QOFT)
- Use OFT for vision-language models
- Research OFT theoretical properties
- Combine OFT with other techniques

For hands-on examples, see the [notebooks](../../notebooks/advanced_methods/) directory.
