# LoRA+ (LoRA Plus) Guide

## Overview

LoRA+ (LoRA Plus) improves upon standard LoRA by using different learning rates for the A and B matrices, leading to better training dynamics and improved performance.

## How LoRA+ Works

### Core Algorithm

1. **Asymmetric Learning Rates**: Use different learning rates for A and B matrices
2. **Matrix Decomposition**: A has learning rate η, B has learning rate η/ratio
3. **Better Convergence**: Improved convergence properties through asymmetric updates
4. **Dropout Integration**: Better regularization through controlled updates

### Mathematical Foundation

For LoRA update ΔW = BA:
- Matrix A ∈ ℝ^(r×k): Learning rate η_A = η
- Matrix B ∈ ℝ^(d×r): Learning rate η_B = η / ratio
- Update rule: A ← A - η_A ∇_A L, B ← B - η_B ∇_B L

## Advantages of LoRA+

- **Better Convergence**: Improved training dynamics
- **Higher Quality**: Better final model performance
- **Robust Training**: More stable than standard LoRA
- **Dropout Benefits**: Better regularization properties
- **Easy Integration**: Simple modification to standard LoRA

## When to Use LoRA+

- **Quality is critical** and you want better performance than standard LoRA
- **Stable training** is needed for difficult datasets
- **Research** comparing different LoRA variants
- **Production models** where quality matters most
- **Fine-tuning** tasks where standard LoRA underperforms

## Configuration Parameters

### Basic Configuration

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
loraplus_lr_ratio: 16  # Key parameter for LoRA+

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

### LoRA+ Parameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|---------|
| `loraplus_lr_ratio` | Learning rate ratio between A and B | 8, 16, 32 | Higher = slower B updates |
| `lora_rank` | Rank of LoRA matrices | 8, 16, 32, 64 | Higher = more parameters |
| `lora_alpha` | Scaling parameter | 16, 32, 64 | Should scale with rank |

## Hardware Requirements

### Minimum Requirements
- **GPU Memory**: 8GB for 7B models, 16GB for 13B models
- **System RAM**: 32GB
- **Storage**: 30GB for models and datasets
- **GPU**: Any modern GPU with sufficient memory

### Performance by Model Size

| Model Size | GPU Memory | Training Time | Quality Improvement |
|------------|------------|---------------|-------------------|
| 7B | 8-12GB | 1-2 hours | +2-5% |
| 13B | 16-24GB | 2-4 hours | +2-4% |
| 30B | 32-48GB | 4-6 hours | +1-3% |
| 70B | 48-80GB | 6-10 hours | +1-2% |

## Training Scripts

### Basic LoRA+ Training
```bash
python src/train.py examples/extras/loraplus/llama3_lora_sft.yaml
```

### LoRA+ with Custom Parameters
```bash
python src/train.py examples/extras/loraplus/llama3_lora_sft.yaml \
  --loraplus_lr_ratio 32 \
  --lora_rank 16 \
  --learning_rate 2.0e-4
```

### LoRA+ for Different Tasks
```bash
# For instruction tuning
python src/train.py examples/extras/loraplus/llama3_lora_sft.yaml

# For math tasks (needs more stability)
python src/train.py examples/extras/loraplus/llama3_lora_sft.yaml \
  --loraplus_lr_ratio 64 \
  --learning_rate 5.0e-5
```

## Model Loading and Inference

### Loading LoRA+ Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
loraplus_model_path = "saves/llama3-8b/loraplus/sft"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, loraplus_model_path)

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
    adapter_path="saves/llama3-8b/loraplus/sft",
    finetuning_type="lora",
    template="llama3"
))

response = model.chat("What are the benefits of renewable energy?")
print(response)
```

## Hyperparameter Tuning

### Learning Rate Ratio Selection

```yaml
# Conservative (more stable)
loraplus_lr_ratio: 32

# Standard (balanced)
loraplus_lr_ratio: 16

# Aggressive (faster convergence)
loraplus_lr_ratio: 8
```

### Rank and Alpha Selection

```yaml
# Small rank (efficient)
lora_rank: 8
lora_alpha: 16

# Medium rank (balanced)
lora_rank: 16
lora_alpha: 32

# Large rank (high quality)
lora_rank: 32
lora_alpha: 64
```

### Learning Rate Tuning

```yaml
# Conservative
learning_rate: 5.0e-5
warmup_ratio: 0.2

# Standard
learning_rate: 1.0e-4
warmup_ratio: 0.1

# Higher
learning_rate: 2.0e-4
warmup_ratio: 0.05
```

## Advanced LoRA+ Configurations

### LoRA+ with QLoRA
```yaml
### QLoRA + LoRA+ configuration
finetuning_type: lora
quantization_bit: 4
quantization_type: nf4
double_quantization: true
lora_rank: 64
lora_alpha: 128
loraplus_lr_ratio: 32
```

### LoRA+ with DeepSpeed
```yaml
# Use DeepSpeed with LoRA+
deepspeed: examples/deepspeed/ds_z3_config.json
loraplus_lr_ratio: 16
```

### LoRA+ with Custom Target Modules

```yaml
# Target specific modules
lora_target: q_proj,k_proj,v_proj,o_proj
loraplus_lr_ratio: 16

# Target MLP modules
lora_target: gate_proj,up_proj,down_proj
loraplus_lr_ratio: 32  # Higher ratio for MLP
```

## Evaluation

### LoRA+ Model Evaluation
```python
from llamafactory.eval import evaluate_loraplus_model

results = evaluate_loraplus_model(
    model_path="saves/llama3-8b/loraplus/sft",
    eval_dataset="alpaca_en_demo",
    metrics=["perplexity", "accuracy", "stability"]
)
```

### Manual Evaluation
```python
def evaluate_loraplus_model(model, test_cases):
    results = []
    for prompt in test_cases:
        response = model.chat(prompt)
        # Evaluate response quality
        # Compare with standard LoRA
        results.append(evaluate_response(response))
    return results
```

### Metrics to Track

- **Perplexity**: Language modeling quality
- **Task Accuracy**: Performance on specific tasks
- **Stability**: Training stability metrics
- **Convergence Speed**: How quickly the model converges
- **Final Performance**: Comparison with standard LoRA

## Best Practices

### 1. Start with Standard Settings
```yaml
lora_rank: 16
lora_alpha: 32
loraplus_lr_ratio: 16
learning_rate: 1.0e-4
```

### 2. Quality vs Speed Trade-off

```yaml
# Maximum quality
lora_rank: 32
lora_alpha: 64
loraplus_lr_ratio: 32
learning_rate: 5.0e-5

# Balanced
lora_rank: 16
lora_alpha: 32
loraplus_lr_ratio: 16
learning_rate: 1.0e-4

# Maximum speed
lora_rank: 8
lora_alpha: 16
loraplus_lr_ratio: 8
learning_rate: 2.0e-4
```

### 3. Training Stability

```yaml
# Conservative settings for LoRA+
learning_rate: 5.0e-5
warmup_ratio: 0.2
gradient_checkpointing: true
loraplus_lr_ratio: 32
```

### 4. Hardware Optimization

```yaml
# For RTX 30/40 series
lora_rank: 16
loraplus_lr_ratio: 16

# For A100/H100
lora_rank: 32
loraplus_lr_ratio: 32
```

### 5. Monitoring

```python
def monitor_loraplus_training():
    # Track A and B matrix norms
    # Monitor learning rate ratios
    # Check convergence curves
    # Validate parameter updates
    pass
```

## Troubleshooting

### Common Issues

1. **Training Instability**
   - Increase loraplus_lr_ratio
   - Reduce learning rate
   - Add gradient clipping

2. **Poor Performance**
   - Decrease loraplus_lr_ratio
   - Increase lora_rank
   - Try longer training

3. **Slow Convergence**
   - Decrease loraplus_lr_ratio
   - Increase learning rate
   - Use different warmup schedule

4. **Memory Issues**
   - Reduce lora_rank
   - Increase loraplus_lr_ratio
   - Use QLoRA

### Debugging Tips

```python
def debug_loraplus_training():
    # Check A and B matrix gradients
    # Monitor parameter update ratios
    # Validate learning rate scheduling
    # Test matrix condition numbers
    pass
```

## Performance Benchmarks

### Quality Comparison

| Model | LoRA | LoRA+ | Improvement | Training Time |
|-------|------|-------|-------------|---------------|
| 7B | 0.62 | 0.64 | +3.2% | +5% |
| 13B | 0.65 | 0.67 | +3.1% | +8% |
| 30B | 0.68 | 0.70 | +2.9% | +10% |

### Efficiency Comparison

| Method | Memory | Training Speed | Quality | Stability |
|--------|--------|----------------|---------|-----------|
| LoRA | 8GB | 100% | 100% | 95% |
| LoRA+ | 8GB | 98% | 103% | 98% |
| LoRA+ (r=32) | 12GB | 85% | 105% | 97% |

## Advanced Techniques

### LoRA+ with Adaptive Ratios

```yaml
# Dynamic learning rate ratios
loraplus_adaptive: true
loraplus_adaptation_freq: 100
loraplus_target_ratio: 16
```

### LoRA+ with Layer-wise Ratios

```yaml
# Different ratios for different layers
loraplus_layer_ratios: {
  "q_proj": 16,
  "k_proj": 16,
  "v_proj": 16,
  "o_proj": 32,
  "gate_proj": 8,
  "up_proj": 8,
  "down_proj": 8
}
```

### LoRA+ with Momentum

```yaml
# Add momentum to LoRA+ updates
loraplus_momentum: 0.9
loraplus_momentum_type: nesterov
```

## Deployment Considerations

### Model Serving
```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load LoRA+ model
generator = pipeline(
    "text-generation",
    model="saves/llama3-8b/loraplus/sft",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

@app.post("/generate")
def generate_text(prompt: str):
    response = generator(prompt, max_length=200)
    return {"response": response[0]["generated_text"]}
```

### Model Comparison

```python
def compare_lora_variants():
    lora_model = load_lora_model()
    loraplus_model = load_loraplus_model()

    results = []
    for prompt in test_prompts:
        lora_response = lora_model.chat(prompt)
        loraplus_response = loraplus_model.chat(prompt)

        # Compare responses
        results.append(compare_responses(lora_response, loraplus_response))

    return results
```

### Production Optimization

```python
# Optimize LoRA+ for production
def optimize_for_production(model):
    # Merge LoRA+ weights if needed
    # Quantize model if possible
    # Optimize inference
    return optimized_model
```

## Comparison with Other Methods

| Aspect | LoRA+ | Standard LoRA | OFT | PISSA |
|--------|-------|---------------|-----|-------|
| Quality | Better | Good | Good | Best |
| Speed | Similar | Similar | Slower | Slower |
| Memory | Same | Same | Same | Same |
| Ease of Use | High | High | Medium | Low |
| Stability | Better | Good | Good | Good |

## Next Steps

- Experiment with different LoRA+ ratios
- Try LoRA+ with quantization
- Use LoRA+ for specialized tasks
- Combine LoRA+ with other techniques
- Research LoRA+ theoretical properties

For hands-on examples, see the [notebooks](../../notebooks/advanced_methods/) directory.
