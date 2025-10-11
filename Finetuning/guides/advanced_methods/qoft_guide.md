# QOFT (Quantized Orthogonal Fine-tuning) Guide

## Overview

QOFT (Quantized Orthogonal Fine-tuning) combines orthogonal fine-tuning with quantization to achieve maximum parameter efficiency while maintaining orthogonality constraints and model quality.

## How QOFT Works

### Core Algorithm

1. **Orthogonal Updates**: Use orthogonal matrix updates like OFT
2. **Quantized Storage**: Store updates in quantized format
3. **Block-wise Quantization**: Apply quantization per block
4. **Dynamic Precision**: Adapt precision based on importance
5. **Orthogonal Constraints**: Maintain orthogonality during quantization

### Mathematical Foundation

For orthogonal matrix O ∈ ℝ^(d×d) with quantization Q:
1. Decompose as O = Q S where Q is orthogonal, S is block-diagonal
2. Quantize S: S_q = Q(S)
3. Update: S' = S_q + ΔS where ΔS maintains quantization constraints
4. Reconstruct O' = Q S'
5. Maintain orthogonality: O'^T O' = I

## Advantages of QOFT

- **Maximum Efficiency**: Combines benefits of OFT and quantization
- **Quality Preservation**: Maintains orthogonality and model quality
- **Storage Efficient**: Very small adapter sizes
- **Fast Inference**: Quantized operations are fast
- **Stable Training**: Orthogonal constraints improve stability

## When to Use QOFT

- **Extreme efficiency** is required
- **Storage constraints** are critical
- **Edge deployment** scenarios
- **Research** with quantized orthogonal methods
- **Memory-constrained** training environments

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

### QOFT-Specific Parameters

| Parameter | Description | Values | Impact |
|-----------|-------------|--------|---------|
| `quantization_bit` | Quantization precision | 4, 8 | Lower = more compression |
| `quantization_type` | Quantization method | qoft, awq | Different quantization strategies |
| `lora_rank` | Block size for orthogonal updates | 4, 8, 16 | Smaller = more efficient |
| `qoft_block_size` | Block size for quantization | 64, 128 | Smaller = better quality |

## Hardware Requirements

### Minimum Requirements
- **GPU Memory**: 8GB for 7B models, 16GB for 13B models
- **System RAM**: 32GB
- **Storage**: 30GB for models and datasets
- **GPU**: Modern GPUs with good memory bandwidth

### Performance by Model Size

| Model Size | GPU Memory | Training Time | Compression Ratio |
|------------|------------|---------------|------------------|
| 7B | 8-12GB | 2-3 hours | 20x |
| 13B | 16-24GB | 3-5 hours | 18x |
| 30B | 32-48GB | 6-8 hours | 16x |
| 70B | 48-80GB | 10-15 hours | 14x |

## Training Scripts

### Basic QOFT Training
```bash
python src/train.py examples/extras/qoft/llama3_oft_sft_awq.yaml
```

### QOFT with Custom Parameters
```bash
python src/train.py examples/extras/qoft/llama3_oft_sft_awq.yaml \
  --quantization_bit 4 \
  --lora_rank 16 \
  --learning_rate 2.0e-4
```

### QOFT for Maximum Compression
```bash
python src/train.py examples/extras/qoft/llama3_oft_sft_gptq.yaml \
  --quantization_bit 4 \
  --lora_rank 8
```

## Model Loading and Inference

### Loading QOFT Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
qoft_model_path = "saves/llama3-8b/qoft/sft"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, qoft_model_path)

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
    adapter_path="saves/llama3-8b/qoft/sft",
    finetuning_type="oft",
    quantization_bit=4,
    quantization_type="qoft",
    template="llama3"
))

response = model.chat("What are the benefits of renewable energy?")
print(response)
```

## Hyperparameter Tuning

### Quantization Precision

```yaml
# 8-bit quantization (higher quality)
quantization_bit: 8
quantization_type: qoft

# 4-bit quantization (balanced)
quantization_bit: 4
quantization_type: qoft

# 3-bit quantization (maximum compression)
quantization_bit: 4  # May need adjustment
quantization_type: qoft
```

### Block Size Selection

```yaml
# Small blocks (better quality)
lora_rank: 8
qoft_block_size: 64

# Medium blocks (balanced)
lora_rank: 16
qoft_block_size: 128

# Large blocks (faster training)
lora_rank: 32
qoft_block_size: 256
```

### Learning Rate Tuning

```yaml
# Conservative (stable)
learning_rate: 5.0e-5
warmup_ratio: 0.2

# Standard
learning_rate: 1.0e-4
warmup_ratio: 0.1

# Higher
learning_rate: 2.0e-4
warmup_ratio: 0.05
```

## Advanced QOFT Configurations

### QOFT with Different Quantization Types

```yaml
# QOFT with AWQ
quantization_type: awq
quantization_bit: 4
finetuning_type: oft

# QOFT with GPTQ
quantization_type: gptq
quantization_bit: 4
finetuning_type: oft
```

### QOFT with Custom Target Modules

```yaml
# Target specific modules with different settings
lora_target: q_proj,k_proj,v_proj,o_proj
quantization_bit: 4
finetuning_type: oft

# Target MLP modules
lora_target: gate_proj,up_proj,down_proj
quantization_bit: 8  # Higher precision for MLP
finetuning_type: oft
```

### QOFT with Dynamic Quantization

```yaml
# Dynamic quantization during training
qoft_dynamic: true
qoft_update_freq: 100
qoft_adaptation_rate: 0.1
```

## Evaluation

### QOFT Model Evaluation
```python
from llamafactory.eval import evaluate_qoft_model

results = evaluate_qoft_model(
    model_path="saves/llama3-8b/qoft/sft",
    eval_dataset="alpaca_en_demo",
    metrics=["perplexity", "quantization_error", "orthogonality"]
)
```

### Manual Evaluation
```python
def evaluate_qoft_model(model, test_cases):
    results = []
    for prompt in test_cases:
        response = model.chat(prompt)
        # Evaluate response quality
        # Check quantization and orthogonality
        results.append(evaluate_response(response))
    return results
```

### Metrics to Track

- **Quantization Error**: Error introduced by quantization
- **Orthogonality Error**: Deviation from perfect orthogonality
- **Perplexity**: Language modeling quality
- **Task Accuracy**: Performance on specific tasks
- **Compression Ratio**: Model size reduction

## Best Practices

### 1. Start with Conservative Settings
```yaml
quantization_bit: 4
lora_rank: 16
qoft_block_size: 128
learning_rate: 1.0e-4
```

### 2. Quality vs Compression Trade-off

```yaml
# Maximum quality
quantization_bit: 8
lora_rank: 32
qoft_block_size: 64

# Balanced
quantization_bit: 4
lora_rank: 16
qoft_block_size: 128

# Maximum compression
quantization_bit: 4
lora_rank: 8
qoft_block_size: 256
```

### 3. Training Stability

```yaml
# Conservative settings for QOFT
learning_rate: 5.0e-5
warmup_ratio: 0.2
gradient_checkpointing: true
```

### 4. Hardware Optimization

```yaml
# For RTX 30/40 series
quantization_bit: 4
qoft_block_size: 128

# For A100/H100
quantization_bit: 4
qoft_block_size: 64
```

### 5. Monitoring

```python
def monitor_qoft_training():
    # Track quantization errors per layer
    # Monitor orthogonality constraints
    # Check block-wise quantization quality
    # Validate training stability
    pass
```

## Troubleshooting

### Common Issues

1. **Quantization Errors**
   - Increase quantization bit precision
   - Reduce block sizes
   - Use different quantization types

2. **Orthogonality Loss**
   - Reduce learning rate
   - Increase warmup steps
   - Use smaller block sizes

3. **Training Instability**
   - Use conservative learning rates
   - Add gradient clipping
   - Increase warmup ratio

4. **Memory Issues**
   - Reduce lora_rank
   - Increase qoft_block_size
   - Reduce batch sizes

### Debugging Tips

```python
def debug_qoft_training():
    # Check quantization ranges
    # Monitor orthogonality metrics
    # Validate block matrix properties
    # Test numerical stability
    pass
```

## Performance Benchmarks

### Efficiency Comparison

| Method | Parameters | Memory | Quality | Compression |
|--------|------------|--------|---------|-------------|
| LoRA (r=16) | 8M | 8GB | 0.62 | 1x |
| OFT (r=16) | 4M | 8GB | 0.63 | 2x |
| QOFT (r=16) | 1M | 8GB | 0.61 | 8x |
| QOFT (r=8) | 0.5M | 8GB | 0.59 | 16x |

### Quality Comparison

| Model | Method | MMLU | GSM8K | HumanEval | Quant Error |
|-------|--------|------|-------|-----------|-------------|
| 7B | LoRA | 0.62 | 0.49 | 0.25 | 0.00 |
| 7B | QOFT | 0.61 | 0.48 | 0.24 | 0.02 |
| 13B | LoRA | 0.65 | 0.52 | 0.27 | 0.00 |
| 13B | QOFT | 0.64 | 0.51 | 0.26 | 0.025 |

## Advanced Techniques

### QOFT with Mixed Precision

```yaml
# Different precision for different components
qoft_mixed_precision: true
qoft_orthogonal_bits: 8  # Higher precision for orthogonal parts
qoft_update_bits: 4  # Lower precision for updates
```

### QOFT with Adaptive Quantization

```yaml
# Adapt quantization during training
qoft_adaptive: true
qoft_adaptation_metric: loss
qoft_adaptation_threshold: 0.1
```

### QOFT with Regularization

```yaml
# Orthogonality regularization
qoft_ortho_reg: 0.1
qoft_quant_reg: 0.01
qoft_reg_type: l2
```

## Deployment Considerations

### Model Serving
```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load QOFT model
generator = pipeline(
    "text-generation",
    model="saves/llama3-8b/qoft/sft",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

@app.post("/generate")
def generate_text(prompt: str):
    response = generator(prompt, max_length=200)
    return {"response": response[0]["generated_text"]}
```

### Quantization Validation

```python
def validate_qoft_model(model):
    # Check quantization accuracy
    # Validate orthogonality constraints
    # Ensure numerical stability
    pass
```

### Edge Deployment

```python
# Optimize QOFT for edge devices
def optimize_for_edge(model, target_size):
    # Further quantize if needed
    # Optimize for specific hardware
    # Compress model components
    return optimized_model
```

## Comparison with Other Methods

| Aspect | QOFT | LoRA | OFT | PISSA |
|--------|------|------|-----|-------|
| Efficiency | Excellent | Good | Excellent | Excellent |
| Quality | Good | Good | Good | Excellent |
| Complexity | Medium | Low | Medium | High |
| Quantized | Yes | No | No | No |
| Orthogonal | Yes | No | Yes | No |

## Next Steps

- Experiment with different QOFT variants
- Try QOFT with mixed precision
- Use QOFT for edge deployment
- Research QOFT theoretical properties
- Combine QOFT with other techniques

For hands-on examples, see the [notebooks](../../notebooks/advanced_methods/) directory.
