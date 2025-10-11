# AQLM (Activation-aware Quantization for Language Models) Guide

## Overview

AQLM (Activation-aware Quantization for Language Models) is an advanced quantization method that uses multiple codebooks and vector quantization to achieve superior compression ratios while maintaining model quality.

## How AQLM Works

### Core Algorithm

1. **Multiple Codebooks**: Use multiple learned codebooks for better representation
2. **Vector Quantization**: Quantize weight vectors using nearest codebook entries
3. **Activation Analysis**: Use activation data to optimize codebook selection
4. **Layer-wise Optimization**: Optimize codebooks per layer for best performance
5. **Residual Learning**: Learn residuals to compensate for quantization errors

### Mathematical Foundation

For a weight matrix W ∈ ℝ^(d×k):
1. Divide W into vectors of size block_size
2. For each vector w, find closest codebook entry: min ||w - c_i||²
3. Use multiple codebooks for better coverage
4. Learn residuals: W_approx = sum c_i + residual
5. Optimize codebooks using activation-aware loss

## Advantages of AQLM

- **Superior Compression**: Higher compression ratios than uniform quantization
- **Learned Codebooks**: Optimized codebooks for specific weight distributions
- **Activation Aware**: Uses activation data for intelligent optimization
- **Flexible**: Supports various precision levels and block sizes
- **High Quality**: Maintains model quality despite high compression

## When to Use AQLM

- **Maximum compression** is required
- **Storage constraints** are critical
- **Deployment** on edge devices
- **Research** with advanced quantization
- **Custom hardware** with specific quantization support

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
quantization_bit: 4
quantization_type: aqlm
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

### AQLM-Specific Parameters

| Parameter | Description | Values | Impact |
|-----------|-------------|--------|---------|
| `aqlm_codebook_size` | Size of codebooks | 16, 32, 64 | Larger = better quality, more memory |
| `aqlm_block_size` | Block size for quantization | 64, 128 | Smaller = better quality, slower |
| `aqlm_n_bits` | Bits per codebook entry | 8, 16 | Lower = more compression |
| `aqlm_scales_bits` | Bits for scales | 4, 8 | Lower = more compression |

### Advanced Configuration

```yaml
### Advanced AQLM configuration
aqlm_codebook_size: 32
aqlm_block_size: 128
aqlm_n_bits: 8
aqlm_scales_bits: 4
aqlm_outlier_bits: 8  # Higher precision for outliers
aqlm_residual: true  # Learn residuals
aqlm_shared_scales: false  # Independent scales per layer
```

## Hardware Requirements

### Minimum Requirements
- **GPU Memory**: 12GB for 7B models, 24GB for 13B models
- **System RAM**: 64GB
- **Storage**: 50GB for models and datasets
- **GPU**: High-end GPUs with sufficient memory (RTX 4090, A100, H100)

### Performance by Model Size

| Model Size | GPU Memory | Quantization Time | Compression Ratio |
|------------|------------|------------------|------------------|
| 7B | 12-16GB | 15-25 minutes | 8.5x |
| 13B | 24-32GB | 25-40 minutes | 8.2x |
| 30B | 48-64GB | 45-60 minutes | 8.0x |
| 70B | 80-120GB | 60-90 minutes | 7.8x |

## Training Scripts

### Basic AQLM Training
```bash
python src/train.py examples/train_qlora/llama3_lora_sft_aqlm.yaml
```

### AQLM with Custom Parameters
```bash
python src/train.py examples/train_qlm/llama3_lora_sft_aqlm.yaml \
  --aqlm_codebook_size 64 \
  --aqlm_block_size 64 \
  --aqlm_n_bits 8
```

### AQLM for Maximum Compression
```bash
python src/train.py examples/train_qlora/llama3_lora_sft_aqlm.yaml \
  --aqlm_codebook_size 16 \
  --aqlm_n_bits 4 \
  --aqlm_scales_bits 4
```

## Model Loading and Inference

### Loading AQLM Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load AQLM adapter
model = PeftModel.from_pretrained(model, "saves/llama3-8b/aqlm/lora/sft")

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
    adapter_path="saves/llama3-8b/aqlm/lora/sft",
    finetuning_type="lora",
    quantization_bit=4,
    quantization_type="aqlm",
    template="llama3"
))

response = model.chat("What are the benefits of renewable energy?")
print(response)
```

## Hyperparameter Tuning

### Codebook Size Selection

```yaml
# For quality (larger codebooks)
aqlm_codebook_size: 64

# For balance
aqlm_codebook_size: 32

# For compression (smaller codebooks)
aqlm_codebook_size: 16
```

### Block Size Selection

```yaml
# For quality (smaller blocks)
aqlm_block_size: 64

# For speed (larger blocks)
aqlm_block_size: 128

# For very large models
aqlm_block_size: 256
```

### Precision Selection

```yaml
# High precision (better quality)
aqlm_n_bits: 16
aqlm_scales_bits: 8

# Standard precision
aqlm_n_bits: 8
aqlm_scales_bits: 4

# Low precision (maximum compression)
aqlm_n_bits: 4
aqlm_scales_bits: 4
```

## Advanced AQLM Configurations

### AQLM with Residual Learning

```yaml
### Residual learning for better quality
aqlm_residual: true
aqlm_residual_bits: 8
aqlm_residual_layers: 2
```

### AQLM with Outlier Handling

```yaml
# Handle outliers with higher precision
aqlm_outlier: true
aqlm_outlier_bits: 8
aqlm_outlier_threshold: 6.0
```

### AQLM with Layer-wise Optimization

```yaml
# Different settings per layer
aqlm_layer_config: {
  "attention": {"codebook_size": 64, "block_size": 128},
  "mlp": {"codebook_size": 32, "block_size": 64}
}
```

## Evaluation

### AQLM Model Evaluation
```python
from llamafactory.eval import evaluate_aqlm_model

results = evaluate_aqlm_model(
    model_path="saves/llama3-8b/aqlm/lora/sft",
    eval_dataset="alpaca_en_demo",
    metrics=["perplexity", "reconstruction_error", "quality_score"]
)
```

### Manual Evaluation
```python
def evaluate_aqlm_model(model, test_cases):
    results = []
    for prompt in test_cases:
        response = model.chat(prompt)
        # Evaluate response quality
        # Check for quantization artifacts
        results.append(evaluate_response(response))
    return results
```

### Metrics to Track

- **Reconstruction Error**: Mean squared error between original and quantized
- **Perplexity**: Language modeling quality
- **Task Accuracy**: Performance on specific tasks
- **Compression Ratio**: Model size reduction
- **Inference Speed**: Tokens per second

## Best Practices

### 1. Start with Balanced Settings
```yaml
aqlm_codebook_size: 32
aqlm_block_size: 128
aqlm_n_bits: 8
aqlm_scales_bits: 4
```

### 2. Quality vs Compression Trade-off

```yaml
# Maximum quality
aqlm_codebook_size: 64
aqlm_n_bits: 16
aqlm_block_size: 64
aqlm_residual: true

# Balanced
aqlm_codebook_size: 32
aqlm_n_bits: 8
aqlm_block_size: 128
aqlm_residual: false

# Maximum compression
aqlm_codebook_size: 16
aqlm_n_bits: 4
aqlm_block_size: 256
aqlm_residual: false
```

### 3. Training Stability

```yaml
# Conservative settings for AQLM
learning_rate: 5.0e-5
warmup_ratio: 0.2
gradient_checkpointing: true
aqlm_codebook_size: 32  # Not too large
```

### 4. Hardware Optimization

```yaml
# For high-end GPUs
aqlm_codebook_size: 64
aqlm_block_size: 64

# For memory-constrained GPUs
aqlm_codebook_size: 16
aqlm_block_size: 256
```

### 5. Monitoring

```python
def monitor_aqlm_training():
    # Track codebook utilization
    # Monitor reconstruction error
    # Check outlier statistics
    # Validate residual learning
    pass
```

## Troubleshooting

### Common Issues

1. **Codebook Underutilization**
   - Increase codebook size
   - Use activation-aware initialization
   - Try residual learning

2. **Quality Degradation**
   - Increase codebook size
   - Reduce block size
   - Use residual learning

3. **Training Instability**
   - Reduce codebook size
   - Increase learning rate slowly
   - Use gradient clipping

4. **Memory Issues**
   - Reduce codebook size
   - Increase block size
   - Use fewer residual layers

### Debugging Tips

```python
def debug_aqlm_quantization():
    # Check codebook distributions
    # Monitor vector quantization errors
    # Validate residual reconstruction
    # Test outlier detection
    pass
```

## Performance Benchmarks

### Compression Comparison

| Method | Compression Ratio | Quality Score | Speed |
|--------|------------------|---------------|-------|
| FP16 | 1x | 100 | 100 |
| AQLM (32,8) | 8.5x | 97 | 110 |
| AQLM (16,4) | 12x | 94 | 130 |
| AQLM (8,4) | 16x | 88 | 150 |

### Quality Comparison

| Model | Precision | MMLU | GSM8K | HumanEval | Model Size |
|-------|-----------|------|-------|-----------|------------|
| FP16 | 16-bit | 0.65 | 0.52 | 0.28 | 14GB |
| AQLM (32,8) | ~2-bit | 0.63 | 0.50 | 0.25 | 1.6GB |
| AQLM (16,4) | ~1.3-bit | 0.61 | 0.48 | 0.23 | 1.2GB |
| AQLM (8,4) | ~1-bit | 0.58 | 0.45 | 0.20 | 0.9GB |

## Advanced Techniques

### AQLM with Hierarchical Codebooks

```yaml
# Hierarchical codebooks for better representation
aqlm_hierarchical: true
aqlm_hierarchy_levels: 2
aqlm_hierarchy_sizes: [32, 16]
```

### AQLM with Dynamic Codebooks

```yaml
# Adapt codebooks during training
aqlm_dynamic: true
aqlm_update_freq: 100
aqlm_update_rate: 0.1
```

### AQLM with Mixed Precision

```yaml
# Different precision for different components
aqlm_mixed: true
aqlm_codebook_bits: 8  # Higher precision for codebooks
aqlm_index_bits: 4  # Lower precision for indices
```

## Deployment Considerations

### Model Serving
```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load AQLM model
generator = pipeline(
    "text-generation",
    model="saves/llama3-8b/aqlm/lora/sft",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

@app.post("/generate")
def generate_text(prompt: str):
    response = generator(prompt, max_length=200)
    return {"response": response[0]["generated_text"]}
```

### Edge Deployment

```python
# Optimize for edge devices
def optimize_for_edge(model, tokenizer):
    # Quantize further if needed
    # Optimize inference graph
    # Reduce memory footprint
    return optimized_model
```

### Memory Optimization

```python
# Optimize memory usage for AQLM
torch.cuda.empty_cache()
model.eval()
# Use streaming generation for long sequences
```

## Comparison with Other Methods

| Aspect | AQLM | AWQ | GPTQ | QLoRA |
|--------|------|-----|------|-------|
| Compression | Excellent | Good | Good | Good |
| Quality | Good | Excellent | Very Good | Good |
| Speed | Very Fast | Fast | Fast | Fast |
| Memory | Very Low | Low | Low | Low |
| Complexity | High | Medium | Medium | Low |

## Next Steps

- Experiment with different AQLM variants
- Try AQLM with hierarchical codebooks
- Use AQLM for edge deployment
- Combine AQLM with other compression techniques
- Research AQLM for specific hardware

For hands-on examples, see the [notebooks](../../notebooks/quantization/) directory.
