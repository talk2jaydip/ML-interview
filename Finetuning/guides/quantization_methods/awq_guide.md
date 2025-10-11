# AWQ (Activation-aware Weight Quantization) Guide

## Overview

AWQ (Activation-aware Weight Quantization) is a state-of-the-art quantization method that protects salient weights during quantization, leading to better quality preservation compared to traditional quantization methods.

## How AWQ Works

### Core Algorithm

1. **Activation Analysis**: Analyze activation patterns during forward passes
2. **Salient Weight Detection**: Identify weights that are important for model outputs
3. **Weight Protection**: Protect important weights from aggressive quantization
4. **Channel-wise Quantization**: Apply quantization per channel with appropriate scaling
5. **Compensation**: Scale remaining weights to compensate for quantization effects

### Mathematical Foundation

For a weight matrix W ∈ ℝ^(d×k), AWQ:
1. Computes activation-based importance scores
2. Protects top-k% most important weights
3. Quantizes remaining weights with learned scaling factors
4. Compensates protected weights for quantization effects

## Advantages of AWQ

- **Quality Preservation**: Maintains model quality better than uniform quantization
- **Activation Aware**: Uses real activation data for intelligent quantization
- **Automatic**: No manual calibration or hyperparameter tuning required
- **Efficient**: Fast quantization process with minimal overhead
- **Robust**: Works well across different model architectures

## When to Use AWQ

- **High-quality quantization** is required
- **Limited computational resources** for quantization
- **Production deployment** where quality matters
- **Research and experimentation** with quantization effects
- **Fine-tuning** quantized models for specific tasks

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
quantization_type: awq
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

### AWQ-Specific Parameters

| Parameter | Description | Values | Impact |
|-----------|-------------|--------|---------|
| `awq_block_size` | Block size for quantization | 64, 128 | Smaller = better quality, slower |
| `awq_group_size` | Group size for scaling | -1, 128 | -1 for channel-wise scaling |
| `awq_w_bit` | Weight quantization bits | 4, 8 | Lower = more compression |
| `awq_version` | AWQ implementation version | GEMM, GEMV | GEMM for modern GPUs |

### Advanced Configuration

```yaml
### Advanced AWQ configuration
awq_block_size: 128
awq_group_size: -1  # Channel-wise scaling
awq_w_bit: 4
awq_version: GEMM  # For modern GPUs
awq_protect_ratio: 0.1  # Protect 10% of weights
awq_skip_layers: null  # Skip quantization for specific layers
```

## Hardware Requirements

### Minimum Requirements
- **GPU Memory**: 8GB for 7B models, 16GB for 13B models
- **System RAM**: 32GB
- **Storage**: 30GB for models and datasets
- **GPU**: Modern NVIDIA GPUs (RTX 30/40 series, A100, V100)

### Performance by Model Size

| Model Size | GPU Memory | Quantization Time | Inference Speed |
|------------|------------|------------------|-----------------|
| 7B | 8-12GB | 5-10 minutes | 150% of FP16 |
| 13B | 16-24GB | 10-15 minutes | 140% of FP16 |
| 30B | 32-48GB | 20-30 minutes | 130% of FP16 |
| 70B | 48-80GB | 30-45 minutes | 120% of FP16 |

## Training Scripts

### Basic AWQ Training
```bash
python src/train.py examples/train_qlora/llama3_lora_sft_awq.yaml
```

### AWQ with Custom Parameters
```bash
python src/train.py examples/train_qlora/llama3_lora_sft_awq.yaml \
  --awq_block_size 64 \
  --awq_w_bit 4 \
  --learning_rate 2.0e-4
```

### AWQ for Different Model Sizes
```bash
# For 7B models
python src/train.py examples/train_qlora/llama3_lora_sft_awq.yaml

# For 13B models
python src/train.py examples/train_qlora/llama3_lora_sft_awq.yaml \
  --awq_block_size 128
```

## Model Loading and Inference

### Loading AWQ Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, AwqConfig

# Configure AWQ quantization
quantization_config = AwqConfig(
    bits=4,
    group_size=128,
    zero_point=True,
    version="GEMM"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
from peft import PeftModel
model = PeftModel.from_pretrained(model, "saves/llama3-8b/awq/lora/sft")

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
    adapter_path="saves/llama3-8b/awq/lora/sft",
    finetuning_type="lora",
    quantization_bit=4,
    quantization_type="awq",
    template="llama3"
))

response = model.chat("What are the benefits of renewable energy?")
print(response)
```

## Hyperparameter Tuning

### Block Size Selection

```yaml
# For quality (smaller blocks)
awq_block_size: 64

# For speed (larger blocks)
awq_block_size: 128

# For balance
awq_block_size: 128  # with group_size=-1 for channel-wise
```

### Group Size Selection

```yaml
# Channel-wise scaling (recommended)
awq_group_size: -1

# Group-wise scaling
awq_group_size: 128

# Per-tensor scaling
awq_group_size: 4096
```

### Weight Protection

```yaml
# Protect more weights for quality
awq_protect_ratio: 0.15  # Protect 15% of weights

# Protect fewer weights for compression
awq_protect_ratio: 0.05  # Protect 5% of weights
```

## Advanced AWQ Configurations

### AWQ with LoRA+
```yaml
### LoRA+ configuration with AWQ
finetuning_type: lora
quantization_bit: 4
quantization_type: awq
lora_rank: 8
lora_alpha: 32
lora_dropout: 0.1
lora_target: all
loraplus_lr_ratio: 16
```

### AWQ with Different LoRA Targets

```yaml
# Attention-only AWQ
lora_target: q_proj,k_proj,v_proj,o_proj
awq_skip_layers: mlp  # Skip quantization for MLP layers

# MLP-only AWQ
lora_target: gate_proj,up_proj,down_proj
awq_skip_layers: attention  # Skip quantization for attention layers
```

### AWQ with Custom Quantization Scheme

```yaml
# Mixed precision AWQ
awq_w_bit: 4  # Most weights 4-bit
awq_w_bit_attn: 8  # Attention weights 8-bit
awq_w_bit_mlp: 4  # MLP weights 4-bit
```

## Evaluation

### AWQ Model Evaluation
```python
from llamafactory.eval import evaluate_quantized_model

results = evaluate_quantized_model(
    model_path="saves/llama3-8b/awq/lora/sft",
    eval_dataset="alpaca_en_demo",
    metrics=["perplexity", "accuracy", "quality_score"]
)
```

### Manual Evaluation
```python
def evaluate_awq_model(model, test_cases):
    results = []
    for prompt in test_cases:
        response = model.chat(prompt)
        # Evaluate response quality
        # Check for quantization artifacts
        results.append(evaluate_response(response))
    return results
```

### Metrics to Track

- **Perplexity**: Language modeling quality
- **Task Accuracy**: Performance on specific tasks
- **Quality Score**: Human-rated response quality
- **Inference Speed**: Tokens per second
- **Memory Usage**: GPU memory consumption

## Best Practices

### 1. Start with Standard Settings
```yaml
awq_block_size: 128
awq_group_size: -1
awq_w_bit: 4
awq_version: GEMM
```

### 2. Quality vs Compression Trade-off

```yaml
# Maximum quality
awq_w_bit: 8
awq_block_size: 64
awq_protect_ratio: 0.2

# Balanced
awq_w_bit: 4
awq_block_size: 128
awq_protect_ratio: 0.1

# Maximum compression
awq_w_bit: 4
awq_block_size: 256
awq_protect_ratio: 0.05
```

### 3. Training Stability

```yaml
# Conservative learning rate for quantized models
learning_rate: 5.0e-5
warmup_ratio: 0.2
gradient_checkpointing: true
```

### 4. Hardware Optimization

```yaml
# For RTX 30/40 series
awq_version: GEMM
awq_group_size: -1

# For A100/H100
awq_version: GEMM
awq_group_size: 128
```

### 5. Monitoring

```python
def monitor_awq_training():
    # Track quantization error
    # Monitor weight distributions
    # Check activation patterns
    # Validate model quality
    pass
```

## Troubleshooting

### Common Issues

1. **Quantization Errors**
   - Check GPU compatibility
   - Update AWQ libraries
   - Use supported model architectures

2. **Quality Degradation**
   - Reduce block size
   - Increase protection ratio
   - Use higher bit precision

3. **Training Instability**
   - Reduce learning rate
   - Increase warmup steps
   - Use gradient clipping

4. **Memory Issues**
   - Increase block size
   - Use group-wise scaling
   - Reduce batch size

### Debugging Tips

```python
def debug_awq_quantization():
    # Check quantization ranges
    # Monitor protected weights
    # Validate scaling factors
    # Test dequantization accuracy
    pass
```

## Performance Benchmarks

### Quality Comparison

| Model | Precision | MMLU | GSM8K | HumanEval | Memory |
|-------|-----------|------|-------|-----------|--------|
| FP16 | 16-bit | 0.65 | 0.52 | 0.28 | 14GB |
| AWQ (8-bit) | 8-bit | 0.64 | 0.51 | 0.27 | 7GB |
| AWQ (4-bit) | 4-bit | 0.62 | 0.49 | 0.25 | 4GB |
| GPTQ (4-bit) | 4-bit | 0.60 | 0.47 | 0.23 | 4GB |

### Inference Speed

| Hardware | FP16 | AWQ (4-bit) | Speedup |
|----------|------|-------------|---------|
| RTX 4090 | 100 tok/s | 150 tok/s | 1.5x |
| A100 | 200 tok/s | 280 tok/s | 1.4x |
| V100 | 80 tok/s | 110 tok/s | 1.4x |

## Advanced Techniques

### AWQ with Mixed Precision

```yaml
# Different precision for different layers
awq_mixed_precision: true
awq_attention_bits: 8  # Higher precision for attention
awq_mlp_bits: 4  # Lower precision for MLP
```

### AWQ with Activation Quantization

```yaml
# Quantize activations as well
awq_activation_bits: 8
awq_activation_quant: true
```

### AWQ with Dynamic Scaling

```yaml
# Learn scaling factors during training
awq_learn_scales: true
awq_scale_lr: 1.0e-3
```

## Deployment Considerations

### Model Serving
```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load AWQ model
generator = pipeline(
    "text-generation",
    model="saves/llama3-8b/awq/lora/sft",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

@app.post("/generate")
def generate_text(prompt: str):
    response = generator(prompt, max_length=200)
    return {"response": response[0]["generated_text"]}
```

### Batch Processing

```python
def batch_inference_awq(model, tokenizer, prompts):
    # Efficient batch processing
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

### Memory Optimization

```python
# Optimize memory usage
torch.cuda.empty_cache()  # Clear cache
model.eval()  # Set to evaluation mode
# Use smaller batch sizes
```

## Comparison with Other Methods

| Aspect | AWQ | GPTQ | AQLM | QLoRA |
|--------|-----|------|------|-------|
| Quality | Excellent | Very Good | Good | Good |
| Speed | Fast | Fast | Very Fast | Fast |
| Memory | Low | Low | Very Low | Low |
| Ease of Use | High | High | Medium | High |
| Hardware Support | Good | Good | Limited | Good |

## Next Steps

- Experiment with different AWQ variants
- Try AWQ with mixed precision
- Use AWQ for model compression
- Combine AWQ with other optimization techniques
- Deploy AWQ models in production

For hands-on examples, see the [notebooks](../../notebooks/quantization/) directory.
