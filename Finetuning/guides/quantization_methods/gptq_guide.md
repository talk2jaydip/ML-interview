# GPTQ (General Purpose Transformer Quantization) Guide

## Overview

GPTQ (General Purpose Transformer Quantization) is a post-training quantization method that uses layer-wise quantization with optimal brain surgeon (OBS) for efficient weight updates.

## How GPTQ Works

### Core Algorithm

1. **Layer-wise Quantization**: Quantize one layer at a time
2. **Optimal Brain Surgeon**: Use second-order optimization to find optimal weight updates
3. **Hessian-based Updates**: Use Hessian information to minimize quantization error
4. **Iterative Process**: Iteratively quantize and update weights
5. **Cholesky Reconstruction**: Reconstruct full precision weights from quantized versions

### Mathematical Foundation

For a weight matrix W ∈ ℝ^(d×k) and quantization function Q:
1. Initialize quantized weights W_q = Q(W)
2. For each layer, solve: min_Δ ||(W + Δ) - W_q||² subject to Q(W + Δ) = W_q
3. Use OBS to find optimal Δ that satisfies the constraint
4. Update W = W + Δ and repeat

## Advantages of GPTQ

- **Post-Training**: Can quantize any pre-trained model without retraining
- **Optimal Updates**: Uses second-order optimization for minimal error
- **Layer-wise**: More stable than end-to-end quantization
- **Flexible**: Works with various quantization schemes
- **Efficient**: Fast quantization with good quality preservation

## When to Use GPTQ

- **Pre-trained models** need to be quantized
- **Limited time** for quantization-aware training
- **Research** with different quantization schemes
- **Deployment** where quantization quality is important
- **Compatibility** with various model architectures

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
quantization_type: gptq
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

### GPTQ-Specific Parameters

| Parameter | Description | Values | Impact |
|-----------|-------------|--------|---------|
| `gptq_block_size` | Block size for quantization | 64, 128 | Smaller = better quality, slower |
| `gptq_group_size` | Group size for scaling | -1, 128 | -1 for channel-wise scaling |
| `gptq_w_bit` | Weight quantization bits | 4, 8 | Lower = more compression |
| `gptq_percdamp` | Damping for OBS | 0.01 | Higher = more conservative updates |

### Advanced Configuration

```yaml
### Advanced GPTQ configuration
gptq_block_size: 128
gptq_group_size: -1
gptq_w_bit: 4
gptq_percdamp: 0.01
gptq_act_order: true  # Activation ordering for better quantization
gptq_static_groups: false  # Dynamic groups for better quality
```

## Hardware Requirements

### Minimum Requirements
- **GPU Memory**: 8GB for 7B models, 16GB for 13B models
- **System RAM**: 32GB
- **Storage**: 30GB for models and datasets
- **GPU**: NVIDIA GPUs with sufficient memory (RTX 30/40, A100, V100)

### Performance by Model Size

| Model Size | GPU Memory | Quantization Time | Quality Score |
|------------|------------|------------------|---------------|
| 7B | 8-12GB | 10-15 minutes | 96/100 |
| 13B | 16-24GB | 15-25 minutes | 94/100 |
| 30B | 32-48GB | 30-45 minutes | 92/100 |
| 70B | 48-80GB | 45-60 minutes | 90/100 |

## Training Scripts

### Basic GPTQ Training
```bash
python src/train.py examples/train_qlora/llama3_lora_sft_gptq.yaml
```

### GPTQ with Custom Parameters
```bash
python src/train.py examples/train_qlora/llama3_lora_sft_gptq.yaml \
  --gptq_block_size 64 \
  --gptq_w_bit 4 \
  --gptq_percdamp 0.01
```

### GPTQ for Large Models
```bash
# For 30B+ models
python src/train.py examples/train_qlora/llama3_lora_sft_gptq.yaml \
  --gptq_block_size 128 \
  --gptq_group_size -1
```

## Model Loading and Inference

### Loading GPTQ Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GptqConfig

# Configure GPTQ quantization
quantization_config = GptqConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    sym=True
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
model = PeftModel.from_pretrained(model, "saves/llama3-8b/gptq/lora/sft")

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
    adapter_path="saves/llama3-8b/gptq/lora/sft",
    finetuning_type="lora",
    quantization_bit=4,
    quantization_type="gptq",
    template="llama3"
))

response = model.chat("What are the benefits of renewable energy?")
print(response)
```

## Hyperparameter Tuning

### Block Size Selection

```yaml
# For quality (smaller blocks)
gptq_block_size: 64

# For speed (larger blocks)
gptq_block_size: 128

# For very large models
gptq_block_size: 256
```

### Group Size Selection

```yaml
# Channel-wise scaling (recommended)
gptq_group_size: -1

# Group-wise scaling
gptq_group_size: 128

# Per-tensor scaling (fastest)
gptq_group_size: 4096
```

### Damping Parameter

```yaml
# Conservative updates
gptq_percdamp: 0.1

# Standard updates
gptq_percdamp: 0.01

# Aggressive updates
gptq_percdamp: 0.001
```

## Advanced GPTQ Configurations

### GPTQ with Activation Ordering

```yaml
### Activation ordering for better quantization
gptq_act_order: true
gptq_static_groups: false
gptq_true_sequential: true
```

### GPTQ with Different LoRA Targets

```yaml
# Attention-only GPTQ
lora_target: q_proj,k_proj,v_proj,o_proj
gptq_skip_layers: mlp

# MLP-only GPTQ
lora_target: gate_proj,up_proj,down_proj
gptq_skip_layers: attention
```

### GPTQ with Custom Quantization Scheme

```yaml
# Mixed precision GPTQ
gptq_w_bit: 4  # Most weights 4-bit
gptq_w_bit_attn: 8  # Attention weights 8-bit
gptq_w_bit_mlp: 4  # MLP weights 4-bit
```

## Evaluation

### GPTQ Model Evaluation
```python
from llamafactory.eval import evaluate_gptq_model

results = evaluate_gptq_model(
    model_path="saves/llama3-8b/gptq/lora/sft",
    eval_dataset="alpaca_en_demo",
    metrics=["perplexity", "accuracy", "gptq_error"]
)
```

### Manual Evaluation
```python
def evaluate_gptq_model(model, test_cases):
    results = []
    for prompt in test_cases:
        response = model.chat(prompt)
        # Evaluate response quality
        # Check for quantization artifacts
        results.append(evaluate_response(response))
    return results
```

### Metrics to Track

- **Quantization Error**: Mean squared error between original and quantized
- **Perplexity**: Language modeling quality
- **Task Accuracy**: Performance on specific tasks
- **Inference Speed**: Tokens per second
- **Memory Usage**: GPU memory consumption

## Best Practices

### 1. Start with Conservative Settings
```yaml
gptq_block_size: 128
gptq_group_size: -1
gptq_w_bit: 4
gptq_percdamp: 0.01
gptq_act_order: true
```

### 2. Quality vs Speed Trade-off

```yaml
# Maximum quality
gptq_block_size: 64
gptq_group_size: -1
gptq_act_order: true
gptq_true_sequential: true

# Balanced
gptq_block_size: 128
gptq_group_size: -1
gptq_act_order: true

# Maximum speed
gptq_block_size: 256
gptq_group_size: 128
gptq_act_order: false
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
gptq_act_order: true
gptq_group_size: -1

# For older GPUs
gptq_act_order: false
gptq_group_size: 128
```

### 5. Monitoring

```python
def monitor_gptq_training():
    # Track quantization error per layer
    # Monitor Hessian condition numbers
    # Check update magnitudes
    # Validate reconstruction quality
    pass
```

## Troubleshooting

### Common Issues

1. **Quantization Errors**
   - Check GPU compatibility
   - Update GPTQ libraries
   - Use supported model architectures

2. **Quality Degradation**
   - Reduce block size
   - Use activation ordering
   - Try different group sizes

3. **Training Instability**
   - Increase damping parameter
   - Use smaller learning rate
   - Increase warmup steps

4. **Memory Issues**
   - Increase block size
   - Use group-wise scaling
   - Reduce batch size

### Debugging Tips

```python
def debug_gptq_quantization():
    # Check Hessian matrices
    # Monitor layer-wise errors
    # Validate OBS updates
    # Test Cholesky reconstruction
    pass
```

## Performance Benchmarks

### Quality Comparison

| Model | Precision | MMLU | GSM8K | HumanEval | Quantization Error |
|-------|-----------|------|-------|-----------|-------------------|
| FP16 | 16-bit | 0.65 | 0.52 | 0.28 | 0.00 |
| GPTQ (8-bit) | 8-bit | 0.63 | 0.50 | 0.26 | 0.02 |
| GPTQ (4-bit) | 4-bit | 0.60 | 0.47 | 0.23 | 0.05 |
| GPTQ (3-bit) | 3-bit | 0.55 | 0.42 | 0.18 | 0.08 |

### Inference Speed

| Hardware | FP16 | GPTQ (4-bit) | Speedup |
|----------|------|-------------|---------|
| RTX 4090 | 100 tok/s | 140 tok/s | 1.4x |
| A100 | 200 tok/s | 260 tok/s | 1.3x |
| V100 | 80 tok/s | 100 tok/s | 1.25x |

## Advanced Techniques

### GPTQ with Layer-wise Mixed Precision

```yaml
# Different precision for different layers
gptq_mixed_precision: true
gptq_attention_bits: 8  # Higher precision for attention
gptq_mlp_bits: 4  # Lower precision for MLP
```

### GPTQ with Dynamic Groups

```yaml
# Dynamic group sizes based on layer characteristics
gptq_static_groups: false
gptq_group_size: -1  # Channel-wise
```

### GPTQ with Quantization-aware Updates

```yaml
# Update quantization during training
gptq_update_freq: 100
gptq_update_method: hessian
```

## Deployment Considerations

### Model Serving
```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load GPTQ model
generator = pipeline(
    "text-generation",
    model="saves/llama3-8b/gptq/lora/sft",
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
def batch_inference_gptq(model, tokenizer, prompts):
    # Efficient batch processing
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

### Memory Optimization

```python
# Optimize memory usage for GPTQ
torch.cuda.empty_cache()
model.eval()
# Use smaller batch sizes if needed
```

## Comparison with Other Methods

| Aspect | GPTQ | AWQ | AQLM | QLoRA |
|--------|------|-----|------|-------|
| Quality | Very Good | Excellent | Good | Good |
| Speed | Fast | Fast | Very Fast | Fast |
| Memory | Low | Low | Very Low | Low |
| Ease of Use | High | High | Medium | High |
| Research Friendly | High | Medium | Low | Medium |

## Next Steps

- Experiment with different GPTQ variants
- Try GPTQ with mixed precision
- Use GPTQ for model compression
- Combine GPTQ with other optimization techniques
- Deploy GPTQ models in production

For hands-on examples, see the [notebooks](../../notebooks/quantization/) directory.
