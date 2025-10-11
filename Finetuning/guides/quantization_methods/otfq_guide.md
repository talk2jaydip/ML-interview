# OTFQ (On-the-fly Quantization) Guide

## Overview

OTFQ (On-the-fly Quantization) performs quantization dynamically during training, adapting quantization parameters based on the current training state for optimal performance.

## How OTFQ Works

### Core Algorithm

1. **Dynamic Quantization**: Quantize/dequantize weights during forward pass
2. **Adaptive Scaling**: Adjust quantization parameters based on gradients
3. **Training-time Optimization**: Optimize quantization for current training phase
4. **Mixed Precision**: Use different precision for different layers
5. **Momentum-based Updates**: Smoothly update quantization parameters

### Mathematical Foundation

For a weight matrix W and quantization function Q:
1. During forward pass: W_q = Q(W, s) where s are learned scales
2. During backward pass: Update s based on gradients
3. Adaptive update: s = s + α * ∇s where α is learning rate
4. Smooth adaptation: s_t = β * s_{t-1} + (1-β) * s_t

## Advantages of OTFQ

- **Adaptive**: Adjusts to training dynamics automatically
- **Dynamic**: No pre-computed quantization parameters
- **Training-aware**: Optimizes quantization for training process
- **Flexible**: Can change precision during training
- **Robust**: Handles various training scenarios well

## When to Use OTFQ

- **Dynamic training** scenarios with changing requirements
- **Adaptive systems** that need to adjust to data
- **Research** with quantization during training
- **Scenarios** requiring automatic quantization tuning
- **Training** with varying computational constraints

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
quantization_type: otfq
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

### OTFQ-Specific Parameters

| Parameter | Description | Values | Impact |
|-----------|-------------|--------|---------|
| `otfq_block_size` | Block size for quantization | 64, 128 | Smaller = better quality, slower |
| `otfq_warmup_steps` | Warmup steps for quantization | 100, 500 | Longer = more stable |
| `otfq_decay` | Decay rate for quantization | 0.9, 0.99 | Higher = slower adaptation |
| `otfq_update_freq` | Update frequency | 10, 50 | Lower = more updates |

### Advanced Configuration

```yaml
### Advanced OTFQ configuration
otfq_block_size: 128
otfq_warmup_steps: 500
otfq_decay: 0.95
otfq_update_freq: 20
otfq_momentum: 0.9  # Momentum for scale updates
otfq_clip: 10.0  # Gradient clipping for scales
otfq_adaptive: true  # Adaptive quantization
otfq_layer_wise: false  # Layer-wise quantization parameters
```

## Hardware Requirements

### Minimum Requirements
- **GPU Memory**: 8GB for 7B models, 16GB for 13B models
- **System RAM**: 32GB
- **Storage**: 30GB for models and datasets
- **GPU**: Modern NVIDIA GPUs with good memory bandwidth

### Performance by Model Size

| Model Size | GPU Memory | Training Time | Adaptation Speed |
|------------|------------|---------------|------------------|
| 7B | 8-12GB | 2-3 hours | Fast |
| 13B | 16-24GB | 3-5 hours | Medium |
| 30B | 32-48GB | 6-8 hours | Slow |
| 70B | 48-80GB | 10-15 hours | Very Slow |

## Training Scripts

### Basic OTFQ Training
```bash
python src/train.py examples/train_qlora/llama3_lora_sft_otfq.yaml
```

### OTFQ with Custom Parameters
```bash
python src/train.py examples/train_qlora/llama3_lora_sft_otfq.yaml \
  --otfq_warmup_steps 1000 \
  --otfq_decay 0.99 \
  --otfq_update_freq 10
```

### OTFQ for Adaptive Training
```bash
python src/train.py examples/train_qlora/llama3_lora_sft_otfq.yaml \
  --otfq_adaptive true \
  --otfq_layer_wise true
```

## Model Loading and Inference

### Loading OTFQ Model
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

# Load OTFQ adapter
model = PeftModel.from_pretrained(model, "saves/llama3-8b/otfq/lora/sft")

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
    adapter_path="saves/llama3-8b/otfq/lora/sft",
    finetuning_type="lora",
    quantization_bit=4,
    quantization_type="otfq",
    template="llama3"
))

response = model.chat("What are the benefits of renewable energy?")
print(response)
```

## Hyperparameter Tuning

### Warmup Steps Selection

```yaml
# Short warmup (aggressive adaptation)
otfq_warmup_steps: 100

# Standard warmup (balanced)
otfq_warmup_steps: 500

# Long warmup (conservative)
otfq_warmup_steps: 1000
```

### Decay Rate Selection

```yaml
# Fast adaptation
otfq_decay: 0.9

# Medium adaptation
otfq_decay: 0.95

# Slow adaptation
otfq_decay: 0.99
```

### Update Frequency Selection

```yaml
# Frequent updates (more adaptive)
otfq_update_freq: 10

# Standard updates (balanced)
otfq_update_freq: 20

# Infrequent updates (more stable)
otfq_update_freq: 50
```

## Advanced OTFQ Configurations

### OTFQ with Momentum

```yaml
### Momentum-based scale updates
otfq_momentum: 0.9
otfq_momentum_type: polyak  # or exponential
otfq_momentum_beta: 0.9
```

### OTFQ with Adaptive Quantization

```yaml
# Adapt quantization based on training dynamics
otfq_adaptive: true
otfq_adaptation_metric: loss  # or gradient_norm
otfq_adaptation_threshold: 0.1
```

### OTFQ with Layer-wise Adaptation

```yaml
# Different quantization for different layers
otfq_layer_wise: true
otfq_attention_quant: 4  # 4-bit for attention
otfq_mlp_quant: 8  # 8-bit for MLP
```

## Evaluation

### OTFQ Model Evaluation
```python
from llamafactory.eval import evaluate_otfq_model

results = evaluate_otfq_model(
    model_path="saves/llama3-8b/otfq/lora/sft",
    eval_dataset="alpaca_en_demo",
    metrics=["perplexity", "adaptation_rate", "quality_score"]
)
```

### Manual Evaluation
```python
def evaluate_otfq_model(model, test_cases):
    results = []
    for prompt in test_cases:
        response = model.chat(prompt)
        # Evaluate response quality
        # Check adaptation effectiveness
        results.append(evaluate_response(response))
    return results
```

### Metrics to Track

- **Adaptation Rate**: How quickly quantization parameters adapt
- **Perplexity**: Language modeling quality
- **Task Accuracy**: Performance on specific tasks
- **Stability**: Training stability over time
- **Efficiency**: Memory and compute efficiency

## Best Practices

### 1. Start with Conservative Settings
```yaml
otfq_warmup_steps: 500
otfq_decay: 0.95
otfq_update_freq: 20
otfq_momentum: 0.9
```

### 2. Quality vs Adaptation Trade-off

```yaml
# Quality-focused (slower adaptation)
otfq_warmup_steps: 1000
otfq_decay: 0.99
otfq_update_freq: 50

# Adaptation-focused (faster adaptation)
otfq_warmup_steps: 200
otfq_decay: 0.9
otfq_update_freq: 10

# Balanced
otfq_warmup_steps: 500
otfq_decay: 0.95
otfq_update_freq: 20
```

### 3. Training Stability

```yaml
# Conservative settings for OTFQ
learning_rate: 5.0e-5
warmup_ratio: 0.2
gradient_checkpointing: true
otfq_clip: 10.0  # Gradient clipping
```

### 4. Hardware Optimization

```yaml
# For stable training
otfq_update_freq: 50
otfq_decay: 0.99

# For fast adaptation
otfq_update_freq: 10
otfq_decay: 0.9
```

### 5. Monitoring

```python
def monitor_otfq_training():
    # Track scale parameter evolution
    # Monitor adaptation rate
    # Check quantization error trends
    # Validate training stability
    pass
```

## Troubleshooting

### Common Issues

1. **Unstable Adaptation**
   - Increase warmup steps
   - Reduce update frequency
   - Use higher decay rate

2. **Quality Degradation**
   - Increase warmup steps
   - Use more conservative adaptation
   - Add regularization

3. **Training Instability**
   - Reduce learning rate
   - Increase gradient clipping
   - Use slower adaptation

4. **Memory Issues**
   - Increase update frequency
   - Reduce block size
   - Use simpler adaptation

### Debugging Tips

```python
def debug_otfq_quantization():
    # Monitor scale parameter distributions
    # Check adaptation step sizes
    # Validate gradient flow
    # Test quantization/dequantization
    pass
```

## Performance Benchmarks

### Adaptation Comparison

| Method | Adaptation Speed | Quality | Stability | Overhead |
|--------|------------------|---------|-----------|----------|
| OTFQ (fast) | High | Good | Medium | 10% |
| OTFQ (balanced) | Medium | Very Good | High | 5% |
| OTFQ (slow) | Low | Excellent | Very High | 2% |

### Quality Comparison

| Model | Precision | MMLU | GSM8K | HumanEval | Adaptation |
|-------|-----------|------|-------|-----------|------------|
| FP16 | 16-bit | 0.65 | 0.52 | 0.28 | N/A |
| OTFQ (4-bit) | Adaptive | 0.62 | 0.49 | 0.25 | High |
| OTFQ (8-bit) | Adaptive | 0.64 | 0.51 | 0.27 | Medium |

## Advanced Techniques

### OTFQ with Curriculum Learning

```yaml
# Start with higher precision, gradually reduce
otfq_curriculum: true
otfq_curriculum_stages: 3
otfq_curriculum_schedule: [8, 6, 4]  # Bits per stage
```

### OTFQ with Multi-objective Optimization

```yaml
# Optimize for both quality and efficiency
otfq_multi_objective: true
otfq_quality_weight: 0.7
otfq_efficiency_weight: 0.3
```

### OTFQ with Dynamic Precision

```yaml
# Change precision based on layer importance
otfq_dynamic_precision: true
otfq_importance_metric: gradient_norm
otfq_precision_range: [3, 8]  # Min/max bits
```

## Deployment Considerations

### Model Serving
```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load OTFQ model
generator = pipeline(
    "text-generation",
    model="saves/llama3-8b/otfq/lora/sft",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

@app.post("/generate")
def generate_text(prompt: str):
    response = generator(prompt, max_length=200)
    return {"response": response[0]["generated_text"]}
```

### Dynamic Adaptation

```python
# Adapt quantization during inference
def dynamic_otfq_inference(model, tokenizer, prompts, adapt=True):
    if adapt:
        # Adapt quantization parameters based on input
        model.adapt_quantization(prompts)
    # Generate responses
    return generate_responses(model, tokenizer, prompts)
```

### Memory Optimization

```python
# Optimize memory usage for OTFQ
torch.cuda.empty_cache()
model.eval()
# Use efficient inference modes
```

## Comparison with Other Methods

| Aspect | OTFQ | AWQ | GPTQ | AQLM |
|--------|------|-----|------|------|
| Adaptivity | Excellent | None | None | Limited |
| Quality | Very Good | Excellent | Very Good | Good |
| Speed | Fast | Fast | Fast | Very Fast |
| Flexibility | Excellent | Medium | Medium | Medium |
| Research Value | High | Medium | Medium | Medium |

## Next Steps

- Experiment with different OTFQ variants
- Try OTFQ with curriculum learning
- Use OTFQ for adaptive systems
- Combine OTFQ with other techniques
- Research OTFQ for specific applications

For hands-on examples, see the [notebooks](../../notebooks/quantization/) directory.
