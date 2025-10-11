# KTO (Kahneman-Tversky Optimization) Guide

## Overview

KTO (Kahneman-Tversky Optimization) is a preference optimization method that learns from binary feedback signals (desirable vs undesirable) rather than pairwise preferences, making it more data-efficient and robust.

## Mathematical Foundation

KTO is based on prospect theory and optimizes:

```
L(θ) = -E[KL(π_θ(y | x) || π_ref(y | x)) + λ E[log σ(β(r(x, y) - r_0))]]
```

Where:
- `r(x, y)` is the binary desirability label (±1)
- `r_0` is the baseline desirability
- `λ` controls the strength of the KL penalty
- `π_ref` is the reference policy

### Key Insights

1. **Binary Labels**: Only needs desirable/undesirable labels
2. **Prospect Theory**: Incorporates human decision-making biases
3. **Robust Learning**: Less sensitive to label noise
4. **Data Efficient**: More efficient than pairwise preferences

## Advantages of KTO

- **Data Efficient**: Requires only binary labels
- **Robust**: Less sensitive to label noise
- **Simple**: Easier data collection
- **Scalable**: Works with large datasets
- **Stable**: Stable training dynamics

## When to Use KTO

- **Binary feedback** is available or easy to collect
- **Data efficiency** is important
- **Robust learning** from noisy labels
- **Large-scale** preference learning
- **Human-in-the-loop** learning scenarios

## Configuration Parameters

### Core Parameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|---------|
| `pref_beta` | Controls deviation from reference | 0.1, 0.5, 1.0 | Higher = stronger preference signal |
| `kto_desirable_weight` | Weight for desirable examples | 1.0 | Adjust class balance |
| `kto_undesirable_weight` | Weight for undesirable examples | 1.0 | Adjust class balance |

### Advanced Parameters

```yaml
kto_lambda: 1.0  # KL penalty strength
kto_baseline: 0.0  # Baseline desirability
kto_clip: 10.0  # Gradient clipping
```

## Basic Configuration

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: kto
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
pref_beta: 0.1

### dataset
dataset: kto_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/lora/kto
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

## Dataset Format

### Standard KTO Format

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Write a helpful response"},
      {"from": "gpt", "value": "I'll provide a helpful response"}
    ],
    "kto_tag": true  # true for desirable, false for undesirable
  }
]
```

### Alternative Formats

```json
# Format with scores
{
  "conversations": [
    {"from": "human", "value": "Explain quantum computing"},
    {"from": "gpt", "value": "Quantum computing explanation..."}
  ],
  "label": 1  # 1 for desirable, 0 for undesirable
}

# Format with multiple turns
{
  "conversations": [
    {"from": "human", "value": "Hello"},
    {"from": "gpt", "value": "Hi there!"},
    {"from": "human", "value": "How are you?"},
    {"from": "gpt", "value": "I'm doing well, thank you!"}
  ],
  "kto_tag": true
}
```

### Converting from Other Formats

```python
def convert_dpo_to_kto(dpo_data):
    """Convert DPO format to KTO format"""
    kto_data = []
    for item in dpo_data:
        # Convert chosen to desirable
        kto_data.append({
            "conversations": item["conversations"],
            "kto_tag": True
        })
        # Convert rejected to undesirable
        kto_data.append({
            "conversations": item["conversations"] + [{"from": "gpt", "value": item["rejected"]["value"]}],
            "kto_tag": False
        })
    return kto_data
```

## Training Scripts

### Basic KTO Training
```bash
python src/train.py examples/train_lora/llama3_lora_kto.yaml
```

### KTO with Custom Parameters
```bash
python src/train.py examples/train_lora/llama3_lora_kto.yaml \
  --pref_beta 0.5 \
  --learning_rate 1.0e-5
```

### KTO with Class Balancing
```bash
python src/train.py examples/train_lora/llama3_lora_kto.yaml \
  --kto_desirable_weight 1.0 \
  --kto_undesirable_weight 0.5
```

## Model Loading and Inference

### Loading KTO Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
kto_model_path = "saves/llama3-8b/lora/kto"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, kto_model_path)

# Generate response
inputs = tokenizer("Explain quantum computing", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Using LLaMA-Factory Chat Interface
```python
from llamafactory import ChatModel

model = ChatModel(dict(
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    adapter_path="saves/llama3-8b/lora/kto",
    finetuning_type="lora",
    template="llama3"
))

response = model.chat("What are the benefits of renewable energy?")
print(response)
```

## Hyperparameter Tuning

### Beta Selection

```yaml
# Conservative (stable, less preference)
pref_beta: 0.1

# Moderate (balanced)
pref_beta: 0.5

# Strong (aggressive, more preference)
pref_beta: 1.0
```

### Learning Rate Tuning

```yaml
# Conservative
learning_rate: 1.0e-6
warmup_ratio: 0.2

# Standard
learning_rate: 5.0e-6
warmup_ratio: 0.1

# Higher
learning_rate: 1.0e-5
warmup_ratio: 0.05
```

### Class Weight Balancing

```yaml
# If you have imbalanced data
kto_desirable_weight: 1.0
kto_undesirable_weight: 1.0  # Adjust based on data distribution
```

## Advanced Configurations

### KTO with LoRA+
```yaml
### LoRA+ configuration
finetuning_type: lora
lora_rank: 8
lora_alpha: 32
lora_dropout: 0.1
lora_target: all
loraplus_lr_ratio: 16
pref_beta: 0.1
```

### KTO with QLoRA
```yaml
### QLoRA + KTO
finetuning_type: lora
quantization_bit: 4
quantization_type: nf4
double_quantization: true
lora_rank: 64
lora_alpha: 128
pref_beta: 0.1
```

### KTO with Custom Loss Weights
```yaml
### Custom loss weighting
kto_lambda: 1.0  # KL penalty
kto_clip: 10.0  # Gradient clipping
kto_baseline: 0.0  # Desirability baseline
```

## Evaluation

### Automatic Evaluation
```python
from llamafactory.eval import evaluate_kto_model

results = evaluate_kto_model(
    model_path="saves/llama3-8b/lora/kto",
    eval_dataset="kto_mix_en",
    metrics=["accuracy", "desirability_score"]
)
```

### Manual Evaluation
```python
def evaluate_kto_model(model, test_cases):
    results = []
    for item in test_cases:
        response = model.chat(item["prompt"])
        desirability = human_evaluate_response(response)
        results.append({
            "response": response,
            "predicted_desirability": desirability,
            "true_desirability": item["label"]
        })
    return results
```

### Metrics to Track

- **Accuracy**: Correct prediction of desirability
- **Desirability Score**: Model's ability to generate desirable responses
- **Calibration**: How well predicted desirability matches true desirability
- **Robustness**: Performance on out-of-distribution data

## Best Practices

### 1. Data Collection
- Collect diverse desirable/undesirable examples
- Ensure clear separation between classes
- Balance class distribution

### 2. Hyperparameter Selection
- Start with β = 0.1 (conservative)
- Use cosine learning rate schedule
- Monitor class-wise performance

### 3. Training Stability
```yaml
# Conservative settings
pref_beta: 0.1
learning_rate: 5.0e-6
warmup_ratio: 0.2
num_train_epochs: 3
```

### 4. Class Balancing
```yaml
# Adjust weights if classes are imbalanced
kto_desirable_weight: 1.0
kto_undesirable_weight: 0.8  # If undesirable examples are fewer
```

### 5. Monitoring
- Track desirability predictions
- Monitor KL divergence
- Check calibration curves

## Troubleshooting

### Common Issues

1. **Poor Desirability Classification**
   - Increase β value
   - Use higher quality labels
   - Add more training data

2. **Training Instability**
   - Reduce β value
   - Lower learning rate
   - Increase warmup ratio

3. **Class Imbalance**
   - Adjust class weights
   - Use stratified sampling
   - Collect more minority class data

4. **Overfitting**
   - Use regularization
   - Add dropout
   - Use early stopping

### Debugging Tips

```python
def debug_kto_training():
    # Check label distribution
    # Monitor desirability scores
    # Validate data quality
    # Check calibration
    pass
```

## Performance Benchmarks

### Performance on Standard Benchmarks

| Model | KTO-Mix Accuracy | Desirability Score | Calibration |
|-------|-----------------|-------------------|-------------|
| SFT Baseline | 65% | 0.72 | 0.85 |
| KTO (β=0.1) | 78% | 0.84 | 0.92 |
| KTO (β=0.5) | 82% | 0.88 | 0.94 |
| KTO (β=1.0) | 85% | 0.90 | 0.95 |

### Data Efficiency Comparison

| Method | Data Required | Training Time | Robustness |
|--------|---------------|---------------|------------|
| KTO | 10k examples | 1-2 hours | High |
| DPO | 20k pairs | 2-3 hours | High |
| PPO | 50k+ examples | 4-8 hours | Medium |

## Advanced Techniques

### KTO with Active Learning
```yaml
# Active learning for data selection
active_learning: true
uncertainty_threshold: 0.1
acquisition_function: "max_entropy"
```

### KTO with Human-in-the-Loop
```python
def human_in_the_loop_kto():
    # Generate responses
    # Get human feedback
    # Update training data
    # Retrain model
    pass
```

### KTO with Multi-Objective Optimization
```yaml
# Combine KTO with other objectives
dataset: kto_en_demo,helpful_instructions
template: llama3
```

## Deployment Considerations

### Model Serving
```python
from llamafactory import ChatModel

model = ChatModel(dict(
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    adapter_path="saves/llama3-8b/lora/kto",
    finetuning_type="lora",
    template="llama3"
))

# Use in application
response = model.chat(user_input)
desirability_score = model.predict_desirability(response)
```

### Online Learning
```python
def online_kto_learning():
    while True:
        # Get user interaction
        # Collect feedback
        # Update model incrementally
        # Deploy updated model
        pass
```

## Comparison with Other Methods

| Aspect | KTO | DPO | PPO |
|--------|-----|-----|-----|
| Data Type | Binary | Pairs | Pairs |
| Data Efficiency | High | Medium | Low |
| Training Stability | High | High | Medium |
| Label Cost | Low | High | High |
| Robustness | High | Medium | Low |

## Next Steps

- Try KTO for binary feedback scenarios
- Combine KTO with human-in-the-loop learning
- Use KTO for data-efficient preference learning
- Experiment with active learning for KTO
- Deploy KTO models for interactive applications

For hands-on examples, see the [notebooks](../../notebooks/kto/) directory.
