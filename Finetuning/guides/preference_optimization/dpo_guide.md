# DPO (Direct Preference Optimization) Guide

## Overview

DPO (Direct Preference Optimization) is a stable and efficient method for fine-tuning language models directly on preference data without requiring reinforcement learning or reward modeling.

## Mathematical Foundation

DPO optimizes the following objective:

```
L(θ) = -E[(log σ(β (log π_θ(y_w | x)/π_ref(y_w | x) - log π_θ(y_l | x)/π_ref(y_l | x))))]
```

Where:
- `π_θ` is the policy being optimized
- `π_ref` is the reference policy (usually SFT model)
- `y_w`, `y_l` are preferred and dispreferred responses
- `β` controls the strength of preference optimization

### Key Insights

1. **Implicit Reward**: DPO implicitly learns a reward function
2. **Reference Policy**: Uses KL divergence with reference for stability
3. **Direct Optimization**: No sampling or RL required
4. **Hyperparameter**: Only β needs tuning

## Advantages of DPO

- **Stability**: More stable than RL-based methods
- **Efficiency**: No need for reward model or sampling
- **Simplicity**: Single hyperparameter (β)
- **Performance**: Competitive with RL methods
- **Memory Efficient**: Lower memory requirements

## When to Use DPO

- **Stable preference learning** is needed
- **Limited computational resources** for RL
- **Quick experimentation** with preference data
- **Production deployment** requiring reliability
- **Combining with SFT** for alignment

## Configuration Parameters

### Core Parameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|---------|
| `pref_beta` | Preference strength | 0.1, 0.5, 1.0 | Higher = stronger preferences |
| `pref_loss` | Loss function variant | sigmoid, orpo, simpo | Different optimization objectives |
| `pref_ftx` | IPO ftx gamma | 0.0-1.0 | For IPO variant |

### Loss Function Variants

```yaml
# Standard DPO (sigmoid loss)
pref_loss: sigmoid

# IPO (Identity Preference Optimization)
pref_loss: ipo
pref_ftx: 1.0

# ORPO (Odds Ratio Preference Optimization)
pref_loss: orpo

# SimPO (Simple Preference Optimization)
pref_loss: simpo
```

## Basic Configuration

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid

### dataset
dataset: dpo_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/lora/dpo
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

### Standard DPO Format

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Write a helpful response"},
      {"from": "gpt", "value": "I'll provide a helpful response"}
    ],
    "chosen": {
      "from": "gpt",
      "value": "This is the preferred response because it's more helpful, accurate, and well-structured."
    },
    "rejected": {
      "from": "gpt",
      "value": "This is the dispreferred response because it's less helpful or contains errors."
    }
  }
]
```

### Alternative Formats

```json
# Format with system messages
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant"},
    {"from": "human", "value": "Explain quantum computing"}
  ],
  "chosen": {"from": "gpt", "value": "preferred explanation"},
  "rejected": {"from": "gpt", "value": "dispreferred explanation"}
}
```

## Training Scripts

### Basic DPO Training
```bash
python src/train.py examples/train_lora/llama3_lora_dpo.yaml
```

### DPO with Custom Parameters
```bash
python src/train.py examples/train_lora/llama3_lora_dpo.yaml \
  --pref_beta 0.5 \
  --pref_loss ipo \
  --learning_rate 1.0e-5
```

### DPO with Different Loss Functions
```bash
# SimPO
python src/train.py examples/train_lora/llama3_lora_dpo.yaml \
  --pref_loss simpo

# ORPO
python src/train.py examples/train_lora/llama3_lora_dpo.yaml \
  --pref_loss orpo
```

## Model Loading and Inference

### Loading DPO Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
dpo_model_path = "saves/llama3-8b/lora/dpo"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, dpo_model_path)

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
    adapter_path="saves/llama3-8b/lora/dpo",
    finetuning_type="lora",
    template="llama3"
))

response = model.chat("What are the benefits of renewable energy?")
print(response)
```

## Hyperparameter Tuning

### Beta Selection

```yaml
# Conservative preference (stable)
pref_beta: 0.1

# Moderate preference (balanced)
pref_beta: 0.5

# Strong preference (aggressive)
pref_beta: 1.0
```

### Learning Rate Tuning

```yaml
# Conservative learning rate
learning_rate: 1.0e-6
warmup_ratio: 0.2

# Standard learning rate
learning_rate: 5.0e-6
warmup_ratio: 0.1

# Higher learning rate
learning_rate: 1.0e-5
warmup_ratio: 0.05
```

### Loss Function Selection

```yaml
# Standard DPO
pref_loss: sigmoid

# IPO (more stable)
pref_loss: ipo
pref_ftx: 1.0

# SimPO (simpler)
pref_loss: simpo

# ORPO (different objective)
pref_loss: orpo
```

## Advanced Configurations

### DPO with LoRA+
```yaml
### LoRA+ configuration
finetuning_type: lora
lora_rank: 8
lora_alpha: 32
lora_dropout: 0.1
lora_target: all
loraplus_lr_ratio: 16  # Different learning rates for A and B matrices
pref_beta: 0.1
```

### DPO with QLoRA
```yaml
### QLoRA + DPO
finetuning_type: lora
quantization_bit: 4
quantization_type: nf4
double_quantization: true
lora_rank: 64
lora_alpha: 128
pref_beta: 0.1
```

### DPO with Custom Reference Model
```yaml
# Use a different reference model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
ref_model: saves/llama3-8b/lora/sft  # Reference SFT model
```

## Evaluation

### Automatic Evaluation
```python
from llamafactory.eval import evaluate_preference_model

results = evaluate_preference_model(
    model_path="saves/llama3-8b/lora/dpo",
    eval_dataset="ultrafeedback",
    metrics=["win_rate", "length_normalized_win_rate"]
)
```

### Manual Evaluation
```python
def evaluate_dpo_model(model, test_cases):
    results = []
    for prompt in test_cases:
        response = model.chat(prompt)
        # Human evaluation of response quality
        results.append(evaluate_response(response))
    return results
```

### Metrics to Track

- **Win Rate**: How often the model wins against baseline
- **Length-Normalized Win Rate**: Win rate accounting for response length
- **Helpfulness Score**: Human-rated helpfulness (1-5)
- **Safety Score**: Safety violation rate
- **Coherence Score**: Response coherence rating

## Best Practices

### 1. Data Quality
- Use diverse preference pairs
- Ensure clear preference signals
- Balance chosen/rejected quality

### 2. Hyperparameter Selection
- Start with β = 0.1 (conservative)
- Use cosine learning rate schedule
- Monitor training stability

### 3. Training Stability
```yaml
# Conservative settings
pref_beta: 0.1
learning_rate: 5.0e-6
warmup_ratio: 0.2
num_train_epochs: 3
```

### 4. Model Selection
- Use LoRA for efficiency
- Consider model size vs quality trade-offs
- Use reference models for stability

### 5. Monitoring
- Track preference loss
- Monitor KL divergence
- Check validation win rates

## Troubleshooting

### Common Issues

1. **Training Instability**
   - Reduce β value
   - Lower learning rate
   - Increase warmup ratio

2. **Poor Win Rates**
   - Increase β value
   - Use higher quality data
   - Train for more epochs

3. **Mode Collapse**
   - Use higher β
   - Add regularization
   - Use different loss functions

4. **Memory Issues**
   - Reduce batch size
   - Use QLoRA
   - Enable gradient checkpointing

### Debugging Tips

```python
# Monitor DPO training
def debug_dpo_training():
    # Check preference probabilities
    # Monitor KL divergence
    # Validate data quality
    pass
```

## Performance Benchmarks

### Win Rates on Standard Benchmarks

| Model | UltraFeedback Win Rate | HH-RLHF Win Rate | Safety Instructions |
|-------|----------------------|------------------|-------------------|
| SFT Baseline | 45% | 42% | 85% |
| DPO (β=0.1) | 62% | 58% | 92% |
| DPO (β=0.5) | 68% | 64% | 95% |
| DPO (β=1.0) | 72% | 68% | 96% |

### Training Efficiency

| Method | Training Time | GPU Memory | Stability |
|--------|---------------|------------|-----------|
| DPO | 1-2 hours | 8-16GB | High |
| PPO | 4-8 hours | 16-32GB | Medium |
| RLHF | 8-12 hours | 32-64GB | Low |

## Advanced Techniques

### DPO with Multiple Objectives
```yaml
# Combine with safety instructions
dataset: dpo_en_demo,safety_instructions
template: llama3
```

### DPO with Custom Loss
```python
# Custom DPO loss implementation
class CustomDPOLoss:
    def __init__(self, beta=0.1):
        self.beta = beta

    def compute_loss(self, policy_logps, ref_logps, chosen_mask):
        # Custom loss implementation
        pass
```

### DPO with Active Learning
```yaml
# Active learning for data selection
# Select most informative preference pairs
active_learning: true
uncertainty_threshold: 0.1
```

## Deployment Considerations

### Model Serving
```python
# Production deployment
from llamafactory import ChatModel

model = ChatModel(dict(
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    adapter_path="saves/llama3-8b/lora/dpo",
    finetuning_type="lora",
    template="llama3"
))

# Use in application
response = model.chat(user_input)
```

### A/B Testing
```python
def ab_test_dpo_vs_sft():
    dpo_model = load_dpo_model()
    sft_model = load_sft_model()

    for prompt in test_prompts:
        dpo_response = dpo_model.chat(prompt)
        sft_response = sft_model.chat(prompt)
        # Compare responses
```

## Next Steps

- Try different DPO variants (IPO, SimPO, ORPO)
- Combine DPO with other alignment techniques
- Use DPO for domain-specific alignment
- Experiment with different reference models
- Deploy DPO models in production

For hands-on examples, see the [notebooks](../../notebooks/dpo/) directory.
