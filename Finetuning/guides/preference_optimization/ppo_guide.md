# PPO (Proximal Policy Optimization) Guide

## Overview

PPO (Proximal Policy Optimization) is a reinforcement learning algorithm that uses a learned reward model to optimize language model policies through iterative policy updates.

## How PPO Works

PPO training involves three main components:

1. **Reward Modeling**: Train a reward model on preference data
2. **Policy Optimization**: Use PPO to optimize the policy using rewards
3. **Iterative Training**: Alternate between reward model and policy updates

### PPO Algorithm

PPO optimizes the following objective:

```
L(θ) = E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]
```

Where:
- `r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)` is the probability ratio
- `A_t` is the advantage estimate
- `ε` is the clipping parameter

### Training Process

1. **Generate responses** using current policy
2. **Score responses** using reward model
3. **Compute advantages** (rewards - baseline)
4. **Update policy** using PPO objective
5. **Repeat** for multiple epochs

## Advantages of PPO

- **Stable Training**: Designed to prevent large policy updates
- **Sample Efficient**: Reuses generated data for multiple updates
- **Flexible Rewards**: Can incorporate various reward signals
- **Proven Method**: Well-studied and widely used
- **Robust**: Handles noisy reward signals well

## When to Use PPO

- **Complex reward signals** need to be learned
- **High-quality alignment** is required
- **Research and experimentation** with RL methods
- **Production systems** requiring stable training
- **Multi-objective optimization** scenarios

## Configuration Parameters

### Core PPO Parameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|---------|
| `ppo_epochs` | PPO training epochs per batch | 2-8 | More = stable, slower |
| `ppo_batch_size` | Batch size for PPO updates | 1-4 | Larger = more stable |
| `ppo_target` | Target KL divergence | 4.0-8.0 | Controls update size |
| `ppo_whiten_rewards` | Reward normalization | true/false | Better for different scales |

### Advanced Parameters

```yaml
ppo_clip: 0.2  # PPO clipping parameter
ppo_value_clip: 0.2  # Value function clipping
ppo_gamma: 1.0  # Discount factor
ppo_lambda: 0.95  # GAE lambda
ppo_init_kl_coef: 0.2  # Initial KL coefficient
ppo_adaptive_kl_coef: true  # Adaptive KL coefficient
```

## Basic Configuration

### Step 1: Reward Model Training

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: rm  # Reward modeling stage
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: dpo_en_demo  # Preference data
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/lora/reward
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

### Step 2: PPO Training

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: ppo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
reward_model: saves/llama3-8b/lora/reward  # Path to reward model

### dataset
dataset: dpo_en_demo  # Preference data for generation
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/lora/ppo
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

### ppo
ppo_epochs: 4
ppo_batch_size: 2
ppo_target: 6.0
ppo_whiten_rewards: true
ppo_clip: 0.2
ppo_value_clip: 0.2
ppo_gamma: 1.0
ppo_lambda: 0.95
ppo_init_kl_coef: 0.2
ppo_adaptive_kl_coef: true
```

## Training Scripts

### Step 1: Train Reward Model
```bash
python src/train.py examples/train_lora/llama3_lora_reward.yaml
```

### Step 2: Train PPO Policy
```bash
python src/train.py examples/train_lora/llama3_lora_ppo.yaml
```

### Complete PPO Pipeline
```bash
# Train reward model
python src/train.py examples/train_lora/llama3_lora_reward.yaml

# Train PPO policy
python src/train.py examples/train_lora/llama3_lora_ppo.yaml \
  --reward_model saves/llama3-8b/lora/reward
```

## Model Loading and Inference

### Loading PPO Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
ppo_model_path = "saves/llama3-8b/lora/ppo"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, ppo_model_path)

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
    adapter_path="saves/llama3-8b/lora/ppo",
    finetuning_type="lora",
    template="llama3"
))

response = model.chat("What are the benefits of renewable energy?")
print(response)
```

## Hyperparameter Tuning

### PPO Target KL
```yaml
# Conservative (stable, slower learning)
ppo_target: 4.0

# Moderate (balanced)
ppo_target: 6.0

# Aggressive (faster learning, less stable)
ppo_target: 8.0
```

### PPO Epochs
```yaml
# Fewer epochs (faster, less stable)
ppo_epochs: 2

# Standard epochs (balanced)
ppo_epochs: 4

# More epochs (slower, more stable)
ppo_epochs: 8
```

### Learning Rate
```yaml
# Conservative
learning_rate: 1.0e-6
ppo_init_kl_coef: 0.1

# Standard
learning_rate: 5.0e-6
ppo_init_kl_coef: 0.2

# Higher
learning_rate: 1.0e-5
ppo_init_kl_coef: 0.3
```

## Advanced PPO Configurations

### PPO with LoRA+
```yaml
### LoRA+ configuration
finetuning_type: lora
lora_rank: 8
lora_alpha: 32
lora_dropout: 0.1
lora_target: all
loraplus_lr_ratio: 16
ppo_target: 6.0
```

### PPO with QLoRA
```yaml
### QLoRA + PPO
finetuning_type: lora
quantization_bit: 4
quantization_type: nf4
double_quantization: true
lora_rank: 64
lora_alpha: 128
ppo_target: 6.0
```

### PPO with Custom Rewards
```yaml
# Multiple reward models
reward_model: saves/llama3-8b/lora/reward
reward_weights: 0.7, 0.3  # Weights for different rewards
```

## Reward Modeling

### Reward Model Configuration

```yaml
### Reward model training
stage: rm
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

# Use preference data
dataset: dpo_en_demo
template: llama3
```

### Reward Model Evaluation

```python
def evaluate_reward_model(reward_model, test_pairs):
    accuracies = []
    for chosen, rejected in test_pairs:
        score_chosen = reward_model.score(chosen)
        score_rejected = reward_model.score(rejected)
        accuracy = 1 if score_chosen > score_rejected else 0
        accuracies.append(accuracy)
    return sum(accuracies) / len(accuracies)
```

## Evaluation

### PPO Model Evaluation
```python
from llamafactory.eval import evaluate_ppo_model

results = evaluate_ppo_model(
    model_path="saves/llama3-8b/lora/ppo",
    reward_model_path="saves/llama3-8b/lora/reward",
    eval_dataset="ultrafeedback",
    metrics=["win_rate", "reward_score", "kl_divergence"]
)
```

### Manual Evaluation
```python
def evaluate_ppo_model(model, reward_model, test_prompts):
    results = []
    for prompt in test_prompts:
        response = model.generate(prompt)
        reward = reward_model.score(response)
        results.append({
            "prompt": prompt,
            "response": response,
            "reward": reward
        })
    return results
```

### Metrics to Track

- **Reward Score**: Average reward from reward model
- **KL Divergence**: Deviation from reference policy
- **Win Rate**: Against baseline models
- **Stability**: Training stability metrics
- **Safety**: Safety violation rates

## Best Practices

### 1. Reward Model Quality
- Use high-quality preference data
- Train reward model thoroughly
- Validate reward model performance

### 2. PPO Stability
```yaml
# Conservative PPO settings
ppo_epochs: 4
ppo_batch_size: 2
ppo_target: 6.0
ppo_clip: 0.2
ppo_adaptive_kl_coef: true
```

### 3. Learning Rate Scheduling
```yaml
# Use adaptive KL coefficient
ppo_init_kl_coef: 0.2
ppo_adaptive_kl_coef: true
```

### 4. Reward Normalization
```yaml
# Normalize rewards for stability
ppo_whiten_rewards: true
```

### 5. Monitoring
- Track reward scores
- Monitor KL divergence
- Check policy stability

## Troubleshooting

### Common Issues

1. **Training Instability**
   - Reduce ppo_target
   - Lower learning rate
   - Use adaptive KL coefficient

2. **Reward Hacking**
   - Improve reward model
   - Add regularization
   - Use multiple reward signals

3. **Poor Performance**
   - Train reward model longer
   - Increase PPO epochs
   - Use better preference data

4. **Memory Issues**
   - Reduce batch sizes
   - Use QLoRA
   - Enable gradient checkpointing

### Debugging PPO Training

```python
def debug_ppo_training():
    # Monitor reward distributions
    # Check KL divergence trends
    # Validate advantage estimates
    # Track policy updates
    pass
```

## Performance Benchmarks

### Performance Comparison

| Model | UltraFeedback Win Rate | Reward Score | Training Time |
|-------|----------------------|--------------|---------------|
| SFT Baseline | 45% | 0.2 | 1 hour |
| DPO | 68% | 0.4 | 2 hours |
| PPO (4 epochs) | 72% | 0.5 | 6 hours |
| PPO (8 epochs) | 75% | 0.6 | 10 hours |

### Computational Requirements

| Method | GPU Memory | Training Time | Stability |
|--------|------------|---------------|-----------|
| DPO | 8-16GB | 1-2 hours | High |
| PPO (basic) | 16-32GB | 4-8 hours | Medium |
| PPO (advanced) | 32-64GB | 8-12 hours | Medium |

## Advanced Techniques

### Multi-Reward PPO
```yaml
# Use multiple reward models
reward_model: saves/llama3-8b/lora/reward1,saves/llama3-8b/lora/reward2
reward_weights: 0.7, 0.3
```

### PPO with Safety Rewards
```yaml
# Combine helpfulness and safety
reward_model: saves/llama3-8b/lora/reward_helpful,saves/llama3-8b/lora/reward_safe
reward_weights: 0.8, 0.2
```

### PPO with Human Feedback
```python
def ppo_with_human_feedback():
    # Generate responses
    # Get human ratings
    # Update reward model
    # Continue PPO training
    pass
```

## Deployment Considerations

### Model Serving
```python
from llamafactory import ChatModel

model = ChatModel(dict(
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    adapter_path="saves/llama3-8b/lora/ppo",
    finetuning_type="lora",
    template="llama3"
))

# Use in application
response = model.chat(user_input)
```

### Online PPO
```python
def online_ppo_deployment():
    while True:
        # Get user interaction
        # Collect implicit feedback
        # Update policy incrementally
        # Deploy updated model
        pass
```

## Comparison with Other Methods

| Aspect | PPO | DPO | KTO |
|--------|-----|-----|-----|
| Training Stability | Medium | High | High |
| Data Efficiency | Low | Medium | High |
| Reward Quality | High | Medium | Medium |
| Computational Cost | High | Low | Low |
| Flexibility | High | Medium | Low |

## Next Steps

- Try advanced PPO variants
- Experiment with multi-reward optimization
- Use PPO for complex alignment tasks
- Combine PPO with other techniques
- Deploy PPO models in production

For hands-on examples, see the [notebooks](../../notebooks/ppo/) directory.
