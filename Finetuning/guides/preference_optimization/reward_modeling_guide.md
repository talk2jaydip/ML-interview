# Reward Modeling Guide

## Overview

Reward modeling trains a separate model to predict human preferences, which serves as the foundation for reinforcement learning-based preference optimization methods like PPO.

## How Reward Modeling Works

1. **Data Collection**: Gather human preference data (chosen vs rejected responses)
2. **Model Training**: Train a model to predict which response is preferred
3. **Reward Scoring**: Use the reward model to score generated responses
4. **Policy Optimization**: Use rewards to optimize the policy (e.g., with PPO)

### Mathematical Foundation

Reward modeling typically optimizes:

```
L(θ) = -E[log p_θ(chosen | prompt) + log (1 - p_θ(rejected | prompt))]
```

Where:
- `p_θ` is the reward model probability
- `chosen` and `rejected` are preferred and dispreferred responses
- The model learns to assign higher scores to preferred responses

## Types of Reward Models

### 1. Bradley-Terry Model
```python
def bradley_terry_reward(chosen_score, rejected_score):
    return chosen_score - rejected_score
```

### 2. Pairwise Classification Model
```python
def pairwise_classification_reward(chosen_score, rejected_score):
    return sigmoid(chosen_score - rejected_score)
```

### 3. Regression Model
```python
def regression_reward(score):
    return score  # Direct score prediction
```

## Advantages of Reward Modeling

- **Explicit Rewards**: Clear optimization objective
- **Interpretable**: Reward scores are meaningful
- **Reusable**: Can be used with different policies
- **Flexible**: Can incorporate various signals
- **Scalable**: Can be trained on large datasets

## When to Use Reward Modeling

- **RL-based optimization** (PPO, etc.)
- **Multi-objective optimization** scenarios
- **Research and analysis** of preferences
- **Production systems** requiring interpretable rewards
- **Iterative improvement** of alignment

## Configuration Parameters

### Basic Configuration

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

### Advanced Parameters

```yaml
# Reward model specific settings
reward_model_type: pairwise  # pairwise, regression, bradley_terry
reward_scale: 1.0  # Scale rewards
reward_clip: 10.0  # Clip extreme rewards
reward_normalize: true  # Normalize rewards
```

## Dataset Format

### Standard Reward Modeling Format

```json
[
  {
    "prompt": "Explain quantum computing",
    "chosen": "Quantum computing is a type of computing...",
    "rejected": "Quantum computing is magic..."
  }
]
```

### Alternative Formats

```json
# Format with multiple responses
{
  "prompt": "Write a helpful response",
  "responses": [
    {"text": "Response 1", "score": 1.0},
    {"text": "Response 2", "score": -1.0}
  ]
}

# Format with ranking
{
  "prompt": "Compare these approaches",
  "candidates": [
    {"text": "Approach A", "rank": 1},
    {"text": "Approach B", "rank": 2},
    {"text": "Approach C", "rank": 3}
  ]
}
```

## Training Scripts

### Basic Reward Model Training
```bash
python src/train.py examples/train_lora/llama3_lora_reward.yaml
```

### Reward Model with Custom Parameters
```bash
python src/train.py examples/train_lora/llama3_lora_reward.yaml \
  --learning_rate 2.0e-5 \
  --num_train_epochs 5
```

### Multi-GPU Reward Model Training
```bash
torchrun --nproc_per_node=2 src/train.py examples/train_lora/llama3_lora_reward.yaml
```

## Model Loading and Inference

### Loading Reward Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
reward_model_path = "saves/llama3-8b/lora/reward"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, reward_model_path)

def get_reward(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    # Extract reward score from model outputs
    return outputs.logits.mean().item()
```

### Using LLaMA-Factory Reward Model Interface
```python
from llamafactory import RewardModel

reward_model = RewardModel(dict(
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    adapter_path="saves/llama3-8b/lora/reward",
    finetuning_type="lora",
    template="llama3"
))

# Score responses
responses = ["Good response", "Bad response"]
scores = reward_model.score(responses)
print(scores)  # Higher score for better response
```

## Evaluation

### Reward Model Evaluation
```python
from llamafactory.eval import evaluate_reward_model

results = evaluate_reward_model(
    model_path="saves/llama3-8b/lora/reward",
    eval_dataset="ultrafeedback",
    metrics=["accuracy", "correlation", "calibration"]
)
```

### Manual Evaluation
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

### Metrics to Track

- **Accuracy**: Correct preference prediction
- **Correlation**: Correlation with human judgments
- **Calibration**: How well scores match actual preferences
- **Robustness**: Performance on different data distributions

## Hyperparameter Tuning

### Learning Rate Selection

```yaml
# Conservative
learning_rate: 5.0e-6
warmup_ratio: 0.2

# Standard
learning_rate: 1.0e-5
warmup_ratio: 0.1

# Higher
learning_rate: 2.0e-5
warmup_ratio: 0.05
```

### Model Architecture

```yaml
# Use LoRA for efficiency
finetuning_type: lora
lora_rank: 8
lora_target: all

# Or full fine-tuning for better quality
finetuning_type: full
```

### Training Parameters

```yaml
# Batch size and accumulation
per_device_train_batch_size: 2
gradient_accumulation_steps: 4

# Number of epochs
num_train_epochs: 3

# Regularization
weight_decay: 0.01
```

## Advanced Reward Modeling

### Multi-Objective Reward Models

```yaml
# Train reward model for multiple objectives
dataset: helpful_preferences,safety_preferences
template: llama3

# Use different heads for different objectives
reward_model_heads: 2
```

### Reward Model Ensembles

```python
def ensemble_reward_models(models, response):
    scores = [model.score(response) for model in models]
    return sum(scores) / len(scores)
```

### Reward Model Distillation

```yaml
# Distill large reward model to smaller one
teacher_model: saves/llama3-8b/lora/reward_large
student_model: meta-llama/Meta-Llama-3-8B-Instruct
distillation_temperature: 2.0
```

## Integration with PPO

### Complete RLHF Pipeline

```python
def rlhf_pipeline():
    # 1. Train reward model
    reward_model = train_reward_model(preference_data)

    # 2. Initialize policy
    policy = initialize_policy()

    # 3. PPO training loop
    for iteration in range(max_iterations):
        # Generate responses
        responses = generate_responses(policy, prompts)

        # Score responses
        rewards = score_responses(reward_model, responses)

        # Update policy
        policy = update_policy_ppo(policy, responses, rewards)

    return policy
```

### Configuration for RLHF

```yaml
# Step 1: Reward Model
stage: rm
output_dir: saves/llama3-8b/lora/reward

# Step 2: PPO
stage: ppo
reward_model: saves/llama3-8b/lora/reward
output_dir: saves/llama3-8b/lora/ppo
```

## Best Practices

### 1. Data Quality
- Use diverse preference pairs
- Ensure clear preference signals
- Balance chosen/rejected quality

### 2. Model Selection
- Use LoRA for efficiency
- Consider model size vs quality trade-offs
- Use appropriate architectures for reward prediction

### 3. Training Stability
```yaml
# Conservative settings
learning_rate: 1.0e-5
warmup_ratio: 0.2
num_train_epochs: 3
weight_decay: 0.01
```

### 4. Evaluation
- Use multiple evaluation datasets
- Check correlation with human judgments
- Validate on out-of-distribution data

### 5. Reward Calibration
```python
def calibrate_rewards(rewards, target_mean=0, target_std=1):
    return (rewards - rewards.mean()) / rewards.std() * target_std + target_mean
```

## Troubleshooting

### Common Issues

1. **Poor Reward Quality**
   - Use higher quality preference data
   - Train for more epochs
   - Try different model architectures

2. **Reward Hacking**
   - Add regularization
   - Use diverse training data
   - Monitor for unintended optimization

3. **Training Instability**
   - Reduce learning rate
   - Add gradient clipping
   - Use more warmup steps

4. **Memory Issues**
   - Reduce batch size
   - Use QLoRA
   - Enable gradient checkpointing

### Debugging Tips

```python
def debug_reward_model():
    # Check reward distributions
    # Analyze failure cases
    # Validate data quality
    # Monitor training dynamics
    pass
```

## Performance Benchmarks

### Reward Model Performance

| Model | Preference Accuracy | Correlation | Calibration |
|-------|-------------------|-------------|-------------|
| 7B LoRA | 78% | 0.85 | 0.92 |
| 7B Full | 82% | 0.88 | 0.95 |
| 13B LoRA | 85% | 0.91 | 0.96 |

### Training Efficiency

| Method | Training Time | GPU Memory | Accuracy |
|--------|---------------|------------|----------|
| LoRA | 1-2 hours | 8-16GB | 78% |
| Full | 2-4 hours | 24-48GB | 82% |
| QLoRA | 1-3 hours | 4-8GB | 75% |

## Advanced Techniques

### Reward Model Uncertainty

```python
def uncertainty_aware_reward(reward_model, response):
    # Get multiple predictions
    predictions = [reward_model.score(response) for _ in range(10)]
    mean_reward = sum(predictions) / len(predictions)
    uncertainty = (sum((p - mean_reward) ** 2 for p in predictions) / len(predictions)) ** 0.5
    return mean_reward, uncertainty
```

### Active Learning for Reward Models

```yaml
# Active learning for data selection
active_learning: true
uncertainty_threshold: 0.1
acquisition_function: "max_uncertainty"
```

### Reward Model Explainability

```python
def explain_reward(reward_model, response):
    # Analyze which parts contribute to reward
    # Provide explanations for reward scores
    pass
```

## Deployment Considerations

### Model Serving
```python
from llamafactory import RewardModel

reward_model = RewardModel(dict(
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    adapter_path="saves/llama3-8b/lora/reward",
    finetuning_type="lora",
    template="llama3"
))

# Use in application
responses = ["Response 1", "Response 2"]
scores = reward_model.score(responses)
best_response = responses[scores.index(max(scores))]
```

### Online Reward Learning

```python
def online_reward_learning():
    while True:
        # Collect user preferences
        # Update reward model
        # Improve policy
        pass
```

## Comparison with Other Methods

| Aspect | Reward Modeling | Direct Optimization (DPO) | Human Feedback |
|--------|-----------------|--------------------------|---------------|
| Interpretability | High | Low | High |
| Flexibility | High | Medium | Low |
| Training Cost | Medium | Low | High |
| Accuracy | High | Medium | High |
| Scalability | Medium | High | Low |

## Next Steps

- Use reward models for PPO training
- Combine multiple reward signals
- Distill reward models for efficiency
- Use reward models for model analysis
- Deploy reward models for content filtering

For hands-on examples, see the [notebooks](../../notebooks/reward_modeling/) directory.
