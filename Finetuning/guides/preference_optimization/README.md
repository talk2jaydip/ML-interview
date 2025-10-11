# Preference Optimization Methods Guide

This comprehensive guide covers preference optimization techniques for fine-tuning language models to align with human preferences and values:

1. **DPO (Direct Preference Optimization)** - Direct optimization of preference data
2. **KTO (Kahneman-Tversky Optimization)** - Learning from binary feedback
3. **PPO (Proximal Policy Optimization)** - Reinforcement learning from human feedback
4. **Reward Modeling** - Training reward models for preference prediction

## Table of Contents

- [Overview](#overview)
- [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
- [KTO (Kahneman-Tversky Optimization)](#kto-kahneman-tversky-optimization)
- [PPO (Proximal Policy Optimization)](#ppo-proximal-policy-optimization)
- [Reward Modeling](#reward-modeling)
- [Dataset Preparation](#dataset-preparation)
- [Evaluation](#evaluation)
- [Best Practices](#best-practices)

## Overview

Preference optimization methods improve model alignment by learning from human preference data rather than just supervised fine-tuning. These methods help models:

- **Follow instructions** more accurately
- **Avoid harmful outputs** and improve safety
- **Generate helpful responses** aligned with human values
- **Handle edge cases** and difficult queries better
- **Balance multiple objectives** in responses

### Key Differences from SFT

| Aspect | Supervised Fine-tuning | Preference Optimization |
|--------|----------------------|------------------------|
| Data Format | Input → Output | Chosen vs Rejected |
| Learning Signal | Direct supervision | Preference signal |
| Stability | Stable | Can be unstable |
| Alignment | Basic | Advanced |
| Safety | Basic | Enhanced |

## DPO (Direct Preference Optimization)

DPO directly optimizes language models to align with human preferences without explicit reward modeling or reinforcement learning.

### How DPO Works

DPO optimizes the policy by maximizing the likelihood ratio between preferred and dispreferred responses:

```
L(θ) = -E[(log σ(β log π_θ(y_w | x) / π_ref(y_w | x) - β log π_θ(y_l | x) / π_ref(y_l | x)))]
```

Where:
- `y_w` is the preferred response
- `y_l` is the dispreferred response
- `π_θ` is the policy being trained
- `π_ref` is the reference policy
- `β` is the hyperparameter controlling deviation from reference

### Key Advantages

- **Stable Training**: More stable than RL-based methods
- **No Reward Model**: Direct optimization without intermediate reward model
- **Efficient**: Faster training than PPO
- **Hyperparameter**: Only needs β hyperparameter

### Configuration

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
pref_loss: sigmoid  # choices: [sigmoid (dpo), orpo, simpo]
pref_ftx: 0.0  # ftx gamma for IPO

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

### DPO Hyperparameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|---------|
| `pref_beta` | Controls deviation from reference | 0.1, 0.5, 1.0 | Higher = stronger preference signal |
| `pref_loss` | Loss function type | sigmoid, orpo, simpo | Different variants |
| `pref_ftx` | IPO ftx gamma | 0.0-1.0 | For IPO variant |

## KTO (Kahneman-Tversky Optimization)

KTO learns from binary feedback signals (desirable vs undesirable) rather than pairwise preferences.

### How KTO Works

KTO uses the Kahneman-Tversky prospect theory to model human preferences from binary labels:

```
L(θ) = -E[KL(π_θ(y | x) || π_ref(y | x)) + λ E[log σ(β(r(x, y) - r_0))]]
```

Where:
- `r(x, y)` is the binary desirability label
- `r_0` is the baseline desirability
- `λ` controls the strength of the KL penalty

### Key Advantages

- **Binary Labels**: Only needs desirable/undesirable labels
- **Efficient**: More data-efficient than pairwise preferences
- **Robust**: Less sensitive to label noise
- **Simple**: Easier to collect training data

### Configuration

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

### KTO Hyperparameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|---------|
| `pref_beta` | Controls deviation from reference | 0.1, 0.5, 1.0 | Higher = stronger signal |

## PPO (Proximal Policy Optimization)

PPO uses reinforcement learning with a learned reward model to optimize language model policies.

### How PPO Works

PPO training involves:
1. **Reward Modeling**: Train reward model on preference data
2. **Policy Training**: Use PPO to optimize policy using reward model
3. **Iterative Process**: Alternate between reward model and policy updates

### Key Advantages

- **Explicit Rewards**: Clear optimization objective
- **Proven Method**: Well-studied RL algorithm
- **Flexible**: Can incorporate various reward signals
- **Stable**: PPO is designed to be stable

### Configuration

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
dataset: dpo_en_demo  # Preference data for reward model training
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
ppo_target: 6.0  # Target KL divergence
ppo_whiten_rewards: true
```

### PPO Hyperparameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|---------|
| `ppo_epochs` | PPO training epochs | 2-8 | More = more stable, slower |
| `ppo_batch_size` | Batch size for PPO | 1-4 | Larger = more stable |
| `ppo_target` | Target KL divergence | 4.0-8.0 | Controls policy updates |
| `ppo_whiten_rewards` | Reward normalization | true/false | Better for different reward scales |

## Reward Modeling

Reward modeling trains a separate model to predict human preferences, which can then be used for RL-based optimization.

### How Reward Modeling Works

1. **Data Collection**: Gather human preference data (chosen vs rejected)
2. **Model Training**: Train a model to predict which response is preferred
3. **Reward Scoring**: Use the reward model to score generated responses
4. **Policy Optimization**: Use rewards to optimize the policy

### Configuration

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

## Dataset Preparation

### DPO Dataset Format

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Write a story about AI"},
      {"from": "gpt", "value": "Once upon a time..."}
    ],
    "chosen": {"from": "gpt", "value": "preferred response"},
    "rejected": {"from": "gpt", "value": "dispreferred response"}
  }
]
```

### KTO Dataset Format

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Write a story"},
      {"from": "gpt", "value": "Once upon a time..."}
    ],
    "kto_tag": true  # true for desirable, false for undesirable
  }
]
```

### Preference Data Sources

- **Human-annotated data**: UltraFeedback, Anthropic HH-RLHF
- **AI-generated preferences**: GPT-4 vs other models
- **Synthetic preferences**: Using strong models to label weaker ones
- **Mixed datasets**: Combining multiple preference sources

## Evaluation

### Automatic Evaluation

```python
from llamafactory.eval import evaluate_model

# Evaluate DPO model
results = evaluate_model(
    model_path="saves/llama3-8b/lora/dpo",
    eval_dataset="ultrafeedback",
    metrics=["accuracy", "win_rate"]
)
```

### Human Evaluation

```python
# A/B testing framework
def compare_responses(response_a, response_b):
    # Human preference collection
    pass
```

### Metrics

- **Win Rate**: Percentage of preferences where model wins
- **Length-normalized Win Rate**: Win rate accounting for response length
- **Helpfulness**: Human-rated helpfulness scores
- **Safety**: Safety violation rates
- **Truthfulness**: Factual accuracy

## Best Practices

### 1. Data Quality
- Use high-quality preference data
- Ensure diverse preference pairs
- Balance chosen/rejected ratios

### 2. Hyperparameter Tuning
- Start with conservative β values (0.1)
- Use learning rate schedules
- Monitor KL divergence for PPO

### 3. Model Selection
- Use LoRA for efficiency
- Consider model size vs quality trade-offs
- Use reference models for stability

### 4. Training Stability
```yaml
# Use conservative settings
pref_beta: 0.1
learning_rate: 5.0e-6
warmup_ratio: 0.2
```

### 5. Evaluation
- Use multiple evaluation datasets
- Include human evaluation
- Monitor for overfitting to training preferences

## Hardware Requirements

| Method | GPU Memory | Training Time | Stability |
|--------|------------|---------------|-----------|
| DPO | 8-16GB | 1-3 hours | High |
| KTO | 8-16GB | 1-2 hours | High |
| PPO | 16-32GB | 4-8 hours | Medium |
| Reward Modeling | 8-16GB | 1-2 hours | High |

## Troubleshooting

### Common Issues

1. **Training Instability**
   - Reduce β/preference strength
   - Use lower learning rates
   - Add more warmup steps

2. **Poor Performance**
   - Use higher quality preference data
   - Increase training epochs
   - Try different preference optimization methods

3. **Memory Issues**
   - Use LoRA instead of full fine-tuning
   - Reduce batch sizes
   - Use gradient checkpointing

4. **Overfitting**
   - Use regularization
   - Add more diverse data
   - Use early stopping

## Next Steps

After preference optimization:
- Deploy aligned models in applications
- Continue with iterative preference learning
- Combine with safety fine-tuning
- Monitor real-world performance

For hands-on examples, see the [notebooks](../notebooks/dpo/) and [notebooks](../notebooks/kto/) directories.
