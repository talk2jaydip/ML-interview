# Ray Guide

## Overview

Ray is a distributed computing framework that provides simple APIs for scaling Python applications, including machine learning training, across multiple nodes with minimal configuration.

## Ray Components

### Ray Core
- **Distributed Execution**: Run Python functions on remote workers
- **Shared Memory**: Efficient data sharing between processes
- **Fault Tolerance**: Automatic recovery from failures

### Ray Train
- **Training APIs**: Simple distributed training interfaces
- **Checkpointing**: Automatic model checkpointing
- **Metrics**: Built-in metrics collection

### Ray Tune
- **Hyperparameter Tuning**: Automated hyperparameter optimization
- **Early Stopping**: Resource-efficient early stopping
- **Experiment Tracking**: Built-in experiment management

## Configuration

### Basic Ray Configuration

```python
import ray
from ray import train

# Initialize Ray
ray.init()

# Define training function
def train_func(config):
    # Training code here
    pass

# Run distributed training
trainer = train.Trainer(
    backend="torch",
    num_workers=4,
    resources_per_worker={"CPU": 2, "GPU": 1}
)
trainer.start()
results = trainer.run(train_func)
trainer.shutdown()
```

### Ray with LLaMA-Factory

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/ray/sft
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

### ray
ray_num_workers: 4
ray_resources_per_worker: {"CPU": 2, "GPU": 1}
ray_placement_group: "PACK"
ray_checkpoint_freq: 100
```

## Usage Patterns

### Single Node Ray

```bash
# Start Ray locally
ray start --head

# Run training
python src/train.py config.yaml

# Stop Ray
ray stop
```

### Multi-Node Ray

```bash
# On head node
ray start --head --port=6379

# On worker nodes
ray start --address=head_node_ip:6379

# Run distributed training
python src/train.py config.yaml
```

## Hardware Requirements

### Single Node Requirements

| CPUs | GPUs | Memory | Storage | Network |
|------|------|--------|---------|---------|
| 16 | 1-4 | 64GB+ | 100GB+ | N/A |
| 32 | 4-8 | 128GB+ | 500GB+ | N/A |

### Multi-Node Requirements

| Nodes | CPUs/Node | GPUs/Node | Network | Storage |
|-------|-----------|-----------|---------|---------|
| 2 | 16 | 4 | 10Gbps+ | Shared |
| 4 | 32 | 8 | 25Gbps+ | Shared |
| 8 | 64 | 8 | 100Gbps+ | Shared |

## Performance Optimization

### Resource Configuration

```python
# Optimal resource allocation
def configure_ray_resources(num_nodes, gpus_per_node):
    ray.init(
        num_cpus=num_nodes * 32,
        num_gpus=num_nodes * gpus_per_node,
        object_store_memory=200 * 1024 * 1024 * 1024  # 200GB
    )
```

### Placement Groups

```python
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Create placement group
pg = placement_group([{"CPU": 2, "GPU": 1}])
ray.get(pg.ready())

# Use placement group
@ray.remote
def train_worker():
    # Worker code
    pass

# Schedule on placement group
train_worker.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(pg)
).remote()
```

### Checkpointing

```python
from ray import train

# Automatic checkpointing
def train_func(config):
    checkpoint = train.get_checkpoint()
    if checkpoint:
        # Restore from checkpoint
        pass

    # Save checkpoint
    train.report({"accuracy": acc}, checkpoint=checkpoint_object)
```

## Monitoring and Debugging

### Ray Dashboard

```bash
# Start Ray dashboard
ray start --head --include-dashboard=True

# Access dashboard at http://localhost:8265
```

### Ray Monitoring

```python
def monitor_ray():
    # Check cluster status
    print(ray.cluster_resources())

    # Monitor workers
    workers = ray.nodes()
    for worker in workers:
        print(worker)

    # Check object store
    print(ray.object_store_memory())
```

### Common Issues

1. **Resource Deadlocks**
   - Use placement groups
   - Set resource limits
   - Monitor resource usage

2. **Network Issues**
   - Check firewall settings
   - Verify network connectivity
   - Use appropriate network interfaces

3. **Memory Issues**
   - Set object store limits
   - Use streaming for large data
   - Monitor memory usage

## Best Practices

### 1. Resource Management

```python
# Proper resource allocation
ray.init(
    num_cpus=32,
    num_gpus=8,
    object_store_memory=100 * 1024 * 1024 * 1024,
    _memory=200 * 1024 * 1024 * 1024
)
```

### 2. Fault Tolerance

```python
# Handle failures gracefully
@ray.remote(max_retries=3, retry_exceptions=[Exception])
def robust_train():
    try:
        # Training code
        pass
    except Exception as e:
        # Handle failure
        pass
```

### 3. Data Loading

```python
# Efficient data loading
@ray.remote
def load_data():
    # Load data in parallel
    return dataset

# Use multiple workers
data_refs = [load_data.remote() for _ in range(4)]
datasets = ray.get(data_refs)
```

## Advanced Features

### Ray Tune Integration

```python
from ray import tune

def trainable(config):
    # Training function for hyperparameter tuning
    pass

# Hyperparameter search
analysis = tune.run(
    trainable,
    config={
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32, 64])
    },
    num_samples=10,
    resources_per_trial={"cpu": 2, "gpu": 1}
)
```

### Ray Serve Integration

```python
from ray import serve

@serve.deployment
class ModelDeployment:
    def __init__(self):
        # Load model
        pass

    def __call__(self, request):
        # Inference
        return response

# Deploy model
serve.run(ModelDeployment.bind())
```

### Custom Training Loop

```python
import ray
from ray.train import Trainer

class CustomTrainer(Trainer):
    def training_loop(self):
        # Custom training logic
        while not self.should_stop():
            # Training step
            pass
```

## Multi-Node Setup

### Cluster Configuration

```yaml
# Ray cluster configuration
cluster:
  head_node:
    instance_type: m5.2xlarge
    resources: {"CPU": 8, "GPU": 1}

  worker_nodes:
    instance_type: p3.8xlarge
    resources: {"CPU": 32, "GPU": 4}
    min_workers: 3
    max_workers: 10
```

### Autoscaling

```python
# Enable autoscaling
ray.autoscaler.sdk.request_resources(
    bundles=[{"CPU": 2, "GPU": 1} for _ in range(10)]
)
```

### Node Management

```bash
# Start Ray cluster on head node
ray up cluster.yaml

# Monitor cluster
ray exec cluster.yaml "top"

# Scale cluster
ray up cluster.yaml --max-workers 5

# Stop cluster
ray down cluster.yaml
```

## Performance Benchmarks

### Scaling Efficiency

| Nodes | Workers | Model Size | Training Time | Efficiency |
|-------|---------|------------|---------------|------------|
| 1 | 4 | 7B | 2h | 100% |
| 2 | 8 | 7B | 1.1h | 91% |
| 4 | 16 | 7B | 0.6h | 83% |
| 1 | 4 | 30B | 8h | 100% |
| 4 | 16 | 30B | 2.2h | 91% |

### Resource Utilization

| Component | CPU Usage | GPU Usage | Memory Usage | Network Usage |
|-----------|-----------|-----------|-------------|---------------|
| Raylet | 10% | 0% | 1GB | Low |
| Workers | 80% | 90% | 8GB | Medium |
| Object Store | 5% | 0% | 50GB | High |
| GCS | 5% | 0% | 2GB | Medium |

## Comparison with Other Frameworks

| Aspect | Ray | DeepSpeed | FSDP | Winner |
|--------|-----|-----------|------|--------|
| Ease of Use | Excellent | Good | Good | Ray |
| Scalability | Very Good | Excellent | Very Good | DeepSpeed |
| Fault Tolerance | Excellent | Good | Good | Ray |
| Multi-Framework | Yes | No | No | Ray |
| Development Speed | Fast | Medium | Medium | Ray |

## Next Steps

After mastering Ray:
- Scale to hundreds of nodes
- Use Ray for hyperparameter tuning
- Deploy models with Ray Serve
- Integrate with other Ray libraries
- Optimize for specific workloads

For hands-on examples, see the [notebooks](../../notebooks/distributed_training/) directory.
