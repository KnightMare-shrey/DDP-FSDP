# Distributed Data Parallel (DDP) Training in PyTorch

This repository demonstrates how to use **PyTorch Distributed Data Parallel (DDP)** for efficient multi-GPU training. DDP is the recommended way to scale training across multiple GPUs and multiple nodes in PyTorch.

## Overview

PyTorch's `DistributedDataParallel` (DDP) synchronizes gradients across processes during the backward pass, enabling faster and more scalable training compared to `DataParallel`.

### Benefits of DDP

- Better GPU utilization
- Faster training performance
- Scales across multiple GPUs and nodes
- Reduced communication overhead
- Officially recommended by PyTorch

---

## Requirements

- Python 3.8+
- PyTorch 2.x
- CUDA-enabled GPUs (for GPU training)

Install dependencies:

```bash
pip install torch torchvision torchaudio
```

Verify installation:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

---

## Project Structure

```text
.
├── train.py
├── dataset.py
├── model.py
├── requirements.txt
└── README.md
```

---

## DDP Workflow

1. Initialize the process group.
2. Assign one GPU per process.
3. Create model and move it to the local GPU.
4. Wrap the model with `DistributedDataParallel`.
5. Use `DistributedSampler` for dataset sharding.
6. Train normally.
7. Destroy the process group after training.

---

## Example DDP Setup

### Initialize Distributed Environment

```python
import os
import torch.distributed as dist

def setup():
    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )

def cleanup():
    dist.destroy_process_group()
```

---

### Create Model

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

---

### Distributed Sampler

```python
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

sampler = DistributedSampler(
    dataset,
    shuffle=True
)

loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler
)
```

---

### Training Loop

```python
for epoch in range(num_epochs):

    sampler.set_epoch(epoch)

    for images, labels in loader:
        images = images.to(local_rank)
        labels = labels.to(local_rank)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
```

---

## Running DDP

### Single Node, Multiple GPUs

Example with 4 GPUs:

```bash
torchrun \
  --nproc_per_node=4 \
  train.py
```

Alternative:

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

---

## Accessing Rank Information

```python
import os

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
```

Definitions:

| Variable | Description |
|-----------|-------------|
| LOCAL_RANK | GPU index on current machine |
| RANK | Global process ID |
| WORLD_SIZE | Total number of processes |


---

## Checkpointing

Only save checkpoints from rank 0:

```python
if dist.get_rank() == 0:
    torch.save(model.module.state_dict(), "model.pt")
```

Loading:

```python
model.load_state_dict(
    torch.load("model.pt", map_location="cpu")
)
```

---

## Common Backends

| Backend | Usage |
|----------|--------|
| NCCL | Recommended for NVIDIA GPUs |
| GLOO | CPU training and debugging |
| MPI | HPC environments |

Example:

```python
dist.init_process_group(
    backend="nccl"
)
```

---

## Best Practices

### Use NCCL for GPU Training

```python
backend = "nccl"
```

### Save Only on Rank 0

```python
if rank == 0:
    save_checkpoint()
```

### Set Epoch for DistributedSampler

```python
sampler.set_epoch(epoch)
```

### Avoid Printing from Every Process

```python
if rank == 0:
    print(metrics)
```

---

## Debugging

Enable distributed debugging:

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL \
torchrun --nproc_per_node=4 train.py
```

NCCL debugging:

```bash
NCCL_DEBUG=INFO \
torchrun --nproc_per_node=4 train.py
```

---

## Performance Tips

- Increase batch size as GPU count grows.
- Use mixed precision (`torch.cuda.amp` or `torch.amp`).
- Use pinned memory in DataLoader.
- Minimize CPU-GPU synchronization.
- Use gradient accumulation when memory is limited.

Example:

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

with autocast(device_type="cuda"):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

---

## Multi-Node Training

Node 1:

```bash
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=10.0.0.1 \
  --master_port=29500 \
  train.py
```

Node 2:

```bash
torchrun \
  --nnodes=2 \
  --node_rank=1 \
  --nproc_per_node=8 \
  --master_addr=10.0.0.1 \
  --master_port=29500 \
  train.py
```

---

## References

- PyTorch Distributed Data Parallel (DDP)
- PyTorch Distributed Communication Package
- torchrun Documentation

---

## License

MIT License
