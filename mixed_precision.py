import os

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler


def setup():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    return local_rank, rank, world_size


def cleanup():
    dist.destroy_process_group()


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.network(x)


def get_dataloader(rank, world_size, batch_size=32):

    dataset = TensorDataset(
        torch.randn(10000, 10),
        torch.randint(0, 2, (10000,))
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=4,
    )

    return dataloader, sampler


def train():

    local_rank, rank, world_size = setup()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model = SimpleClassifier().to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    dataloader, sampler = get_dataloader(
        rank=rank,
        world_size=world_size,
        batch_size=64,
    )

    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()
    epochs = 5

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for x, y in dataloader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Forward Pass
            with autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                outputs = model(x)
                loss = criterion(outputs, y)

            # Mixed Precision Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)

        if rank == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "f"Loss: {avg_loss:.4f}")

    if rank == 0:
        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "checkpoint.pt",
        )

    cleanup()


if __name__ == "__main__":
    train()