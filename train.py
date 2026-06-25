import os

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

from model import SimpleClassifier
from dataset import get_dataloader


def setup():

    dist.init_process_group(
        backend="nccl"
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return local_rank, rank, world_size


def cleanup():
    dist.destroy_process_group()


def train():
    local_rank, rank, world_size = setup()
    torch.cuda.set_device(local_rank)
    model = SimpleClassifier().to(local_rank)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    dataloader, sampler = get_dataloader(
        batch_size=32,
        rank=rank,
        world_size=world_size,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.CrossEntropyLoss()
    epochs = 5

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        for x, y in dataloader:
            x = x.to(local_rank, non_blocking=True)
            y = y.to(local_rank, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(
                outputs,
                y,
            )

            loss.backward()
            optimizer.step()

        if rank == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {loss.item():.4f}"
            )

    if rank == 0:
        torch.save(
            model.module.state_dict(),
            "model.pt",
        )

    cleanup()

if __name__ == "__main__":
    train()