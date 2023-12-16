import os
import argparse
from time import perf_counter
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms
from torch.utils.data import Dataset

from dataset import ArtistClassificationDataset
from model import VisionTransformer
from model_inception import iformer_large_384


# Code reference: https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run(rank, world_size, args):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    batch_size = args.batch_size
    num_workers = args.num_workers
    data_path = args.data_path
    epochs = args.epochs
    world_size = args.world_size
    lr = args.lr
    betas = args.betas
    eps = args.eps
    weight_decay = args.weight_decay

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])

    dataset = ArtistClassificationDataset(root=data_path, transform=transform)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset into training and testing sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=val_sampler)

    model = iformer_large_384(pretrained=False)
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_dataloader.sampler.set_epoch(epoch)

        correct = 0
        total = 0
        data_loading_time = 0.0
        training_time = 0.0

        start = data_loading_checkpoint = perf_counter()
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            data_loading_time += (perf_counter() - data_loading_checkpoint)
            training_start_time = perf_counter()

            inputs = inputs.cuda(rank)
            targets = targets.cuda(rank)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            training_time += (perf_counter() - training_start_time)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            data_loading_checkpoint = perf_counter()

        running_time = perf_counter() - start
        accuracy = 100. * correct / total

        if rank == 0:
            print(f"Epoch {epoch}/{epochs} ")
            print(f"Data loading time (sec) is {data_loading_time:.3f}")
            print(f"Training time (sec) is {training_time:.3f}")
            print(f"Running time (sec) is {running_time:.3f}")
            print(f"Accuracy is {accuracy}%")

        scheduler.step()

    cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument("--num_workers", type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--data_path', type=str, default='/scratch/tn2151/DL/images/',
                        help='Path to the training data')
    parser.add_argument("--epochs", type=int, default=30,
                        help='Use the first 4 epochs as warmup')
    parser.add_argument("--world_size", type=int, default=4,
                        help='Number of distributed processes')
    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999),
                        help="Betas for AdamW")
    parser.add_argument("--eps", type=float, default=1e-08,
                        help="Epsilon for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    args = parser.parse_args()

    # The project will run DDP on one node (each node equipped with 8 GPUs),
    # so counting how many GPUs using on the node should be sufficient
    n_gpus = torch.cuda.device_count()
    world_size = args.world_size
    print("World size: ", world_size)
    print("Number of GPUs: ", n_gpus)
    assert n_gpus >= world_size, f"Request more GPUs"

    print(f"Distributed Training with {world_size} CUDA Devices")

    mp.spawn(run,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

