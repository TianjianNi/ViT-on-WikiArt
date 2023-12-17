import os
import argparse
from time import perf_counter
import random
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import Dataset
from dataset import ArtistClassificationDataset
from model import VisionTransformer
from model_inception import iformer_base_384
import torchvision.models as models


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(train_dataloader, model, criterion, optimizer, epoch, total_epoch, rank):
    model.train()
    correct = 0
    total = 0

    start = perf_counter()
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs = inputs.cuda(rank)
        targets = targets.cuda(rank)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    running_time = perf_counter() - start
    accuracy = 100. * correct / total

    if rank == 0:
        print(f"Epoch {epoch}/{total_epoch} ")
        print(f"Running time (sec) is {running_time:.3f}")
        print(f"Training Accuracy is {accuracy}%")


def val(val_dataloader, model, criterion, rank):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_dataloader):
            inputs = inputs.cuda(rank)
            targets = targets.cuda(rank)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total

    if rank == 0:
        print(f"Validation Accuracy is {accuracy}%")


def run(rank, world_size, args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    batch_size = args.batch_size
    num_workers = args.num_workers
    data_path = args.data_path
    epochs = args.epochs
    model_selection = args.model_selection
    train_test_split = args.train_test_split
    lr = args.lr
    betas = args.betas
    eps = args.eps
    weight_decay = args.weight_decay

    print(f"Running DDP on rank {rank} for model {model_selection}.")
    setup(rank, world_size)

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])

    dataset = ArtistClassificationDataset(root=data_path, transform=transform)
    train_size = int(train_test_split * len(dataset))
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

    if model_selection == 'ViT':
        model = VisionTransformer(
            image_size=384,
            patch_size=16,
            in_channels=3,
            n_classes=10,
            embedding_dim=768,
            num_blocks=6,
            num_attention_heads=6,
            mlp_ratio=4.0
        )
    elif model_selection == 'model_ViT':
        model = timm.create_model('vit_base_patch16_384', img_size=384, pretrained=False)
    elif model_selection == 'model_iFormer':
        model = iformer_base_384(pretrained=False)
    elif model_selection == 'ResNet':
        model = models.resnet50(pretrained=False)
        num_classes = 10
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):

        train_dataloader.sampler.set_epoch(epoch)
        train(train_dataloader, model, criterion, optimizer, epoch, epochs, rank)
        val(val_loader, model, criterion, rank)

    cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=9, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument("--num_workers", type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--data_path', type=str, default='/scratch/tn2151/DL/images/',
                        help='Path to the training data')
    parser.add_argument("--epochs", type=int, default=50,
                        help='Number of epochs')
    parser.add_argument("--train_test_split", type=float, default=0.85,
                        help='training data percentage')
    parser.add_argument('--model_selection', type=str, required=True, default='ViT',
                        choices=['ViT', 'model_ViT', 'model_iFormer', 'ResNet'],
                        help='Specify the deep learning, Note that ViT is implemented from scratch '
                             'and model_ViT is from timm official implementation')
    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=0.0008,
                        help="Learning rate")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999),
                        help="Betas for AdamW")
    parser.add_argument("--eps", type=float, default=1e-08,
                        help="Epsilon for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    args = parser.parse_args()

    # The project will run DDP on one node (each node equipped with 4 GPUs),
    # so counting how many GPUs using on the node should be sufficient
    n_gpus = torch.cuda.device_count()
    print("Number of GPUs: ", n_gpus)
    assert n_gpus > 1, f"Request more GPUs!"
    print(f"Distributed Training with {n_gpus} CUDA Devices")

    # Start processes on GPUs
    mp.spawn(run,
             args=(n_gpus, args),
             nprocs=n_gpus,
             join=True)
