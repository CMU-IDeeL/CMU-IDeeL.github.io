import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# suppress OMP warning
os.environ["OMP_NUM_THREADS"] = "1"

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train():
    # fetch environment variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # initialize process group
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    if rank == 0:
        datasets.MNIST("./data", train=True, download=True)
    dist.barrier()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST("./data", train=True, download=False, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=1024, sampler=sampler, num_workers=4, pin_memory=True)

    model = SimpleCNN().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().to(local_rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    dist.barrier()
    start = time.time()

    for epoch in range(5):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        
        for images, labels in loader:
            images, labels = images.to(local_rank), labels.to(local_rank)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # aggregate loss for logging
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            epoch_loss += loss.item() / world_size

        if rank == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")

    dist.barrier()
    if rank == 0:
        print(f"Total DDP Training Time: {time.time() - start:.2f}s")

    dist.destroy_process_group()

if __name__ == "__main__":
    train()
