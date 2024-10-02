import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torchvision import datasets, transforms


def setup():
    os.environ['MASTER_ADDR'] = 'localhost'  # 设置主节点地址
    os.environ['MASTER_PORT'] = '12355'  # 设置主节点端口号
    rank=int(os.environ['LOCAL_RANK'])
    # Initializes the default process group
    dist.init_process_group(
        backend='nccl',  # 'nccl' for NVIDIA GPUs, 'gloo' for CPU or cross-platform
        init_method='env://',
        world_size=4,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)
def train():
    rank = int(os.environ['LOCAL_RANK'])


    print(f"Running DDP on rank {rank}.")
    # Create model and move it to the corresponding device
    model = SimpleModel().to(f'cuda:{rank}')

    # Wrap the model in DistributedDataParallel
    ddp_model = DDP(model, device_ids=[rank])


    # Prepare data (using MNIST dataset)
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    # 定义损失函数和优化器
    criterion = nn.MSELoss().to(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(5000):
        sampler.set_epoch(epoch)  # 设置 epoch 保证分布式数据采样的随机性
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(rank), targets.to(rank)

            # 前向传播
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    # Cleanup
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    setup()
    train()

    # Spawn multiple processes, one per GPU
    # torch.multiprocessing.spawn(
    #     train,
    #     args=(world_size,),
    #     nprocs=world_size,
    #     join=True
    # )
