import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # 设置主节点地址
    os.environ['MASTER_PORT'] = '12355'  # 设置主节点端口号
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    print(f"Running DDP training on rank {rank}.")

    # 初始化分布式环境
    setup(rank, world_size)

    # 创建模型并包装为 DDP 模型
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 创建数据集和数据加载器
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

    # 清理分布式环境
    cleanup()


def main():
    world_size = torch.cuda.device_count()  # 获取 GPU 的数量
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
