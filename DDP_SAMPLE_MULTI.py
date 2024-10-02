import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def setup(rank, world_size, backend='nccl'):
    """设置分布式训练环境."""
    os.environ['MASTER_ADDR'] = '10.113.13.77'  # 主节点的IP地址
    os.environ['MASTER_PORT'] = '12355'     # 主节点的通信端口
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    """清理分布式环境."""
    dist.destroy_process_group()

def train(rank, world_size, node_rank,gpus_per_node,backend='nccl'):
    """DDP 训练函数."""
    local_rank=rank
    global_rank=node_rank*gpus_per_node+local_rank
    print(f"Running DDP on rank {global_rank} (local rank {local_rank}).")

    # 设置分布式环境
    setup(global_rank, world_size, backend)

    # 设置模型并封装为 DDP
    model = SimpleModel().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    # 创建数据集和分布式采样器
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    # 定义损失函数和优化器
    criterion = nn.MSELoss().to(local_rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(10):
        #print("kaishi")
        sampler.set_epoch(epoch)  # 每个 epoch 保证采样不同的数据
        #print("sampler")
        for batch in dataloader:
            #print("start")
            inputs, targets = batch
            inputs, targets = inputs.to(local_rank), targets.to(local_rank)

            # 前向传播
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

def main():
    """主函数，设置多进程 DDP."""
    world_size = 8  # 假设有 8 张 GPU，跨多个节点
    nodes = 2       # 假设跨 2 个节点
    gpus_per_node = 4  # 每个节点有 4 张 GPU

    # 获取当前节点的 rank 和 local rank
    node_rank = int(os.environ['NODE_RANK'])  # 当前节点的 rank，手动在运行脚本时设置
    # 启动每个进程对应的训练
    mp.spawn(train, args=(world_size,node_rank,gpus_per_node), nprocs=4, join=True)

if __name__ == "__main__":
    main()
