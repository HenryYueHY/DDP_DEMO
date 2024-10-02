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

def setup(nnode,gpus_per_node):
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = nnode * gpus_per_node + local_rank
    """设置分布式训练环境."""
    os.environ['MASTER_ADDR'] = '10.113.13.77'  # 主节点的IP地址
    os.environ['MASTER_PORT'] = '12355'     # 主节点的通信端口
    dist.init_process_group('nccl', rank=global_rank, world_size=8, init_method='env://')
    torch.cuda.set_device(local_rank)

def cleanup():
    """清理分布式环境."""
    dist.destroy_process_group()

def train(nnode,gpus_per_node):
    """DDP 训练函数."""
    local_rank=int(os.environ['LOCAL_RANK'])
    global_rank=nnode*gpus_per_node+local_rank
    print(f"Running DDP on rank {global_rank} (local rank {local_rank}).")

    # 设置分布式环境


    # 设置模型并封装为 DDP
    model = SimpleModel().to(f'cuda:{local_rank}')
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

        print(f"Rank {local_rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnode", type=int)
    args = parser.parse_args()
    gpus_per_node = torch.cuda.device_count()
    setup(args.nnode,gpus_per_node)
    train(args.nnode,gpus_per_node)

