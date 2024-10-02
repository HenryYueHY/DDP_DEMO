import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 创建模型
model = SimpleModel()

# 如果有多个 GPU，使用 DataParallel 包装模型
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

# 将模型放到 GPU 上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 创建数据集
x = torch.randn(100000, 100)  # 1000 个样本，每个样本 100 维
y = torch.randint(0, 10, (100000,))  # 1000 个标签，分类任务
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(10000):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
