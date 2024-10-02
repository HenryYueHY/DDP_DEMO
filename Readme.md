# Pytorch Distributed Data Parallel

# 1. Basic Concept


During forward propagation, the model is broadcasted from the primary GPU (usually the default logical GPU 0) to all devices. 

The input data is split along the batch dimension and scattered to different devices for forward computation 

(tuple, list, and dict types will be shallow copied. Other types will be shared among different threads and can be corrupted 

if written to during the model's forward pass). After computation, the network outputs are gathered on the primary GPU,

where the loss is subsequently computed. This explains why the primary GPU bears a heavier load — loss calculation always 

happens on the primary GPU, leading to a significantly higher load compared to the other GPUs. During backward propagation, 

the loss is scattered to each device, and each GPU independently performs backpropagation to compute gradients. The 

gradients are then reduced to the primary GPU (i.e., the sum of the gradients from each device is computed and then 

averaged according to the batch size). The model parameters are updated on the primary GPU using backpropagation, and 

the updated parameters are broadcasted  to the other GPUs for the next round of forward propagation, thus enabling parallelism.


## 2.Data Parallel (DP) nn.DataParallel
``` 
torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
#module: your model
#device_ids (list of python:int or torch.device) your gpus
#output_device where you put your results.
#dim batch divide dim.
``` 
Data Parallel is the simplest way to realize multi-gpu training in pytorch

what you are going to do is just 

``` 
import torch.nn as nn

class SimpleModel(nn.Module):
    # Model init
    
model = SimpleModel()
model = nn.DataParallel(model) # Wrap your model
                               # using nn.DataParallel(model)
model.to(device)

def train() #Some training process.
    ...
    #pass
    for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            
    ...
```

Problem: Single process. No model parallel.

## 3. Distributed Data Parallel (DDP)

So the substitute of DP is the Distributed Data Parallel

DDP has two way to be deployed.

### 3.1 One node with multiple gpu
It's basically the same idea.

The difference between DDP and DP is to init_process_group and mp.spawn()
``` 
def main():
    world_size = torch.cuda.device_count()  # get gpu number
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```
```
def train(rank,world_size):
    setup(rank, world_size)
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    sampler = DistributedSampler(dataset)
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    ...
    for epoch in range(5000):
        sampler.set_epoch(epoch)  
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(rank), targets.to(rank)
    
```
``` 
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  - the ip host of the main node
    os.environ['MASTER_PORT'] = '12355'  - the port for TCP communication
    dist.init_process_group("nccl", rank=rank, world_size=world_size) - init process group
    # NCCL is for backend communication protocol other options like gloo 
    # mpi used for cpu only training
    # so rank here stands for which gpu you r using  
    # world_size means how many gpus are involved for this training
``` 

``` 
def cleanup():
    dist.destroy_process_group()
    # This is where you destory all process for safety.
```


### 3.2 Multi nodes with multiple gpus

So for multinode with multi gpus it needs communication between main node and sub-node

Here we are going to first introduce about the global rank and the local rank.

For global rank, you can consider it as the unique id of each process.

And local rank you may consider it as the number of gpu that your current process using.

Simple example:

Assume we have 2 nodes and each node has 4 gpus.

Then we have a world_size = nodes*gpu_per_nodes =2*4=8

Each node will run 4 process:

``` 
Node 0:
Running process with global rank 0 on gpu 0
Running process with global rank 1 on gpu 1
Running process with global rank 2 on gpu 2
Running process with global rank 3 on gpu 3


Node 1:
Running process with global rank 4 on gpu 0
Running process with global rank 5 on gpu 1
Running process with global rank 6 on gpu 2
Running process with global rank 7 on gpu 3
``` 
with that we will go through the code

``` 
def main():
    world_size = 8 
    nodes = 2
    gpus_per_node = 4
    node_rank = int(os.environ['NODE_RANK']) #from env
    mp.spawn(train, args=(world_size,node_rank,gpus_per_node), nprocs=4, join=True)
``` 

``` 
def train(rank, world_size, node_rank,gpus_per_node,backend='nccl'):
    local_rank=rank
    global_rank=node_rank*gpus_per_node+local_rank
    setup(global_rank, world_size, backend)

    model = SimpleModel().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
    ... #Some traing settings 
    for epoch in range(10):
        sampler.set_epoch(epoch)  
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(local_rank), targets.to(local_rank)
            ... # Some training setting.
``` 

``` 
def setup(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '10.113.13.77'  # Main Node IP
    os.environ['MASTER_PORT'] = '12355'     # Main Node Port for TCP
    dist.init_process_group(backend, rank=rank, world_size=world_size)
``` 
Demo using doraemon19 and doraemon20

### 3.3 Start DDP USING CMD

So all these DDP can be init by using torchrun (which is updated Since version 1.9.0 (released on Jun 2021))

By using the torchrun we can assign Env Variable directly from the cmd.

Basically the torchrun receive following parameters.
``` 
torchrun --nnodes= #how many nodes are used.
         --node_rank= #current node number, which starts from 0 and 0 stands for the main node.
         --nproc_per_node= #for each node how many process will be init.
         --master_addr= # ip_address for main node.
         --master_port= # port for tcp connection.
         --standalone #if you are using single node.
         your_python_file.py
         your_args
``` 
A quick Example for torchrun DDP
``` 
CUDA_VISIBLE_DEVICES="0,1,2,3" \
NCCL_DEBUG=INFO \
torchrun \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr=10.113.13.77 \
        --master_port=12355 \
        DDP_SAMPLE_MULTI_ELASTIC.py \
        --nnode 1
``` 

The main difference between the file way and the cmd way in process setup.

``` 
def setup():
    #os.environ['MASTER_ADDR'] = 'localhost'  # No need for now cuz args for torchrun has higher priority
    #os.environ['MASTER_PORT'] = '12355' 
    rank=int(os.environ['LOCAL_RANK'])
    # Initializes the default process group
    dist.init_process_group(
        backend='nccl',  # 'nccl' for NVIDIA GPUs, 'gloo' for CPU or cross-platform
        init_method='env://', # means from env get para.
        world_size=4,
        rank=rank
    )
    torch.cuda.set_device(rank)
    
    
.....



if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    setup()
    train()

``` 



## 4. Model Parallel
``` 
import torch
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cuda:1'))
``` 

## 5. Model Saving in DDP

One important setting for model saving  in ddp is that only Global_rank 0 is allowed 

to save the model 
```
def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])


    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
```
## 6. Slurm sbatch for task submitting
```
#!/bin/bash
#SBATCH --job-name=ddp_job                # Job name
#SBATCH --nodes=2                         # Number of nodes
#SBATCH --ntasks-per-node=4               # Number of tasks per node (adjust according to GPUs per node)
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --gres=gpu:4                      # Number of GPUs per node (adjust to match your cluster)
#SBATCH --mem=32GB                        # Total memory for each node
#SBATCH --time=01:00:00                   # Time limit hrs:min:sec
#SBATCH --output=ddp_out_%j.log           # Standard output and error log (%j will be replaced by the job ID)
#SBATCH --partition=your_partition_name   # Specify partition (if required by the cluster)

# Load required modules (adjust based on your cluster environment)
module load python/3.8
module load cuda/11.2
module load pytorch

# Activate virtual environment (if you are using one)
source ~/your_virtual_env/bin/activate

# Define master address and port (used by PyTorch DDP)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  # First node's address
MASTER_PORT=12345  # Choose a port (ensure it's free for use)

# Set the number of processes (this is equal to the total number of GPUs used across nodes)
WORLD_SIZE=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Run the distributed PyTorch script
srun python -m torch.distributed.launch \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    your_ddp_script.py --arg1 --arg2
```
```
sbatch ddp_job.sh
```
## 7. Docker for DDP

Continue