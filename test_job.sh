#!/bin/bash
#SBATCH --job-name=test_python       # 作业名称
#SBATCH --output=test_python_output.txt  # 输出文件
#SBATCH --ntasks=1                   # 任务数量
#SBATCH --time=00:01:00              # 最大运行时间
#SBATCH --partition=general          # 分区名称

# 加载 Python 模块（如果需要）
# module load python

# 执行 Python 代码
python -c "print('Hello from Python within Slurm!')"
