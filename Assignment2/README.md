# 强化学习项目 - 深度Q网络

本项目实现了基于Dueling DQN算法的强化学习解决方案，用于LunarLander-v3环境。

## 环境要求

- Python >= 3.8
- PyTorch (支持CUDA可选)

## 安装依赖

```bash
# CPU版本
pip install -r requirements.txt

# GPU版本 (安装CUDA后)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 快速开始

### 1. 环境检查
```bash
python check_environment.py
```

### 2. 运行训练脚本
```bash
python train_with_gpu_check.py
```

### 3. 使用Jupyter Notebook
```bash
jupyter notebook
```
然后打开 `Problem1_With_GPU_Check.ipynb` 文件。如果是第一次运行，请确保更换内核为 "PyTorch GPU Environment"。

## 文件说明

- `requirements.txt`: 项目依赖
- `lunar_lander_dqn.py`: 主要程序

## GPU加速

如果安装了NVIDIA GPU和CUDA兼容的PyTorch版本，训练将自动使用GPU加速。