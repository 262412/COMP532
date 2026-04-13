# 强化学习项目 - 深度Q网络

本项目实现了基于 Dueling Double DQN 算法的强化学习方案，用于 LunarLander-v3 环境。

## 环境要求

- Python >= 3.8
- PyTorch (支持 CUDA 可选)

## 安装依赖

```bash
# CPU 版本
pip install -r requirements.txt

# GPU 版本 (安装 CUDA 后)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 快速开始

### 1. 运行训练脚本
```bash
python lunar_lander_dqn.py
```

### 2. 使用 Jupyter Notebook
```bash
jupyter notebook
```
然后打开 `Problem1_Clean.ipynb` 或 `Problem1.ipynb`。

## 生成文件位置

- 训练曲线图会保存为 `training_curves.png`
- 最终演示 GIF 会保存为 `lander_final.gif`（脚本和 Clean notebook）或 `lander.gif`（Problem1 notebook）
- 以上文件都会写入 `Assignment2` 目录，不依赖你从哪个工作目录启动程序

## 文件说明

- `requirements.txt`: 项目依赖
- `lunar_lander_dqn.py`: 主要训练脚本
- `Problem1_Clean.ipynb`: 清理版 notebook
- `Problem1.ipynb`: 练习版 notebook

## GPU加速

如果安装了 NVIDIA GPU 和 CUDA 兼容的 PyTorch 版本，训练将自动使用 GPU 加速。