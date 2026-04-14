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
然后打开 `Problem1_Clean.ipynb`。

## 生成文件位置

- 训练曲线图会保存为 `training_curves.png`
- 最终演示 GIF 会保存为 `lander_final.gif`（脚本和 Clean notebook）
- 运行 `lunar_lander_dqn.py` 时，输出写入脚本所在目录
- 运行 `Problem1_Clean.ipynb` 时，输出写入当前工作目录（通常是你打开 notebook 的目录）

## 文件说明

- `requirements.txt`: 项目依赖
- `lunar_lander_dqn.py`: 主要训练脚本
- `Problem1_Clean.ipynb`: 清理版 notebook

## GPU加速

当前代码默认使用 CPU 训练。如果需要启用 GPU，可将设备设置改回自动检测 CUDA 的写法。