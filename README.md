
# nanoGPT 实战：从零开始的莎士比亚剧本训练

本项目基于 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT)，旨在记录在 RTX 4070 Ti Super 硬件环境下，从零实现 Transformer 模型训练、监控与调优的完整过程。

---

## 📚 学习导航 (Study Navigator)

为了保持主分支代码的纯净，我将详细的实验记录、源码拆解及心得感悟存放在 **`study-notes`** 分支中。

* **[👉 点击进入：Day 1 学习笔记与实验复盘](../../tree/study-notes)**
    * *包含：W&B 实时监控图表、4070 Ti Super 性能分析、模型生成结果深度意译。*
* **[👉 点击查看：带详细注释的预处理脚本](../../blob/study-notes/prepare_with_comments.py)**

---

## 🛠️ 快速开始 (Quick Start)

### 1. 环境准备
```bash
pip install torch numpy transformers datasets tiktoken wandb

```

### 2. 数据处理

```bash
python data/shakespeare_char/prepare.py

```

### 3. 模型训练

```bash
python train.py --dataset=shakespeare_char --device=cuda --compile=False --wandb_log=True --eval_interval=50 --max_iters=2000

```

---

## 🔬 实验摘要 (Experiment Summary)

* **模型规模**: 3.16M Parameters
* **训练耗时**: ~15 mins (on RTX 4070 Ti Super)
* **最终 Loss**: Train 0.94 / Val 1.73
* **主要工具**: PyTorch, Weights & Biases, TortoiseGit

```
