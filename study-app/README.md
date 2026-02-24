# nanoGPT 初中互动学习实验室

这个目录是一个纯前端学习应用，目标是用初中数学知识理解 `nanoGPT` 的核心实现。
当前版本以图表为主，包含多张可交互可视化。

## 快速打开

1. 直接双击 `study-app/index.html`
2. 或者在仓库根目录运行:

```powershell
python -m http.server 8000
```

然后访问 `http://localhost:8000/study-app/`

## 学习路径

1. 实验 1: 文字编码和右移配对，理解 `x -> y` 训练样本
2. 实验 2: 因果注意力和 softmax，理解为什么“不能看未来”
3. 实验 3: 一个 Block 的残差流程
4. 实验 4: 学习率/批量对 loss 的影响，和 `temperature/top_k` 采样

## 图表清单

- 字符频率柱状图
- 注意力热力图 (Attention Matrix)
- Block 阶段能量图 (L2 范数)
- 当前阶段向量分量图
- 训练 loss 双曲线 (train/val)
- 学习率计划图 (warmup + cosine)
- 采样候选概率柱状图
- GPT 组件架构图
- 数据管线图
- 张量形状流转图
- 注意力头并行结构图
- 参数分布图 (近似)
- FLOPs-上下文长度曲线 (近似)
- 训练循环时序图
- 显存压力估计图 (近似)

## 术语全解

- 页面顶部新增“术语全解”区，覆盖常见易混名词
- 每个术语都包含:
  - 一句话定义
  - 详细解释
  - 初中类比
  - 常见迷糊点
  - 源码位置
- 页面技术词支持悬浮解释气泡（鼠标停留即可，不跳转）
- 点击图表元素会自动高亮关联术语卡片

## 对应源码

- `train.py:116` `get_batch(split)`
- `model.py:68` `masked_fill` 因果 mask
- `model.py:69` `softmax` 注意力归一化
- `model.py:104` 与 `model.py:105` Block 两次残差
- `train.py:231` 学习率函数
- `train.py:255` 训练主循环
- `model.py:306` 生成函数
- `model.py:320` 与 `model.py:326` top-k 与采样
