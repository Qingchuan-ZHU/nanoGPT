# nanoGPT 初中互动学习实验室

这个目录是一个纯前端学习应用，目标是用初中数学知识理解 `nanoGPT` 的核心实现。
当前版本已从单页拆分为多页，每页只保留一个主题模块。

## 快速打开

1. 直接双击 `study-app/index.html`
2. 或者在仓库根目录运行:

```powershell
python -m http.server 8000
```

然后访问 `http://localhost:8000/study-app/`

## 页面结构

- `index.html`: 首页和学习路径导航
- `glossary.html`: 术语全解 + 源码映射
- `architecture.html`: 架构图谱总览
- `exp1.html`: 文本编码与右移配对
- `exp2.html`: 因果注意力与 softmax
- `exp3.html`: Transformer Block 流程
- `exp4.html`: 训练模拟与采样控制

## 学习路径

1. `glossary.html`: 先把术语和代码位置建立映射
2. `architecture.html`: 看整机流程和形状/参数/算力关系
3. `exp1.html` 到 `exp4.html`: 逐个完成四个交互实验

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
