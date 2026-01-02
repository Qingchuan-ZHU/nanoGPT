# 我的 nanoGPT 学习项目 (Master Branch)

> **💡 分支管理说明**：
> 本分支 (`master`) 仅存放核心训练代码与基准配置。
> 所有的实验过程、深度源码解析及学习心得已归档至 **`study-notes`** 目录（详见 `study-notes` 分支）。

## 📂 学习笔记导航 (Study Notes Directory)

* **[👉 点击进入：study-notes分支](../../tree/study-notes)**
* **[👉 点击进入：实验日志](../../tree/study-notes/notes/1_test_run_records.md)**
    * *在 Windows 环境下使用 RTX 4070 Ti Super 训练莎士比亚字符级模型的实验日志，记录了环境配置、训练过程监控以及对生成文本风格与局限性的深度分析。*
* **[👉 点击查看：数据预处理](../../tree/study-notes/notes/2_prepare_with_comments.py)**
    * *详细注释了如何将莎士比亚文本转化为由 65 个唯一字符组成的词表，并编码为二进制格式以供模型进行高效的预测训练。*
* **[👉 点击查看：数据预处理](../../tree/study-notes/notes/3_model.py_Layer_norm_learn_note.md)**
    * *深度拆解了 LayerNorm 的源码实现，从类继承、初始化参数到 forward 计算逻辑，详细解析了这一“架构防火墙”如何保障模型数值的稳定性与灵活性。*
