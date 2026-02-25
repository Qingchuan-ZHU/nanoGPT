(() => {
  const byId = (id) => document.getElementById(id);
  const hasEl = (id) => Boolean(byId(id));
  const on = (id, eventName, handler) => {
    const el = byId(id);
    if (!el) return false;
    el.addEventListener(eventName, handler);
    return true;
  };

  const safeChar = (ch) => {
    if (ch === " ") return "␠";
    if (ch === "\n") return "\\n";
    if (ch === "\t") return "\\t";
    return ch;
  };

  const escapeHtml = (s) =>
    String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");

  const fmtNum = (x, n = 3) => Number(x).toFixed(n);
  const fmtVec = (arr) => `[${arr.map((v) => fmtNum(v, 3)).join(", ")}]`;
  const sum = (arr) => arr.reduce((a, b) => a + b, 0);

  const softmax = (scores) => {
    const finite = scores.filter((v) => Number.isFinite(v));
    if (!finite.length) return scores.map(() => 0);
    const mx = Math.max(...finite);
    const exps = scores.map((v) => (Number.isFinite(v) ? Math.exp(v - mx) : 0));
    const denom = sum(exps) || 1;
    return exps.map((v) => v / denom);
  };

  const withCanvas = (id, draw, fallbackW = 520, fallbackH = 280) => {
    const canvas = byId(id);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || fallbackW;
    const h = canvas.clientHeight || fallbackH;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    draw(ctx, w, h);
  };

  const drawEmptyChart = (ctx, w, h, text = "暂无数据") => {
    ctx.fillStyle = "#fffdf6";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#d9d7cf";
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
    ctx.fillStyle = "#6f7e90";
    ctx.font = "14px LXGW WenKai, Source Han Sans SC, sans-serif";
    ctx.fillText(text, Math.max(14, w / 2 - 36), h / 2);
  };

  const drawGrid = (ctx, x, y, w, h, lines = 4) => {
    ctx.strokeStyle = "#d9dde1";
    ctx.lineWidth = 1;
    for (let i = 0; i <= lines; i += 1) {
      const yy = y + (h * i) / lines;
      ctx.beginPath();
      ctx.moveTo(x, yy);
      ctx.lineTo(x + w, yy);
      ctx.stroke();
    }
  };

  const heatColor = (v) => {
    const t = Math.max(0, Math.min(1, v));
    const r = Math.round(255 - 174 * t);
    const g = Math.round(241 - 86 * t);
    const b = Math.round(222 - 138 * t);
    return `rgb(${r},${g},${b})`;
  };

  const textExamples = [
    "to be or not to be",
    "hello nanogpt",
    "small model big idea",
    "attention sees only left",
    "gpt predicts next token",
  ];

  const charState = {
    freqEntries: [],
  };

  const attnState = {
    tokens: [],
    scores: [],
    matrix: [],
    probs: [],
  };

  const stageDefs = [
    { key: "x0", title: "1) 输入 x" },
    { key: "ln1", title: "2) LayerNorm 1" },
    { key: "attn", title: "3) Attention 输出" },
    { key: "x1", title: "4) 残差 x + attn" },
    { key: "ln2", title: "5) LayerNorm 2" },
    { key: "mlp", title: "6) MLP 输出" },
    { key: "x2", title: "7) 残差 x + mlp" },
  ];

  const stageExplain = [
    "初始向量: 你可以把它看成这个 token 的数字特征。",
    "LayerNorm: 让数值分布更稳定，训练更容易收敛。",
    "Attention: 用上下文信息重算当前 token 的表示。",
    "第一次残差: 保留原信息并叠加注意力信息。",
    "第二次 LayerNorm: 再次稳定数值范围。",
    "MLP: 做非线性变换，补充表达能力。",
    "第二次残差: 得到 Block 最终输出。",
  ];

  const blockState = {
    step: 0,
    vectors: {},
    autoTimer: null,
  };

  const trainState = {
    train: [],
    val: [],
    lrs: [],
    warmupEnd: 0,
  };

  const sampleState = {
    seq: [],
    dist: [],
  };

  const archState = {
    nLayer: 12,
    nHead: 12,
    nEmbd: 768,
    blockSize: 1024,
    vocabSize: 50304,
    warning: "",
    totalParams: 0,
  };

  const canvasHitZones = {};
  const setCanvasHitZones = (canvasId, zones) => {
    canvasHitZones[canvasId] = zones || [];
  };

  let currentExample = 0;

  /* ---------------------- 术语全解 ---------------------- */
  const glossaryTerms = [
    {
      name: "字符",
      alias: "character",
      level: "入门",
      plain: "文本里最小单位，比如 a、空格、逗号。",
      detail: "在字符级训练里，每个字符都被映射成一个数字 id，再送进模型。",
      analogy: "像把一句话拆成一个个字母积木。",
      mistake: "容易和 token 混淆。token 可能是一个词，也可能是词的一部分。",
      code: "train.py:116, data/*/prepare.py",
      scene: "实验 1 的输入文本和词表。",
    },
    {
      name: "词元",
      alias: "token",
      level: "入门",
      plain: "模型真正处理的单位，可能是字、词或子词。",
      detail: "GPT 常用子词分词，一个 token 不一定是完整单词。",
      analogy: "像拼图块，不一定一块就是完整图案。",
      mistake: "把 token 当成“词”会导致对长度和成本估计错误。",
      code: "sample.py:74-78, model.py:170",
      scene: "实验 2 token 序列、实验 4 采样。",
    },
    {
      name: "词表",
      alias: "vocab / vocabulary",
      level: "入门",
      plain: "模型能识别的全部 token 清单。",
      detail: "每个 token 在词表里有唯一 id，预测时输出也是这个词表上的概率分布。",
      analogy: "像字典目录，先查目录编号再取词。",
      mistake: "词表越大不一定越好，会增加计算和参数量。",
      code: "model.py:110-113, model.py:125",
      scene: "实验 1 的 stoi/itos。",
    },
    {
      name: "词表大小",
      alias: "vocab_size",
      level: "入门",
      plain: "词表里 token 的总数。",
      detail: "决定输出层维度，`lm_head` 的输出列数就是 `vocab_size`。",
      analogy: "像多选题里选项个数。",
      mistake: "改 vocab_size 后，旧权重通常不能直接兼容。",
      code: "model.py:110, model.py:125, model.py:188",
      scene: "术语理解区和源码映射表。",
    },
    {
      name: "字符到编号",
      alias: "stoi",
      level: "入门",
      plain: "string-to-index，把字符/词映射到 id。",
      detail: "编码时用 `stoi[c]` 把输入文本变成数字序列。",
      analogy: "像通讯录里“名字 -> 电话号”。",
      mistake: "stoi 和 itos 必须成对一致，否则解码会错乱。",
      code: "sample.py:69-70",
      scene: "实验 1 词表展示。",
    },
    {
      name: "编号到字符",
      alias: "itos",
      level: "入门",
      plain: "index-to-string，把 id 还原成字符/词。",
      detail: "生成阶段把模型采样出的 id 序列还原成可读文本。",
      analogy: "像“电话号 -> 联系人名字”。",
      mistake: "若词表顺序改变，itos 还原内容会完全变样。",
      code: "sample.py:69-71",
      scene: "实验 1 与实验 4 的文本输出概念。",
    },
    {
      name: "编码",
      alias: "encode",
      level: "入门",
      plain: "把文本转换为 token id 序列。",
      detail: "训练和推理都要先编码，模型只接收数字张量。",
      analogy: "把汉字翻译成机器能读的条形码。",
      mistake: "编码规则不一致会直接导致模型输入语义错位。",
      code: "sample.py:70, sample.py:76-77",
      scene: "实验 1 编码序列。",
    },
    {
      name: "解码",
      alias: "decode",
      level: "入门",
      plain: "把 token id 序列还原为文本。",
      detail: "采样后必须解码才能看见自然语言结果。",
      analogy: "扫码后把数字还原成商品名称。",
      mistake: "误以为模型直接“输出文字”，实际先输出 id。",
      code: "sample.py:71, sample.py:78",
      scene: "实验 4 生成输出。",
    },
    {
      name: "上下文长度",
      alias: "block_size",
      level: "入门",
      plain: "一次最多看到多少个 token。",
      detail: "超出就会截断历史，注意力矩阵是 block_size 范围内计算。",
      analogy: "像记事本最多只能记 1024 个字。",
      mistake: "把它当 batch 大小。它是序列长度，不是样本数。",
      code: "model.py:170-171, train.py:50, train.py:121",
      scene: "实验 2 注意力可视化。",
    },
    {
      name: "训练输入 x 与标签 y",
      alias: "x / y shifted targets",
      level: "入门",
      plain: "x 是当前序列，y 是右移一位后的目标序列。",
      detail: "模型学的是“给定前文预测下一个 token”。",
      analogy: "做完形填空：看前面，猜下一格。",
      mistake: "不是分类到固定类别，而是每个位置都要预测一次。",
      code: "train.py:121-122",
      scene: "实验 1 的配对列表。",
    },
    {
      name: "取批函数",
      alias: "get_batch",
      level: "入门",
      plain: "每步训练随机抽一批样本。",
      detail: "从 train.bin/val.bin 随机切片，拼成 `(batch, block)` 的 x、y。",
      analogy: "从题库随机抽一页题来练。",
      mistake: "误以为按顺序读完整数据；这里是随机抽样。",
      code: "train.py:116-131",
      scene: "实验 1 与实验 4 的数据来源概念。",
    },
    {
      name: "训练集",
      alias: "train split",
      level: "入门",
      plain: "用来更新参数的数据。",
      detail: "反向传播只在 train 上做，模型直接在这部分学习。",
      analogy: "平时做练习题。",
      mistake: "train loss 低不代表泛化好。",
      code: "train.py:118, train.py:222",
      scene: "实验 4 loss 图中的 train 曲线。",
    },
    {
      name: "验证集",
      alias: "val split",
      level: "入门",
      plain: "只评估不训练的数据。",
      detail: "用于判断模型在新样本上的表现，辅助保存最佳 checkpoint。",
      analogy: "月考题，不用于日常背答案。",
      mistake: "把 val 也拿来训练会造成评估失真。",
      code: "train.py:120, train.py:217-225, train.py:264",
      scene: "实验 4 loss 图中的 val 曲线。",
    },
    {
      name: "批大小",
      alias: "batch_size",
      level: "入门",
      plain: "一次并行处理多少条序列样本。",
      detail: "batch 大通常更稳但更占显存，batch 小噪声更大。",
      analogy: "一次批改 8 份还是 64 份作业。",
      mistake: "和 block_size 混淆。一个是样本数，一个是序列长度。",
      code: "train.py:49, train.py:121",
      scene: "实验 4 训练控制器。",
    },
    {
      name: "迭代步",
      alias: "iter / step",
      level: "入门",
      plain: "参数更新一次记作一步。",
      detail: "训练循环每次 `scaler.step(optimizer)` 之后迭代数增加。",
      analogy: "跑步计步器，每迈一步记 1。",
      mistake: "不要和 epoch 混淆，step 更细粒度。",
      code: "train.py:255, train.py:311, train.py:328",
      scene: "实验 4 横轴步数。",
    },
    {
      name: "嵌入向量",
      alias: "embedding",
      level: "中级",
      plain: "把离散 id 映射成连续向量。",
      detail: "token embedding + position embedding 相加后进入 Transformer。",
      analogy: "把每个词变成一串坐标，方便算距离和关系。",
      mistake: "embedding 不是 one-hot；它是可训练参数矩阵。",
      code: "model.py:125-126, model.py:176-178",
      scene: "实验 3 向量流程。",
    },
    {
      name: "词嵌入表",
      alias: "wte",
      level: "中级",
      plain: "word token embedding 参数表。",
      detail: "输入 token id 查表得到语义向量。",
      analogy: "像查新华字典里每个字的属性卡。",
      mistake: "wte 和 lm_head 在这里做了权重共享。",
      code: "model.py:125, model.py:133",
      scene: "模型结构讲解中的 embedding。",
    },
    {
      name: "位置嵌入表",
      alias: "wpe",
      level: "中级",
      plain: "position embedding 参数表。",
      detail: "告诉模型“这是第几个位置”，弥补顺序信息。",
      analogy: "给每个字贴上座位号。",
      mistake: "仅有 token embedding 时模型不天然知道顺序。",
      code: "model.py:126, model.py:177",
      scene: "实验 2 的顺序与因果关系。",
    },
    {
      name: "隐藏维度",
      alias: "n_embd",
      level: "中级",
      plain: "每个 token 向量长度。",
      detail: "维度越大表达能力越强，但计算和显存成本上升。",
      analogy: "描述一个人时用 8 个特征还是 768 个特征。",
      mistake: "不是层数，也不是头数。",
      code: "model.py:113, model.py:122",
      scene: "实验 3 向量分量图。",
    },
    {
      name: "层数",
      alias: "n_layer",
      level: "中级",
      plain: "Transformer Block 堆叠次数。",
      detail: "层数更深通常能力更强，也更难训练。",
      analogy: "解题过程写 3 步还是 20 步。",
      mistake: "层数增加不保证一定更好，需要配合数据与算力。",
      code: "model.py:111, model.py:128",
      scene: "实验 3 Block 概念。",
    },
    {
      name: "注意力头数",
      alias: "n_head",
      level: "中级",
      plain: "并行关注关系的“视角”数量。",
      detail: "每个头独立算注意力，再拼接回去。",
      analogy: "同一段话同时让语法老师和语义老师看。",
      mistake: "头数不是越多越好，受 n_embd 整除限制。",
      code: "model.py:32-33, model.py:112",
      scene: "实验 2 注意力理解。",
    },
    {
      name: "层归一化",
      alias: "LayerNorm",
      level: "中级",
      plain: "把一层特征拉到更稳定的数值范围。",
      detail: "按特征维度标准化，减少训练震荡。",
      analogy: "把各科成绩先按统一标准换算再比较。",
      mistake: "不是把数据裁剪到 0-1，而是标准化后再加可学习缩放偏移。",
      code: "model.py:18-27, model.py:100, model.py:102",
      scene: "实验 3 第 2 步和第 5 步。",
    },
    {
      name: "注意力",
      alias: "self-attention",
      level: "中级",
      plain: "每个位置按权重汇总其他位置信息。",
      detail: "核心是 `softmax(score)` 得权重，再对 value 做加权和。",
      analogy: "做阅读理解时给关键词不同重视度。",
      mistake: "注意力不是简单平均，是可学习的加权。",
      code: "model.py:52-74",
      scene: "实验 2 概率条与热力图。",
    },
    {
      name: "因果约束",
      alias: "causal mask",
      level: "中级",
      plain: "当前词不能偷看未来词。",
      detail: "未来位置打成 `-inf`，softmax 后概率变 0。",
      analogy: "考试不能看后面同学答案。",
      mistake: "不是“弱化未来”，而是严格屏蔽。",
      code: "model.py:68-69",
      scene: "实验 2 中被划线的 future token。",
    },
    {
      name: "查询",
      alias: "Query (Q)",
      level: "进阶",
      plain: "表示“我现在想找什么信息”。",
      detail: "当前位置的 Q 与其他位置 K 点积，得到相关性打分。",
      analogy: "提问句。",
      mistake: "Q 不是概率，它只是匹配打分前的向量。",
      code: "model.py:56-58",
      scene: "实验 2 打分滑条对应 query 行。",
    },
    {
      name: "键",
      alias: "Key (K)",
      level: "进阶",
      plain: "表示“我能提供什么信息”。",
      detail: "K 像索引标签，供 Q 来匹配。",
      analogy: "图书馆索引卡。",
      mistake: "K 不是最终输出内容。",
      code: "model.py:56-58",
      scene: "实验 2 热力图列方向。",
    },
    {
      name: "值",
      alias: "Value (V)",
      level: "进阶",
      plain: "真正被加权汇总的信息向量。",
      detail: "权重来自 QK，内容来自 V，最后求和得到上下文。",
      analogy: "查到目录后真正拿到的书。",
      mistake: "很多人把 K 当内容，实际 V 才是内容载体。",
      code: "model.py:56-58, model.py:71",
      scene: "实验 2 概率加权结果。",
    },
    {
      name: "softmax",
      alias: "softmax",
      level: "中级",
      plain: "把一组分数转成和为 1 的概率。",
      detail: "分数越大概率越高，且所有项总和恰好 1。",
      analogy: "把同学投票数折算成百分比。",
      mistake: "softmax 输出不是“是否正确”，只是相对概率。",
      code: "model.py:69, model.py:324",
      scene: "实验 2 概率条、实验 4 采样概率图。",
    },
    {
      name: "logits",
      alias: "logits",
      level: "中级",
      plain: "softmax 之前的原始打分。",
      detail: "logits 可正可负，经过 softmax 才变成概率。",
      analogy: "比赛原始评分表。",
      mistake: "把 logits 当概率会出错。",
      code: "model.py:187-190, model.py:318",
      scene: "实验 4 采样前概念。",
    },
    {
      name: "概率分布",
      alias: "probs",
      level: "入门",
      plain: "每个候选 token 被选中的概率。",
      detail: "采样时按这个分布随机抽一个 token。",
      analogy: "抽奖箱里每个球占的比例。",
      mistake: "最高概率不代表必然选中。",
      code: "model.py:324-326",
      scene: "实验 4 候选概率柱状图。",
    },
    {
      name: "多层感知机",
      alias: "MLP",
      level: "中级",
      plain: "Block 内的前馈网络，做非线性变换。",
      detail: "常见结构是升维 -> 激活 -> 降维。",
      analogy: "把信息先展开细看，再压缩总结。",
      mistake: "不是分类器末层；在每个 Block 内都存在。",
      code: "model.py:78-92, model.py:102, model.py:105",
      scene: "实验 3 第 6 步。",
    },
    {
      name: "激活函数",
      alias: "GELU",
      level: "进阶",
      plain: "给网络增加非线性能力。",
      detail: "GELU 相对平滑，GPT 系列常用。",
      analogy: "给直线规则加弯曲，让表达更灵活。",
      mistake: "激活函数不是随机噪声。",
      code: "model.py:83, model.py:89",
      scene: "实验 3 MLP 过程概念。",
    },
    {
      name: "残差连接",
      alias: "residual connection",
      level: "中级",
      plain: "把原输入直接加回输出。",
      detail: "`x = x + f(x)` 让深层训练更稳定，不易梯度消失。",
      analogy: "做笔记时保留原文，再加批注。",
      mistake: "不是跳过学习，而是保底保真。",
      code: "model.py:104-105",
      scene: "实验 3 第 4 步与第 7 步。",
    },
    {
      name: "丢弃",
      alias: "dropout",
      level: "中级",
      plain: "训练时随机屏蔽一部分神经元输出。",
      detail: "用于防过拟合，推理时通常关闭。",
      analogy: "练习时故意不看部分提示，防止依赖。",
      mistake: "dropout 不是删参数，只是训练时临时屏蔽。",
      code: "model.py:40-41, model.py:85, train.py:54",
      scene: "实验 4 控制器。",
    },
    {
      name: "偏置项",
      alias: "bias",
      level: "中级",
      plain: "线性变换里的平移项。",
      detail: "`y = Wx + b` 里的 `b`，可让模型拟合更灵活。",
      analogy: "在函数图像里整体上移下移。",
      mistake: "这个 bias 不是注意力 mask 的 `attn.bias` 缓冲区概念。",
      code: "model.py:21-24, model.py:36, model.py:114",
      scene: "模型配置术语。",
    },
    {
      name: "损失",
      alias: "loss",
      level: "入门",
      plain: "模型预测和正确答案差得有多远。",
      detail: "训练目标是让 loss 逐步下降。",
      analogy: "做题错得越多，扣分越多。",
      mistake: "单步 loss 抖动是正常的，要看趋势。",
      code: "model.py:186-188, train.py:264, train.py:301",
      scene: "实验 4 loss 曲线。",
    },
    {
      name: "交叉熵",
      alias: "cross_entropy",
      level: "中级",
      plain: "语言模型最常见的分类损失。",
      detail: "对每个位置比较预测分布和真实 token id。",
      analogy: "把你给每个选项的概率和标准答案对比打分。",
      mistake: "不是简单做差，而是对数概率损失。",
      code: "model.py:187",
      scene: "实验 4 loss 含义。",
    },
    {
      name: "优化器",
      alias: "optimizer / AdamW",
      level: "中级",
      plain: "根据梯度更新参数的规则。",
      detail: "AdamW 会结合动量和二阶统计，更新更平稳。",
      analogy: "根据错题统计来调整学习计划。",
      mistake: "优化器不是模型本体，但会强烈影响收敛速度。",
      code: "model.py:263-282, train.py:205-208",
      scene: "实验 4 训练过程。",
    },
    {
      name: "学习率",
      alias: "learning_rate",
      level: "入门",
      plain: "每次参数更新迈多大步。",
      detail: "太大容易震荡，太小收敛慢。",
      analogy: "下山步幅，太大容易摔，太小太慢。",
      mistake: "学习率好坏与 batch、模型大小有关，不是固定神值。",
      code: "train.py:58, train.py:231, train.py:258",
      scene: "实验 4 学习率滑条和 LR 图。",
    },
    {
      name: "预热",
      alias: "warmup",
      level: "中级",
      plain: "训练初期把学习率从小逐步拉高。",
      detail: "减少初始阶段不稳定更新。",
      analogy: "运动前先热身，不直接冲刺。",
      mistake: "warmup 不是额外训练轮次，而是学习率阶段。",
      code: "train.py:65, train.py:233-234",
      scene: "实验 4 LR 计划图左侧。",
    },
    {
      name: "余弦衰减",
      alias: "cosine decay",
      level: "中级",
      plain: "后期学习率按余弦曲线缓慢下降。",
      detail: "让训练后半程更细腻地逼近最优点。",
      analogy: "冲刺后逐渐减速，稳定到终点。",
      mistake: "不是线性下降，曲线前快后慢。",
      code: "train.py:238-241",
      scene: "实验 4 LR 计划图右侧。",
    },
    {
      name: "梯度累计",
      alias: "gradient_accumulation_steps",
      level: "中级",
      plain: "多次小 batch 累积后再更新一次参数。",
      detail: "常用于显存不足时模拟更大 batch。",
      analogy: "把几次小测平均后再记总评。",
      mistake: "累计后要除以步数，不然梯度尺度会偏大。",
      code: "train.py:48, train.py:294-301",
      scene: "实验 4 控制器。",
    },
    {
      name: "梯度裁剪",
      alias: "grad_clip",
      level: "中级",
      plain: "限制梯度最大范数，防止爆炸。",
      detail: "尤其在大模型或不稳定阶段有保护作用。",
      analogy: "给速度设限，防止失控。",
      mistake: "裁剪不是让训练更快，而是更稳。",
      code: "train.py:63, train.py:307-309",
      scene: "训练稳定性概念。",
    },
    {
      name: "检查点",
      alias: "checkpoint",
      level: "入门",
      plain: "训练中途保存的模型快照。",
      detail: "包含模型参数、优化器状态、迭代步等，可恢复训练。",
      analogy: "打游戏存档点。",
      mistake: "只存模型不存优化器，恢复后轨迹会变。",
      code: "train.py:273-286, sample.py:40-52",
      scene: "实验 4 训练流程理解。",
    },
    {
      name: "过拟合",
      alias: "overfitting",
      level: "中级",
      plain: "训练集表现好，但新数据表现差。",
      detail: "常见信号是 train loss 持续降、val loss 不降反升。",
      analogy: "只会背题库原题，变题就不会。",
      mistake: "只看 train 曲线会误判模型真的变好。",
      code: "train.py:264-266",
      scene: "实验 4 train/val 双曲线。",
    },
    {
      name: "生成函数",
      alias: "generate",
      level: "入门",
      plain: "按“预测一个 -> 追加一个”循环生成文本。",
      detail: "每步把最新序列再喂回模型，直到达到长度限制。",
      analogy: "接龙，一次接一个词。",
      mistake: "不是一次性生成全部词。",
      code: "model.py:306-329, sample.py:89-95",
      scene: "实验 4 采样控制台。",
    },
    {
      name: "温度",
      alias: "temperature",
      level: "中级",
      plain: "调节概率分布尖锐程度。",
      detail: "温度低更保守，温度高更发散。",
      analogy: "考试答题时是保守选稳妥项还是大胆猜想。",
      mistake: "温度不改变知识量，只改变采样随机性。",
      code: "model.py:306, model.py:318, sample.py:16",
      scene: "实验 4 温度滑条。",
    },
    {
      name: "候选截断",
      alias: "top_k",
      level: "中级",
      plain: "只保留概率最高的前 k 个候选。",
      detail: "其余候选直接置 0，能减少胡言乱语。",
      analogy: "只在前 k 名同学里抽奖。",
      mistake: "k 过小会变得机械，k 过大容易发散。",
      code: "model.py:320-323, sample.py:17",
      scene: "实验 4 top_k 滑条。",
    },
    {
      name: "按概率采样",
      alias: "multinomial sampling",
      level: "中级",
      plain: "按概率随机抽取下一个 token。",
      detail: "不是永远选最大概率，这能保留多样性。",
      analogy: "按权重抽签。",
      mistake: "“随机”不等于完全乱，是受概率分布控制的随机。",
      code: "model.py:326",
      scene: "实验 4 连续生成。",
    },
    {
      name: "最大新词数",
      alias: "max_new_tokens",
      level: "入门",
      plain: "一次生成最多追加多少 token。",
      detail: "控制输出长度和计算成本。",
      analogy: "作文规定最多写多少字。",
      mistake: "不是总长度，是“新增长度”。",
      code: "model.py:306, sample.py:15",
      scene: "实验 4 生成步数概念。",
    },
    {
      name: "编译优化",
      alias: "torch.compile",
      level: "进阶",
      plain: "用图编译优化前向/反向执行速度。",
      detail: "首次编译慢，后续训练可能更快。",
      analogy: "先做模板，之后流水线更快。",
      mistake: "不是所有环境都稳定提速。",
      code: "train.py:210-214, sample.py:58-60",
      scene: "高级性能术语。",
    },
    {
      name: "混合精度",
      alias: "fp16 / bf16",
      level: "进阶",
      plain: "用更低精度数值换速度和显存。",
      detail: "常配 GradScaler 防止梯度下溢。",
      analogy: "草稿纸先用近似算，关键处再精算。",
      mistake: "精度低不一定精度差，关键是数值稳定策略。",
      code: "train.py:73-75, train.py:202, train.py:305-312",
      scene: "训练效率术语。",
    },
    {
      name: "DDP 分布式训练",
      alias: "DistributedDataParallel",
      level: "进阶",
      plain: "多 GPU 并行训练同一模型。",
      detail: "每卡算一部分 batch，再同步梯度。",
      analogy: "多人分工做题，最后合并答案更新总笔记。",
      mistake: "batch 规模和梯度累计要按 world_size 调整。",
      code: "train.py:81-101, train.py:213-215",
      scene: "大规模训练背景概念。",
    },
    {
      name: "模型利用率",
      alias: "MFU",
      level: "进阶",
      plain: "模型计算吞吐接近硬件峰值的程度。",
      detail: "是训练效率指标，不是模型准确率指标。",
      analogy: "发动机满负荷运转比例。",
      mistake: "MFU 高不代表 loss 低。",
      code: "model.py:284-304, train.py:321-323",
      scene: "训练日志里的 mfu 字段。",
    },
  ];

  const normText = (s) => String(s || "").trim().toLowerCase();
  const escapeRegex = (s) => String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const isAsciiWordLike = (s) => /^[A-Za-z0-9_./+-]+$/.test(s);
  const buildTermExample = (term) => {
    if (term.example) return term.example;
    if (term.scene) return `例如: ${term.scene}`;
    return `例如: 在学习流程中，用“${term.name}”描述对应步骤。`;
  };

  glossaryTerms.forEach((t, idx) => {
    t.example = buildTermExample(t);
    t._key = `term-${idx}`;
    t._searchBlob = normText(
      `${t.name} ${t.alias} ${t.level} ${t.plain} ${t.detail} ${t.analogy} ${t.mistake} ${t.example} ${t.scene}`
    );
  });

  const termByKey = new Map(glossaryTerms.map((t) => [t._key, t]));
  const triggerToKey = new Map();

  const registerTrigger = (raw, key) => {
    const cleaned = String(raw || "").replace(/\([^)]*\)/g, " ").replace(/\s+/g, " ").trim();
    if (!cleaned) return;
    const ascii = /^[\x00-\x7F]+$/.test(cleaned);
    if (ascii && cleaned.length < 3) return;
    if (!ascii && cleaned.length < 2) return;
    const trigger = normText(cleaned);
    if (!triggerToKey.has(trigger)) triggerToKey.set(trigger, key);
  };

  glossaryTerms.forEach((t) => {
    registerTrigger(t.name, t._key);
    registerTrigger(t.alias, t._key);
    t.alias
      .split(/[\/,|]/g)
      .map((x) => x.trim())
      .forEach((piece) => {
        registerTrigger(piece, t._key);
        registerTrigger(piece.replace(/_/g, " "), t._key);
      });
  });

  const triggerPatterns = [...triggerToKey.keys()].sort((a, b) => b.length - a.length);
  const termTriggerRegex = triggerPatterns.length
    ? new RegExp(triggerPatterns.map(escapeRegex).join("|"), "gi")
    : null;

  const findGlossaryTermKeys = (...keywords) => {
    const kws = keywords.map((k) => normText(k)).filter(Boolean);
    if (!kws.length) return [];
    const out = new Set();
    glossaryTerms.forEach((t) => {
      if (kws.some((k) => t._searchBlob.includes(k))) out.add(t._key);
    });
    return [...out];
  };

  const chartTermGroups = {
    charEncoding: findGlossaryTermKeys("字符", "词表", "stoi", "编码", "token"),
    attentionCore: findGlossaryTermKeys("注意力", "softmax", "causal mask", "query", "key", "value"),
    blockInput: findGlossaryTermKeys("嵌入向量", "token", "n_embd"),
    blockNorm: findGlossaryTermKeys("LayerNorm", "层归一化"),
    blockAttn: findGlossaryTermKeys("注意力", "softmax", "query", "key", "value"),
    blockResidual: findGlossaryTermKeys("残差连接"),
    blockMLP: findGlossaryTermKeys("MLP", "GELU"),
    lossCurve: findGlossaryTermKeys("loss", "交叉熵", "训练集", "验证集"),
    lrWarmup: findGlossaryTermKeys("学习率", "warmup"),
    lrCosine: findGlossaryTermKeys("学习率", "cosine decay"),
    sampling: findGlossaryTermKeys("temperature", "top_k", "概率分布", "multinomial", "logits"),
    archModel: findGlossaryTermKeys("wte", "wpe", "n_layer", "n_head", "n_embd", "block_size", "vocab_size"),
    archTensor: findGlossaryTermKeys("block_size", "n_embd", "vocab_size", "logits"),
    archData: findGlossaryTermKeys("dataset", "get_batch", "x / y", "train split", "val split"),
    archParams: findGlossaryTermKeys("embedding", "n_layer", "n_embd", "参数"),
    archFlops: findGlossaryTermKeys("MFU", "block_size", "n_layer", "n_head", "n_embd"),
    archTrainLoop: findGlossaryTermKeys("loss", "optimizer", "gradient_accumulation_steps", "grad_clip", "checkpoint"),
    archMemory: findGlossaryTermKeys("batch_size", "block_size", "n_embd", "fp16 / bf16"),
  };

  const blockStageTermMap = {
    x0: chartTermGroups.blockInput,
    ln1: chartTermGroups.blockNorm,
    attn: chartTermGroups.blockAttn,
    x1: chartTermGroups.blockResidual,
    ln2: chartTermGroups.blockNorm,
    mlp: chartTermGroups.blockMLP,
    x2: chartTermGroups.blockResidual,
  };

  let glossaryHighlightTimer = null;
  const clearGlossaryHighlights = () => {
    const termList = byId("termList");
    if (!termList) return;
    termList.querySelectorAll("details.term-card.term-highlight").forEach((card) => card.classList.remove("term-highlight"));
  };

  const highlightGlossaryTerms = (termKeys) => {
    const keys = [...new Set((termKeys || []).filter(Boolean))];
    if (!keys.length) return;
    const termList = byId("termList");
    const termSearch = byId("termSearch");
    if (!termList || !termSearch) return;

    const missing = keys.some((k) => !termList.querySelector(`details.term-card[data-term-key="${k}"]`));
    if (missing) {
      termSearch.value = "";
      renderGlossary();
    }

    clearGlossaryHighlights();
    keys.forEach((k) => {
      const card = termList.querySelector(`details.term-card[data-term-key="${k}"]`);
      if (card) {
        card.classList.add("term-highlight");
        card.open = true;
      }
    });

    if (glossaryHighlightTimer) clearTimeout(glossaryHighlightTimer);
    glossaryHighlightTimer = setTimeout(clearGlossaryHighlights, 2600);
  };

  const tooltipEl = byId("termTooltip");
  let activeTipEl = null;

  const positionTooltip = (x, y) => {
    if (!tooltipEl || tooltipEl.hidden) return;
    const gap = 14;
    let left = x + gap;
    let top = y + gap;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const rect = tooltipEl.getBoundingClientRect();
    if (left + rect.width > vw - 8) left = x - rect.width - gap;
    if (top + rect.height > vh - 8) top = y - rect.height - gap;
    tooltipEl.style.left = `${Math.max(8, left)}px`;
    tooltipEl.style.top = `${Math.max(8, top)}px`;
  };

  const showTooltipForTerm = (term, x, y) => {
    if (!tooltipEl || !term) return;
    tooltipEl.innerHTML = `
      <strong>${escapeHtml(term.name)} <span class="term-alias">${escapeHtml(term.alias)}</span></strong>
      <div>${escapeHtml(term.plain)}</div>
    `;
    tooltipEl.hidden = false;
    positionTooltip(x, y);
  };

  const hideTooltip = () => {
    if (!tooltipEl) return;
    tooltipEl.hidden = true;
    activeTipEl = null;
  };

  const annotateTechTerms = (root = document.body) => {
    if (!root || !termTriggerRegex) return;

    const skipTags = new Set(["SCRIPT", "STYLE", "TEXTAREA", "INPUT", "BUTTON", "CODE", "PRE", "CANVAS", "SVG"]);
    const walker = document.createTreeWalker(
      root,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          const parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;
          if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
          if (skipTags.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
          if (parent.closest(".tech-term, #termList, #termTooltip")) return NodeFilter.FILTER_REJECT;
          termTriggerRegex.lastIndex = 0;
          return termTriggerRegex.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
        },
      },
      false
    );

    const nodes = [];
    let n = walker.nextNode();
    while (n) {
      nodes.push(n);
      n = walker.nextNode();
    }

    nodes.forEach((node) => {
      const text = node.nodeValue;
      const fragment = document.createDocumentFragment();
      let last = 0;
      let hasTerm = false;
      termTriggerRegex.lastIndex = 0;

      let match = termTriggerRegex.exec(text);
      while (match) {
        const hit = match[0];
        const start = match.index;
        const end = start + hit.length;
        const prev = start > 0 ? text[start - 1] : "";
        const next = end < text.length ? text[end] : "";
        const isAsciiHit = isAsciiWordLike(hit);
        const boundaryOk = !isAsciiHit || (!/[A-Za-z0-9_]/.test(prev) && !/[A-Za-z0-9_]/.test(next));
        const termKey = triggerToKey.get(normText(hit));

        if (termKey && boundaryOk) {
          if (start > last) fragment.appendChild(document.createTextNode(text.slice(last, start)));
          const span = document.createElement("span");
          span.className = "tech-term";
          span.tabIndex = 0;
          span.dataset.termKey = termKey;
          span.textContent = hit;
          const term = termByKey.get(termKey);
          if (term) span.setAttribute("aria-label", `${term.name}: ${term.plain}`);
          fragment.appendChild(span);
          last = end;
          hasTerm = true;
        }
        match = termTriggerRegex.exec(text);
      }

      if (hasTerm) {
        if (last < text.length) fragment.appendChild(document.createTextNode(text.slice(last)));
        node.parentNode.replaceChild(fragment, node);
      }
    });
  };

  const renderGlossary = () => {
    const termSearch = byId("termSearch");
    const termCount = byId("termCount");
    const termList = byId("termList");
    if (!termSearch || !termCount || !termList) return;

    const q = normText(termSearch.value);
    const filtered = glossaryTerms.filter((t) => (q ? t._searchBlob.includes(q) : true));

    termCount.textContent = `${filtered.length}/${glossaryTerms.length}`;

    if (!filtered.length) {
      termList.innerHTML = `<div class="card">没有匹配术语，请换一个关键词。</div>`;
      return;
    }

    const row = (label, value) => `<div class="term-row"><b>${label}:</b> ${escapeHtml(value)}</div>`;

    termList.innerHTML = filtered
      .map(
        (t) => `
      <details class="term-card" data-term-key="${t._key}">
        <summary>
          <span class="term-head">
            <span class="term-name">${escapeHtml(t.name)}</span>
            <span class="term-alias">${escapeHtml(t.alias)}</span>
          </span>
          <span class="term-level">${escapeHtml(t.level)}</span>
        </summary>
        <div class="term-body">
          ${row("一句话", t.plain)}
          ${row("详细解释", t.detail)}
          ${row("例子", t.example)}
          ${row("初中类比", t.analogy)}
          ${row("常见迷糊点", t.mistake)}
          ${row("本页对应", t.scene)}
        </div>
      </details>
    `
      )
      .join("");

    if (!q) {
      const firstCards = termList.querySelectorAll("details.term-card");
      for (let i = 0; i < Math.min(3, firstCards.length); i += 1) firstCards[i].open = true;
    }
  };

  on("termSearch", "input", renderGlossary);
  on("expandTermsBtn", "click", () => {
    const termList = byId("termList");
    if (!termList) return;
    termList.querySelectorAll("details.term-card").forEach((item) => {
      item.open = true;
    });
  });
  on("collapseTermsBtn", "click", () => {
    const termList = byId("termList");
    if (!termList) return;
    termList.querySelectorAll("details.term-card").forEach((item) => {
      item.open = false;
    });
  });

  document.addEventListener("mouseover", (event) => {
    const target = event.target.closest(".tech-term");
    if (!target) return;
    const term = termByKey.get(target.dataset.termKey);
    if (!term) return;
    activeTipEl = target;
    showTooltipForTerm(term, event.clientX, event.clientY);
  });

  document.addEventListener("mousemove", (event) => {
    if (!activeTipEl || tooltipEl.hidden) return;
    positionTooltip(event.clientX, event.clientY);
  });

  document.addEventListener("mouseout", (event) => {
    if (!activeTipEl) return;
    const related = event.relatedTarget;
    if (related && related.closest && related.closest(".tech-term") === activeTipEl) return;
    if (!related || !related.closest || !related.closest(".tech-term")) hideTooltip();
  });

  document.addEventListener("focusin", (event) => {
    const target = event.target.closest(".tech-term");
    if (!target) return;
    const term = termByKey.get(target.dataset.termKey);
    if (!term) return;
    const rect = target.getBoundingClientRect();
    activeTipEl = target;
    showTooltipForTerm(term, rect.left + rect.width / 2, rect.bottom + 6);
  });

  document.addEventListener("focusout", (event) => {
    if (event.target.closest(".tech-term")) hideTooltip();
  });

  /* ---------------------- 架构图谱 ---------------------- */
  const drawArrow = (ctx, x1, y1, x2, y2, color = "#4d6880") => {
    const headLen = 7;
    const angle = Math.atan2(y2 - y1, x2 - x1);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y2 - headLen * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y2 - headLen * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  };

  const roundedRectPath = (ctx, x, y, w, h, r = 8) => {
    if (typeof ctx.roundRect === "function") {
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, r);
      return;
    }
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.lineTo(x + w - rr, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
    ctx.lineTo(x + w, y + h - rr);
    ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
    ctx.lineTo(x + rr, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
    ctx.lineTo(x, y + rr);
    ctx.quadraticCurveTo(x, y, x + rr, y);
  };

  const drawBox = (ctx, x, y, w, h, title, subtitle, fill = "#fffaf0", stroke = "#bfcbd4") => {
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.2;
    roundedRectPath(ctx, x, y, w, h, 8);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#2b4761";
    ctx.font = "12px LXGW WenKai, Source Han Sans SC, sans-serif";
    ctx.fillText(title, x + 8, y + 18);
    if (subtitle) {
      ctx.fillStyle = "#5a7087";
      ctx.font = "11px Cascadia Mono, Consolas, monospace";
      ctx.fillText(subtitle, x + 8, y + 35);
    }
  };

  const calcArchDerived = () => {
    const L = archState.nLayer;
    const H = archState.nHead;
    const C = archState.nEmbd;
    const T = archState.blockSize;
    const V = archState.vocabSize;
    const headDimRaw = C / H;
    const headDim = Math.max(1, Math.floor(headDimRaw));
    archState.warning = Number.isInteger(headDimRaw)
      ? ""
      : `提示: n_embd(${C}) 不能被 n_head(${H}) 整除，演示中按 head_dim=${headDim} 近似。`;

    const tokenEmb = V * C;
    const posEmb = T * C;
    const attnPerLayer = 4 * C * C;
    const mlpPerLayer = 8 * C * C;
    const blockCore = L * (attnPerLayer + mlpPerLayer);
    const normsAndBias = L * (8 * C) + C;
    const total = tokenEmb + posEmb + blockCore + normsAndBias;
    archState.totalParams = total;

    return {
      L,
      H,
      C,
      T,
      V,
      headDim,
      tokenEmb,
      posEmb,
      attnTotal: L * attnPerLayer,
      mlpTotal: L * mlpPerLayer,
      normsAndBias,
      total,
    };
  };

  const formatParam = (n) => {
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return String(Math.round(n));
  };

  const updateArchLabels = () => {
    const labels = [
      ["archLayerValue", archState.nLayer],
      ["archHeadValue", archState.nHead],
      ["archEmbdValue", archState.nEmbd],
      ["archBlockValue", archState.blockSize],
      ["archVocabValue", archState.vocabSize],
    ];
    labels.forEach(([id, value]) => {
      const el = byId(id);
      if (el) el.textContent = `${value}`;
    });
  };

  const syncArchStateFromInputs = () => {
    const archLayer = byId("archLayer");
    const archHead = byId("archHead");
    const archEmbd = byId("archEmbd");
    const archBlock = byId("archBlock");
    const archVocab = byId("archVocab");
    if (!archLayer || !archHead || !archEmbd || !archBlock || !archVocab) return;

    archState.nLayer = Number(archLayer.value);
    archState.nHead = Number(archHead.value);
    archState.nEmbd = Number(archEmbd.value);
    archState.blockSize = Number(archBlock.value);
    archState.vocabSize = Number(archVocab.value);
    updateArchLabels();
    drawArchitectureCharts();
  };

  const drawArchitectureFlowChart = () => {
    withCanvas("gptArchitectureCanvas", (ctx, w, h) => {
      const d = calcArchDerived();
      ctx.fillStyle = "#fffdf6";
      ctx.fillRect(0, 0, w, h);
      const hitZones = [];

      const nodes = [
        {
          id: "input",
          x: 12,
          y: 28,
          w: 118,
          h: 48,
          title: "Token IDs",
          sub: `[B,T]=[8,${d.T}]`,
          terms: chartTermGroups.archTensor,
          fill: "#f8fbff",
        },
        {
          id: "embed",
          x: 152,
          y: 20,
          w: 136,
          h: 64,
          title: "wte + wpe",
          sub: `[B,T,C], C=${d.C}`,
          terms: chartTermGroups.archModel,
          fill: "#fff8ed",
        },
        {
          id: "drop",
          x: 308,
          y: 30,
          w: 86,
          h: 44,
          title: "Dropout",
          sub: "drop",
          terms: findGlossaryTermKeys("dropout"),
          fill: "#eef9ff",
        },
        {
          id: "blocks",
          x: 410,
          y: 12,
          w: 170,
          h: 82,
          title: `Block x ${d.L}`,
          sub: "LN -> Attn -> + -> LN -> MLP -> +",
          terms: [...chartTermGroups.blockNorm, ...chartTermGroups.blockAttn, ...chartTermGroups.blockMLP],
          fill: "#fff2e9",
        },
        {
          id: "lnf",
          x: 598,
          y: 30,
          w: 84,
          h: 44,
          title: "ln_f",
          sub: "LayerNorm",
          terms: chartTermGroups.blockNorm,
          fill: "#f2fcf9",
        },
        {
          id: "head",
          x: 698,
          y: 28,
          w: 96,
          h: 48,
          title: "lm_head",
          sub: `V=${d.V}`,
          terms: chartTermGroups.archModel,
          fill: "#ecf5ff",
        },
      ];

      const scale = Math.min(1, (w - 20) / 808);
      const offsetX = (w - 808 * scale) / 2;
      const offsetY = 18;
      const sx = (v) => offsetX + v * scale;
      const sy = (v) => offsetY + v * scale;
      const sw = (v) => v * scale;
      const sh = (v) => v * scale;

      nodes.forEach((n) => {
        const x = sx(n.x);
        const y = sy(n.y);
        const ww = sw(n.w);
        const hh = sh(n.h);
        drawBox(ctx, x, y, ww, hh, n.title, n.sub, n.fill);
        hitZones.push({ x, y, w: ww, h: hh, termKeys: n.terms });
      });

      for (let i = 0; i < nodes.length - 1; i += 1) {
        const a = nodes[i];
        const b = nodes[i + 1];
        drawArrow(
          ctx,
          sx(a.x + a.w),
          sy(a.y + a.h / 2),
          sx(b.x),
          sy(b.y + b.h / 2),
          "#4f6a84"
        );
      }

      ctx.fillStyle = "#435e76";
      ctx.font = "12px Cascadia Mono, Consolas, monospace";
      ctx.fillText(`heads=${d.H}, head_dim=${d.headDim}, total≈${formatParam(d.total)} params`, 14, h - 14);
      if (archState.warning) {
        ctx.fillStyle = "#b34a3a";
        ctx.fillText(archState.warning, 14, 16);
      }
      setCanvasHitZones("gptArchitectureCanvas", hitZones);
    });
  };

  const drawDataPipelineChart = () => {
    withCanvas("dataPipelineCanvas", (ctx, w, h) => {
      ctx.fillStyle = "#fffdf6";
      ctx.fillRect(0, 0, w, h);
      const hitZones = [];
      const boxes = [
        { x: 14, y: 20, w: 120, h: 46, title: "train.bin", sub: "dataset", terms: chartTermGroups.archData },
        { x: 14, y: 86, w: 120, h: 46, title: "val.bin", sub: "dataset", terms: chartTermGroups.archData },
        { x: 160, y: 52, w: 128, h: 56, title: "get_batch", sub: "x,y", terms: chartTermGroups.archData },
        { x: 314, y: 52, w: 122, h: 56, title: "forward", sub: "logits, loss", terms: chartTermGroups.archTensor },
        { x: 454, y: 18, w: 122, h: 44, title: "backward", sub: "grad", terms: chartTermGroups.archTrainLoop },
        { x: 454, y: 88, w: 122, h: 44, title: "optimizer", sub: "step", terms: chartTermGroups.archTrainLoop },
        { x: 594, y: 52, w: 124, h: 56, title: "checkpoint", sub: "ckpt.pt", terms: chartTermGroups.archTrainLoop },
      ];

      const scale = Math.min(1, (w - 20) / 734);
      const offsetX = (w - 734 * scale) / 2;
      const offsetY = 14;
      const sx = (v) => offsetX + v * scale;
      const sy = (v) => offsetY + v * scale;
      const sw = (v) => v * scale;
      const sh = (v) => v * scale;

      boxes.forEach((b) => {
        const x = sx(b.x);
        const y = sy(b.y);
        const ww = sw(b.w);
        const hh = sh(b.h);
        drawBox(ctx, x, y, ww, hh, b.title, b.sub, "#fff9ef", "#c7d3dc");
        hitZones.push({ x, y, w: ww, h: hh, termKeys: b.terms });
      });

      const arrow = (a, b, ay = 0, by = 0) =>
        drawArrow(ctx, sx(a.x + a.w), sy(a.y + a.h / 2 + ay), sx(b.x), sy(b.y + b.h / 2 + by), "#54708b");
      arrow(boxes[0], boxes[2], -8, -8);
      arrow(boxes[1], boxes[2], 8, 8);
      arrow(boxes[2], boxes[3]);
      drawArrow(
        ctx,
        sx(boxes[3].x + boxes[3].w),
        sy(boxes[3].y + 16),
        sx(boxes[4].x),
        sy(boxes[4].y + boxes[4].h / 2),
        "#54708b"
      );
      drawArrow(
        ctx,
        sx(boxes[4].x + boxes[4].w),
        sy(boxes[4].y + boxes[4].h / 2),
        sx(boxes[5].x),
        sy(boxes[5].y + boxes[5].h / 2),
        "#54708b"
      );
      drawArrow(
        ctx,
        sx(boxes[5].x + boxes[5].w),
        sy(boxes[5].y + boxes[5].h / 2),
        sx(boxes[6].x),
        sy(boxes[6].y + boxes[6].h / 2),
        "#54708b"
      );

      ctx.fillStyle = "#4e6479";
      ctx.font = "12px LXGW WenKai, Source Han Sans SC, sans-serif";
      ctx.fillText("train.py: get_batch -> model(X,Y) -> backward -> optimizer.step -> save ckpt", 14, h - 12);
      setCanvasHitZones("dataPipelineCanvas", hitZones);
    });
  };

  const drawTensorShapeChart = () => {
    withCanvas("tensorShapeCanvas", (ctx, w, h) => {
      const d = calcArchDerived();
      const B = 8;
      ctx.fillStyle = "#fffdf6";
      ctx.fillRect(0, 0, w, h);
      const hitZones = [];
      const steps = [
        { t: "idx", s: `[${B},${d.T}]`, terms: chartTermGroups.archTensor },
        { t: "tok_emb", s: `[${B},${d.T},${d.C}]`, terms: chartTermGroups.archModel },
        { t: "pos_emb", s: `[${d.T},${d.C}]`, terms: chartTermGroups.archModel },
        { t: "x+drop", s: `[${B},${d.T},${d.C}]`, terms: chartTermGroups.archTensor },
        { t: `Blocks x${d.L}`, s: `[${B},${d.T},${d.C}]`, terms: chartTermGroups.archModel },
        { t: "lm_head", s: `[${B},${d.T},${d.V}]`, terms: chartTermGroups.archTensor },
      ];

      const margin = { l: 16, r: 16, t: 24, b: 24 };
      const pw = w - margin.l - margin.r;
      const itemW = Math.min(134, (pw - (steps.length - 1) * 12) / steps.length);
      let x = margin.l;
      const y = 54;
      steps.forEach((s, i) => {
        drawBox(ctx, x, y, itemW, 60, s.t, s.s, i % 2 ? "#eef8ff" : "#fff7ee");
        hitZones.push({ x, y, w: itemW, h: 60, termKeys: s.terms });
        if (i < steps.length - 1) drawArrow(ctx, x + itemW, y + 30, x + itemW + 10, y + 30, "#4f6a84");
        x += itemW + 12;
      });

      ctx.fillStyle = "#4f667d";
      ctx.font = "12px Cascadia Mono, Consolas, monospace";
      ctx.fillText("inference: logits[:, [-1], :] -> [B,1,V]", margin.l, h - 10);
      setCanvasHitZones("tensorShapeCanvas", hitZones);
    });
  };

  const drawHeadGridChart = () => {
    withCanvas(
      "headGridCanvas",
      (ctx, w, h) => {
        const d = calcArchDerived();
        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, w, h);
        const hitZones = [];
        const cols = Math.ceil(Math.sqrt(d.H));
        const rows = Math.ceil(d.H / cols);
        const margin = { l: 16, r: 16, t: 24, b: 24 };
        const pw = w - margin.l - margin.r;
        const ph = h - margin.t - margin.b;
        const gap = 8;
        const cw = (pw - (cols - 1) * gap) / cols;
        const ch = (ph - (rows - 1) * gap) / rows;

        for (let i = 0; i < d.H; i += 1) {
          const r = Math.floor(i / cols);
          const c = i % cols;
          const x = margin.l + c * (cw + gap);
          const y = margin.t + r * (ch + gap);
          ctx.fillStyle = i % 2 ? "#edf7ff" : "#fdf1e6";
          ctx.strokeStyle = "#c8d1db";
          roundedRectPath(ctx, x, y, cw, ch, 6);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = "#2d4962";
          ctx.font = "11px Cascadia Mono, Consolas, monospace";
          ctx.fillText(`h${i}`, x + 6, y + 14);
          ctx.fillStyle = "#59708a";
          ctx.fillText(`d=${d.headDim}`, x + 6, y + 28);
          hitZones.push({
            x,
            y,
            w: cw,
            h: ch,
            termKeys: [...chartTermGroups.attentionCore, ...findGlossaryTermKeys("n_head", "n_embd")],
          });
        }
        if (archState.warning) {
          ctx.fillStyle = "#b44a39";
          ctx.font = "12px LXGW WenKai, Source Han Sans SC, sans-serif";
          ctx.fillText(archState.warning, 14, 14);
        }
        setCanvasHitZones("headGridCanvas", hitZones);
      },
      520,
      220
    );
  };

  const drawParamBreakdownChart = () => {
    withCanvas(
      "paramBreakdownCanvas",
      (ctx, w, h) => {
        const d = calcArchDerived();
        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, w, h);
        const hitZones = [];

        const parts = [
          { name: "token_emb", value: d.tokenEmb, color: "#2f8fbd", terms: findGlossaryTermKeys("wte", "vocab_size", "n_embd") },
          { name: "pos_emb", value: d.posEmb, color: "#3fb8a4", terms: findGlossaryTermKeys("wpe", "block_size", "n_embd") },
          { name: "attention", value: d.attnTotal, color: "#d35f29", terms: findGlossaryTermKeys("attention", "n_layer", "n_embd") },
          { name: "mlp", value: d.mlpTotal, color: "#c35a7f", terms: findGlossaryTermKeys("MLP", "n_layer", "n_embd") },
          { name: "norm+bias", value: d.normsAndBias, color: "#8c7be0", terms: findGlossaryTermKeys("LayerNorm", "bias") },
        ];

        const total = sum(parts.map((p) => p.value));
        const cx = w * 0.34;
        const cy = h * 0.52;
        const rOuter = Math.min(w, h) * 0.32;
        const rInner = rOuter * 0.56;
        let start = -Math.PI / 2;
        parts.forEach((p) => {
          const ang = (p.value / total) * Math.PI * 2;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.arc(cx, cy, rOuter, start, start + ang);
          ctx.closePath();
          ctx.fillStyle = p.color;
          ctx.fill();
          const mid = start + ang / 2;
          hitZones.push({
            x: cx + Math.cos(mid) * (rInner + 6) - 16,
            y: cy + Math.sin(mid) * (rInner + 6) - 16,
            w: 32,
            h: 32,
            termKeys: [...chartTermGroups.archParams, ...p.terms],
          });
          start += ang;
        });

        ctx.beginPath();
        ctx.fillStyle = "#fffdf6";
        ctx.arc(cx, cy, rInner, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#3d566d";
        ctx.font = "12px Cascadia Mono, Consolas, monospace";
        ctx.fillText("Total", cx - 18, cy - 4);
        ctx.fillText(formatParam(total), cx - 30, cy + 14);

        let ly = 22;
        parts.forEach((p) => {
          ctx.fillStyle = p.color;
          ctx.fillRect(w * 0.58, ly - 9, 10, 10);
          ctx.fillStyle = "#344f67";
          ctx.font = "11px Cascadia Mono, Consolas, monospace";
          const pct = ((p.value / total) * 100).toFixed(1);
          ctx.fillText(`${p.name} ${pct}%`, w * 0.58 + 14, ly);
          ly += 20;
        });

        setCanvasHitZones("paramBreakdownCanvas", [
          {
            x: cx - rOuter,
            y: cy - rOuter,
            w: rOuter * 2,
            h: rOuter * 2,
            termKeys: chartTermGroups.archParams,
          },
          ...hitZones,
        ]);
      },
      520,
      220
    );
  };

  const drawFlopsChart = () => {
    withCanvas(
      "flopsCanvas",
      (ctx, w, h) => {
        const d = calcArchDerived();
        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, w, h);

        const N = d.total;
        const L = d.L;
        const H = d.H;
        const Q = d.headDim;
        const tValues = [64, 128, 256, 512, 768, 1024, 1536, 2048].filter((t) => t <= Math.max(d.T, 2048));
        if (!tValues.includes(d.T)) tValues.push(d.T);
        tValues.sort((a, b) => a - b);
        const points = tValues.map((T) => {
          const flopsPerToken = 6 * N + 12 * L * H * Q * T;
          const flopsFwdbwd = flopsPerToken * T;
          return { T, F: flopsFwdbwd / 1e12 };
        });

        const margin = { l: 44, r: 14, t: 16, b: 28 };
        const pw = w - margin.l - margin.r;
        const ph = h - margin.t - margin.b;
        const minX = points[0].T;
        const maxX = points[points.length - 1].T;
        const minY = 0;
        const maxY = Math.max(...points.map((p) => p.F)) * 1.1;
        drawGrid(ctx, margin.l, margin.t, pw, ph, 4);

        const toXY = (p) => [
          margin.l + ((p.T - minX) / Math.max(maxX - minX, 1)) * pw,
          margin.t + ((maxY - p.F) / Math.max(maxY - minY, 1e-9)) * ph,
        ];

        ctx.strokeStyle = "#d35f29";
        ctx.lineWidth = 2;
        ctx.beginPath();
        points.forEach((p, i) => {
          const [x, y] = toXY(p);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();

        const hitZones = points.map((p) => {
          const [x, y] = toXY(p);
          ctx.fillStyle = "#2d8f9c";
          ctx.beginPath();
          ctx.arc(x, y, 3.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#3f5d76";
          ctx.font = "10px Cascadia Mono, Consolas, monospace";
          ctx.fillText(`${p.T}`, x - 8, margin.t + ph + 12);
          return { x: x - 6, y: y - 6, w: 12, h: 12, termKeys: chartTermGroups.archFlops };
        });

        ctx.fillStyle = "#4f667d";
        ctx.font = "11px Cascadia Mono, Consolas, monospace";
        ctx.fillText("approx fwdbwd TFLOPs/iter", margin.l, 12);
        setCanvasHitZones("flopsCanvas", [
          { x: margin.l, y: margin.t, w: pw, h: ph, termKeys: chartTermGroups.archFlops },
          ...hitZones,
        ]);
      },
      520,
      220
    );
  };

  const drawTrainTimelineChart = () => {
    withCanvas(
      "trainTimelineCanvas",
      (ctx, w, h) => {
        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, w, h);
        const stages = [
          { name: "get_batch", pct: 12, color: "#3ca6c4", terms: chartTermGroups.archData },
          { name: "forward", pct: 26, color: "#2f8fbd", terms: chartTermGroups.lossCurve },
          { name: "backward", pct: 28, color: "#d35f29", terms: chartTermGroups.archTrainLoop },
          { name: "clip", pct: 6, color: "#9a7bcf", terms: findGlossaryTermKeys("grad_clip") },
          { name: "opt.step", pct: 12, color: "#2f9c7c", terms: chartTermGroups.archTrainLoop },
          { name: "eval/save", pct: 16, color: "#c35a7f", terms: chartTermGroups.archTrainLoop },
        ];
        const hitZones = [];
        const margin = { l: 20, r: 20, t: 42, b: 28 };
        const pw = w - margin.l - margin.r;
        const barH = 34;
        let x = margin.l;
        stages.forEach((s) => {
          const ww = (s.pct / 100) * pw;
          ctx.fillStyle = s.color;
          ctx.fillRect(x, margin.t, ww, barH);
          ctx.strokeStyle = "#fffdf6";
          ctx.strokeRect(x, margin.t, ww, barH);
          ctx.fillStyle = "#13334d";
          ctx.font = "11px Cascadia Mono, Consolas, monospace";
          ctx.fillText(s.name, x + 4, margin.t + 20);
          hitZones.push({ x, y: margin.t, w: ww, h: barH, termKeys: s.terms });
          x += ww;
        });
        ctx.fillStyle = "#4c647b";
        ctx.font = "12px LXGW WenKai, Source Han Sans SC, sans-serif";
        ctx.fillText("单次训练迭代时序 (示意比例)", margin.l, 22);
        ctx.fillText("while True: forward -> backward -> scaler.step -> zero_grad", margin.l, h - 10);
        setCanvasHitZones("trainTimelineCanvas", hitZones);
      },
      520,
      220
    );
  };

  const drawMemoryChart = () => {
    withCanvas(
      "memoryCanvas",
      (ctx, w, h) => {
        const d = calcArchDerived();
        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, w, h);
        const batches = [4, 8, 16, 32, 48, 64];
        const dtypeBytes = 2; // fp16/bf16
        const factor = 6; // activations + grads rough factor
        const points = batches.map((b) => ({
          b,
          gb: (b * d.T * d.C * d.L * dtypeBytes * factor) / (1024 ** 3),
        }));
        const maxGb = Math.max(...points.map((p) => p.gb), 0.1);
        const margin = { l: 40, r: 14, t: 18, b: 32 };
        const pw = w - margin.l - margin.r;
        const ph = h - margin.t - margin.b;
        drawGrid(ctx, margin.l, margin.t, pw, ph, 4);
        const gap = 8;
        const bw = (pw - gap * (points.length - 1)) / points.length;
        const hitZones = [];
        points.forEach((p, i) => {
          const x = margin.l + i * (bw + gap);
          const hh = (p.gb / maxGb) * ph;
          const y = margin.t + ph - hh;
          ctx.fillStyle = i % 2 ? "#2f8fbd" : "#2ea486";
          ctx.fillRect(x, y, bw, hh);
          ctx.fillStyle = "#38526c";
          ctx.font = "10px Cascadia Mono, Consolas, monospace";
          ctx.fillText(`b${p.b}`, x, margin.t + ph + 13);
          ctx.fillText(`${p.gb.toFixed(1)}G`, x, y - 4);
          hitZones.push({ x, y, w: bw, h: hh, termKeys: chartTermGroups.archMemory });
        });
        ctx.fillStyle = "#4b647d";
        ctx.font = "11px Cascadia Mono, Consolas, monospace";
        ctx.fillText("activation memory rough estimate", margin.l, 12);
        setCanvasHitZones("memoryCanvas", hitZones);
      },
      520,
      220
    );
  };

  const drawArchitectureBigMap = () => {
    withCanvas(
      "architectureBigCanvas",
      (ctx, w, h) => {
        const d = calcArchDerived();
        const hitZones = [];
        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, w, h);

        const pad = 18;
        const leftW = Math.max(250, Math.floor(w * 0.26));
        const centerW = Math.max(300, Math.floor(w * 0.43));
        const rightW = w - leftW - centerW - pad * 4;
        const topY = 54;
        const rowH = Math.max(72, Math.floor((h - topY - 170) / 4));
        const gapY = 18;

        const lx = pad;
        const cx = lx + leftW + pad;
        const rx = cx + centerW + pad;
        const rowY = (i) => topY + i * (rowH + gapY);

        const addBox = (x, y, ww, hh, title, sub, fill, terms) => {
          drawBox(ctx, x, y, ww, hh, title, sub, fill, "#c7d3dc");
          hitZones.push({ x, y, w: ww, h: hh, termKeys: terms });
        };

        addBox(
          lx,
          rowY(0),
          leftW,
          rowH,
          "dataset",
          "train.bin / val.bin",
          "#f2f9ff",
          chartTermGroups.archData
        );
        addBox(
          lx,
          rowY(1),
          leftW,
          rowH,
          "get_batch",
          `B=8, T=${d.T}`,
          "#eefbff",
          chartTermGroups.archData
        );
        addBox(
          lx,
          rowY(2),
          leftW,
          rowH,
          "x,y shifted",
          "x -> y next token",
          "#eef7ff",
          [...chartTermGroups.archData, ...chartTermGroups.charEncoding]
        );
        addBox(
          lx,
          rowY(3),
          leftW,
          rowH,
          "train loop",
          "forward/backward/step",
          "#f7f5ff",
          chartTermGroups.archTrainLoop
        );

        addBox(
          cx,
          rowY(0),
          centerW,
          rowH,
          "Token + Position Embedding",
          `wte(${d.V}x${d.C}) + wpe(${d.T}x${d.C})`,
          "#fff6ea",
          chartTermGroups.archModel
        );
        addBox(
          cx,
          rowY(1),
          centerW,
          rowH,
          `Transformer Blocks x${d.L}`,
          "LN -> Attn -> + -> LN -> MLP -> +",
          "#fff2e9",
          [...chartTermGroups.blockNorm, ...chartTermGroups.blockAttn, ...chartTermGroups.blockMLP]
        );
        addBox(
          cx,
          rowY(2),
          centerW,
          rowH,
          "Self-Attention Heads",
          `n_head=${d.H}, head_dim=${d.headDim}`,
          "#eefbfb",
          [...chartTermGroups.attentionCore, ...findGlossaryTermKeys("n_head", "n_embd")]
        );
        addBox(
          cx,
          rowY(3),
          centerW,
          rowH,
          "ln_f + lm_head",
          `logits: [B,T,V], V=${d.V}`,
          "#f0f7ff",
          chartTermGroups.archTensor
        );

        addBox(
          rx,
          rowY(0),
          rightW,
          rowH,
          "Params",
          `total≈${formatParam(d.total)}`,
          "#fff7ee",
          chartTermGroups.archParams
        );
        addBox(
          rx,
          rowY(1),
          rightW,
          rowH,
          "Compute",
          "FLOPs ~ O(T^2)",
          "#fff3ef",
          chartTermGroups.archFlops
        );
        addBox(
          rx,
          rowY(2),
          rightW,
          rowH,
          "Memory",
          "activation + grad",
          "#eefbf5",
          chartTermGroups.archMemory
        );
        addBox(
          rx,
          rowY(3),
          rightW,
          rowH,
          "Sampling",
          "temperature / top_k",
          "#f7f5ff",
          chartTermGroups.sampling
        );

        drawArrow(ctx, lx + leftW, rowY(0) + rowH / 2, cx, rowY(0) + rowH / 2, "#4f6a84");
        drawArrow(ctx, lx + leftW, rowY(1) + rowH / 2, cx, rowY(1) + rowH / 2, "#4f6a84");
        drawArrow(ctx, lx + leftW, rowY(2) + rowH / 2, cx, rowY(2) + rowH / 2, "#4f6a84");
        drawArrow(ctx, cx + centerW, rowY(1) + rowH / 2, rx, rowY(1) + rowH / 2, "#4f6a84");
        drawArrow(ctx, cx + centerW, rowY(3) + rowH / 2, rx, rowY(3) + rowH / 2, "#4f6a84");
        drawArrow(ctx, lx + leftW / 2, rowY(2) + rowH, lx + leftW / 2, rowY(3), "#50708c");
        drawArrow(ctx, cx + centerW / 2, rowY(1) + rowH, cx + centerW / 2, rowY(2), "#50708c");

        ctx.fillStyle = "#3f5d76";
        ctx.font = "13px LXGW WenKai, Source Han Sans SC, sans-serif";
        ctx.fillText("nanoGPT 架构图谱总览 (单图)", pad, 26);
        ctx.font = "12px Cascadia Mono, Consolas, monospace";
        ctx.fillText(
          `n_layer=${d.L}  n_head=${d.H}  n_embd=${d.C}  block_size=${d.T}  vocab_size=${d.V}`,
          pad,
          44
        );
        if (archState.warning) {
          ctx.fillStyle = "#b44a39";
          ctx.fillText(archState.warning, pad, h - 124);
        }

        const summaryY = h - 102;
        const summaryH = 82;
        drawBox(
          ctx,
          pad,
          summaryY,
          w - pad * 2,
          summaryH,
          "Summary",
          `params≈${formatParam(d.total)} | attn=${formatParam(d.attnTotal)} | mlp=${formatParam(d.mlpTotal)}`,
          "#fffaf0",
          "#d4d1c6"
        );
        hitZones.push({
          x: pad,
          y: summaryY,
          w: w - pad * 2,
          h: summaryH,
          termKeys: [...chartTermGroups.archParams, ...chartTermGroups.archFlops, ...chartTermGroups.archMemory],
        });

        setCanvasHitZones("architectureBigCanvas", hitZones);
      },
      1080,
      920
    );
  };

  const drawArchitectureCharts = () => {
    drawArchitectureBigMap();
    drawArchitectureFlowChart();
    drawDataPipelineChart();
    drawTensorShapeChart();
    drawHeadGridChart();
    drawParamBreakdownChart();
    drawFlopsChart();
    drawTrainTimelineChart();
    drawMemoryChart();
    const archRoot = byId("arch");
    if (archRoot) annotateTechTerms(archRoot);
  };

  const randomChoice = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const randomizeArchConfig = () => {
    const archLayer = byId("archLayer");
    const archHead = byId("archHead");
    const archEmbd = byId("archEmbd");
    const archBlock = byId("archBlock");
    const archVocab = byId("archVocab");
    if (!archLayer || !archHead || !archEmbd || !archBlock || !archVocab) return;

    archLayer.value = String(randomChoice([6, 8, 10, 12, 16, 20]));
    archHead.value = String(randomChoice([4, 6, 8, 12, 16]));
    archEmbd.value = String(randomChoice([384, 512, 640, 768, 1024, 1280]));
    archBlock.value = String(randomChoice([256, 512, 768, 1024, 1536, 2048]));
    archVocab.value = String(randomChoice([20000, 30000, 40000, 50304, 56000]));
    syncArchStateFromInputs();
  };

  ["archLayer", "archHead", "archEmbd", "archBlock", "archVocab"].forEach((id) => {
    on(id, "input", syncArchStateFromInputs);
  });
  on("archRandomBtn", "click", randomizeArchConfig);

  /* ---------------------- 实验 1: 编码 + 频率图 ---------------------- */
  const drawCharFreqChart = () => {
    withCanvas(
      "charFreqCanvas",
      (ctx, w, h) => {
        const data = charState.freqEntries;
        const hitZones = [];
        if (!data.length) {
          setCanvasHitZones("charFreqCanvas", []);
          drawEmptyChart(ctx, w, h, "输入文本后显示频率图");
          return;
        }

        const margin = { l: 34, r: 14, t: 14, b: 42 };
        const pw = w - margin.l - margin.r;
        const ph = h - margin.t - margin.b;
        const maxV = Math.max(...data.map((d) => d.count), 1);

        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, w, h);
        drawGrid(ctx, margin.l, margin.t, pw, ph, 4);

        const gap = Math.max(4, pw * 0.02);
        const barW = (pw - gap * (data.length - 1)) / data.length;

        data.forEach((d, i) => {
          const x = margin.l + i * (barW + gap);
          const hh = (d.count / maxV) * ph;
          const y = margin.t + ph - hh;
          const grd = ctx.createLinearGradient(0, y, 0, y + hh);
          grd.addColorStop(0, "#1f9a9a");
          grd.addColorStop(1, "#3d74d8");
          ctx.fillStyle = grd;
          ctx.fillRect(x, y, barW, hh);
          ctx.fillStyle = "#4b5f74";
          ctx.font = "11px Cascadia Mono, Consolas, monospace";
          ctx.fillText(String(d.count), x, Math.max(10, y - 4));
          ctx.fillStyle = "#31465c";
          ctx.fillText(safeChar(d.ch), x, margin.t + ph + 14);
          hitZones.push({
            x,
            y,
            w: barW,
            h: hh,
            termKeys: chartTermGroups.charEncoding,
          });
        });

        ctx.strokeStyle = "#bfc8cf";
        ctx.strokeRect(margin.l, margin.t, pw, ph);
        setCanvasHitZones("charFreqCanvas", hitZones);
      },
      520,
      220
    );
  };

  const buildCharDemo = () => {
    const charInput = byId("charInput");
    const charHint = byId("charHint");
    const vocabView = byId("vocabView");
    const encodedView = byId("encodedView");
    const pairView = byId("pairView");
    if (!charInput || !charHint || !vocabView || !encodedView || !pairView) return;

    const text = charInput.value;
    if (!text) {
      charHint.textContent = "请输入至少 1 个字符。";
      vocabView.innerHTML = "";
      encodedView.innerHTML = "";
      pairView.innerHTML = "";
      charState.freqEntries = [];
      drawCharFreqChart();
      return;
    }

    const chars = [...text];
    const stoi = {};
    const itos = [];
    const freq = {};

    chars.forEach((ch) => {
      if (!(ch in stoi)) {
        stoi[ch] = itos.length;
        itos.push(ch);
      }
      freq[ch] = (freq[ch] || 0) + 1;
    });

    const encoded = chars.map((ch) => stoi[ch]);
    charState.freqEntries = itos.map((ch) => ({ ch, count: freq[ch] }));

    vocabView.innerHTML = itos
      .map((ch, i) => `<span class="chip"><b>${safeChar(ch)}</b> -> ${i}</span>`)
      .join("");

    encodedView.innerHTML = encoded
      .map((id, i) => `<span class="token">${i}:${id}</span>`)
      .join("");

    const pairRows = [];
    for (let i = 0; i < encoded.length - 1; i += 1) {
      const left = `${safeChar(chars[i])}(${encoded[i]})`;
      const right = `${safeChar(chars[i + 1])}(${encoded[i + 1]})`;
      pairRows.push(`<div class="pair-row">x[${i}] = ${left}  ->  y[${i}] = ${right}</div>`);
    }
    pairView.innerHTML = pairRows.join("");
    charHint.textContent =
      `共 ${chars.length} 个字符，去重后词表大小 ${itos.length}。右移配对对应 train.py:116。`;
    const exp1Root = byId("exp1");
    if (exp1Root) annotateTechTerms(exp1Root);
    drawCharFreqChart();
  };

  on("buildCharBtn", "click", buildCharDemo);
  on("randomTextBtn", "click", () => {
    const charInput = byId("charInput");
    if (!charInput) return;
    currentExample = (currentExample + 1) % textExamples.length;
    charInput.value = textExamples[currentExample];
    buildCharDemo();
  });
  on("charInput", "input", buildCharDemo);

  /* ---------------------- 实验 2: 注意力 + 热力图 ---------------------- */
  const renderTokenStrip = (focus) => {
    const tokenStrip = byId("tokenStrip");
    if (!tokenStrip) return;
    tokenStrip.innerHTML = attnState.tokens
      .map((tok, i) => {
        const cls = i === focus ? "token-pill focus" : i > focus ? "token-pill masked" : "token-pill";
        return `<span class="${cls}">${i}:${escapeHtml(tok)}</span>`;
      })
      .join("");
  };

  const renderScoreControls = () => {
    const focusIndex = byId("focusIndex");
    const root = byId("scoreControls");
    if (!focusIndex || !root) return;
    const focus = Number(focusIndex.value);
    root.innerHTML = "";

    attnState.tokens.forEach((tok, i) => {
      const row = document.createElement("div");
      row.className = "score-row";

      const left = document.createElement("div");
      left.textContent = `${i}: ${tok}`;
      row.appendChild(left);

      if (i > focus) {
        const masked = document.createElement("div");
        masked.className = "masked-tag";
        masked.textContent = "未来词，已屏蔽";
        row.appendChild(masked);
        const blank = document.createElement("div");
        blank.textContent = "-";
        row.appendChild(blank);
      } else {
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "-5";
        slider.max = "5";
        slider.step = "0.1";
        slider.value = String(attnState.scores[i]);
        slider.addEventListener("input", () => {
          attnState.scores[i] = Number(slider.value);
          updateAttention();
        });
        row.appendChild(slider);

        const val = document.createElement("div");
        val.textContent = fmtNum(attnState.scores[i], 1);
        row.appendChild(val);
      }
      root.appendChild(row);
    });
  };

  const buildAttentionMatrix = (focus) => {
    const n = attnState.tokens.length;
    const matrix = [];
    for (let i = 0; i < n; i += 1) {
      let rowScores;
      if (i === focus) {
        rowScores = attnState.tokens.map((_, j) => (j <= i ? attnState.scores[j] : -Infinity));
      } else {
        rowScores = attnState.tokens.map((_, j) => {
          if (j > i) return -Infinity;
          const nearBoost = -0.24 * (i - j);
          const tokenBias = 0.3 * Math.cos((i + 1) * (j + 1) * 0.25);
          return 1.1 + nearBoost + tokenBias;
        });
      }
      matrix.push(softmax(rowScores));
    }
    return matrix;
  };

  const drawAttentionHeatmap = (focus) => {
    withCanvas(
      "attnHeatmapCanvas",
      (ctx, w, h) => {
        const n = attnState.tokens.length;
        const hitZones = [];
        if (!n) {
          setCanvasHitZones("attnHeatmapCanvas", []);
          drawEmptyChart(ctx, w, h, "输入 token 后显示热力图");
          return;
        }

        const margin = { l: 72, r: 16, t: 36, b: 18 };
        const pw = w - margin.l - margin.r;
        const ph = h - margin.t - margin.b;
        const cell = Math.min(pw / n, ph / n);
        const gx = margin.l;
        const gy = margin.t;

        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, w, h);

        for (let i = 0; i < n; i += 1) {
          for (let j = 0; j < n; j += 1) {
            const v = (attnState.matrix[i] && attnState.matrix[i][j]) || 0;
            const x = gx + j * cell;
            const y = gy + i * cell;
            ctx.fillStyle = heatColor(v);
            ctx.fillRect(x, y, cell, cell);
            ctx.strokeStyle = "#efe8d9";
            ctx.strokeRect(x, y, cell, cell);
            hitZones.push({
              x,
              y,
              w: cell,
              h: cell,
              termKeys: chartTermGroups.attentionCore,
            });
          }
        }

        ctx.strokeStyle = "#365671";
        ctx.lineWidth = 2;
        ctx.strokeRect(gx, gy + focus * cell, cell * n, cell);
        ctx.lineWidth = 1;

        ctx.fillStyle = "#30485e";
        ctx.font = "11px Cascadia Mono, Consolas, monospace";
        for (let i = 0; i < n; i += 1) {
          const topLabel = `${i}`;
          const leftLabel = `${i}:${attnState.tokens[i].slice(0, 5)}`;
          ctx.fillText(topLabel, gx + i * cell + cell * 0.35, gy - 8);
          ctx.fillText(leftLabel, 8, gy + i * cell + cell * 0.63);
        }

        ctx.fillStyle = "#51657b";
        ctx.font = "12px LXGW WenKai, Source Han Sans SC, sans-serif";
        ctx.fillText("列: 被关注的 key", gx, 14);
        ctx.fillText("行: 当前 query", 8, 24);
        setCanvasHitZones("attnHeatmapCanvas", hitZones);
      },
      520,
      280
    );
  };

  const updateAttention = () => {
    const focusIndex = byId("focusIndex");
    const focusLabel = byId("focusLabel");
    const attnBars = byId("attnBars");
    const attnExplain = byId("attnExplain");
    if (!focusIndex || !focusLabel || !attnBars || !attnExplain) return;

    const focus = Number(focusIndex.value);
    focusLabel.textContent = String(focus);
    renderTokenStrip(focus);
    renderScoreControls();

    const scores = attnState.tokens.map((_, i) => (i <= focus ? attnState.scores[i] : -Infinity));
    const probs = softmax(scores);
    attnState.probs = probs;
    attnState.matrix = buildAttentionMatrix(focus);

    const tokenValues = attnState.tokens.map((tok, i) => tok.length + i * 0.2 + 1);
    const context = sum(probs.map((p, i) => p * tokenValues[i]));

    attnBars.innerHTML = attnState.tokens
      .map((tok, i) => {
        const pct = probs[i] * 100;
        const cls = i > focus ? "bar-fill masked" : "bar-fill";
        return `
          <div class="bar-row">
            <div>${i}:${escapeHtml(tok)}</div>
            <div class="bar-track"><div class="${cls}" style="width:${pct.toFixed(2)}%"></div></div>
            <div>${pct.toFixed(1)}%</div>
          </div>
        `;
      })
      .join("");

    attnExplain.textContent =
      `第 ${focus} 个 token 只能看左边(含自己)。未来词被屏蔽后概率为 0。简化 context=${fmtNum(context)}。`;

    const exp2Root = byId("exp2");
    if (exp2Root) annotateTechTerms(exp2Root);
    drawAttentionHeatmap(focus);
  };

  const applyAttentionInput = () => {
    const attnInput = byId("attnInput");
    const focusSlider = byId("focusIndex");
    if (!attnInput || !focusSlider) return;

    const raw = attnInput.value.trim();
    const tokens = raw.length ? raw.split(/\s+/) : [];
    attnState.tokens = tokens.length ? tokens : ["I", "am", "learning", "nanoGPT", "today"];
    attnState.scores = attnState.tokens.map((_, i) => Number((1.8 - i * 0.25).toFixed(1)));

    focusSlider.max = String(Math.max(attnState.tokens.length - 1, 0));
    focusSlider.value = String(Math.max(attnState.tokens.length - 1, 0));
    updateAttention();
  };

  on("focusIndex", "input", updateAttention);
  on("attnApplyBtn", "click", applyAttentionInput);
  on("attnInput", "change", applyAttentionInput);

  /* ---------------------- 实验 3: Block + 图表 ---------------------- */
  const randomVec = () => Array.from({ length: 4 }, () => Number((Math.random() * 2 - 1).toFixed(2)));
  const addVec = (a, b) => a.map((x, i) => x + b[i]);

  const normalize = (v) => {
    const mean = sum(v) / v.length;
    const variance = sum(v.map((x) => (x - mean) ** 2)) / v.length;
    const std = Math.sqrt(Math.max(variance, 1e-6));
    return v.map((x) => (x - mean) / std);
  };

  const l2Norm = (v) => Math.sqrt(sum(v.map((x) => x * x)));

  const runBlockMath = (x0) => {
    const ln1 = normalize(x0);
    const attn = ln1.map((x, i) => 0.62 * x + 0.38 * ln1[(i + 1) % ln1.length]);
    const x1 = addVec(x0, attn);
    const ln2 = normalize(x1);
    const mlp = ln2.map((x) => Math.max(0, 1.7 * x - 0.12));
    const x2 = addVec(x1, mlp);
    return { x0, ln1, attn, x1, ln2, mlp, x2 };
  };

  const drawBlockEnergyChart = () => {
    withCanvas(
      "blockEnergyCanvas",
      (ctx, w, h) => {
        const hitZones = [];
        if (!blockState.vectors.x0) {
          setCanvasHitZones("blockEnergyCanvas", []);
          drawEmptyChart(ctx, w, h, "点击重置向量后显示");
          return;
        }
        const data = stageDefs.map((s) => ({ key: s.key, value: l2Norm(blockState.vectors[s.key]) }));
        const margin = { l: 36, r: 14, t: 14, b: 40 };
        const pw = w - margin.l - margin.r;
        const ph = h - margin.t - margin.b;
        const maxV = Math.max(...data.map((d) => d.value), 1);
        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, w, h);
        drawGrid(ctx, margin.l, margin.t, pw, ph, 4);

        const gap = Math.max(4, pw * 0.015);
        const bw = (pw - gap * (data.length - 1)) / data.length;
        data.forEach((d, i) => {
          const x = margin.l + i * (bw + gap);
          const hh = (d.value / maxV) * ph;
          const y = margin.t + ph - hh;
          const active = i === blockState.step;
          ctx.fillStyle = active ? "#d35f29" : "#1f9a9a";
          ctx.fillRect(x, y, bw, hh);
          ctx.fillStyle = "#2f4a66";
          ctx.font = "10px Cascadia Mono, Consolas, monospace";
          ctx.fillText(stageDefs[i].key, x, margin.t + ph + 12);
          hitZones.push({
            x,
            y,
            w: bw,
            h: hh,
            termKeys: blockStageTermMap[d.key] || chartTermGroups.blockInput,
          });
        });
        ctx.strokeStyle = "#bfc8cf";
        ctx.strokeRect(margin.l, margin.t, pw, ph);
        setCanvasHitZones("blockEnergyCanvas", hitZones);
      },
      520,
      220
    );
  };

  const drawBlockVectorChart = () => {
    withCanvas(
      "blockVectorCanvas",
      (ctx, w, h) => {
        const hitZones = [];
        if (!blockState.vectors.x0) {
          setCanvasHitZones("blockVectorCanvas", []);
          drawEmptyChart(ctx, w, h, "点击下一步查看分量");
          return;
        }
        const key = stageDefs[blockState.step].key;
        const vec = blockState.vectors[key];
        const margin = { l: 42, r: 14, t: 14, b: 36 };
        const pw = w - margin.l - margin.r;
        const ph = h - margin.t - margin.b;
        const maxAbs = Math.max(1, ...vec.map((v) => Math.abs(v)));

        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, w, h);
        drawGrid(ctx, margin.l, margin.t, pw, ph, 4);
        const zeroY = margin.t + (maxAbs / (2 * maxAbs)) * ph;
        ctx.strokeStyle = "#7c8da0";
        ctx.beginPath();
        ctx.moveTo(margin.l, zeroY);
        ctx.lineTo(margin.l + pw, zeroY);
        ctx.stroke();

        const gap = Math.max(8, pw * 0.04);
        const bw = (pw - gap * (vec.length - 1)) / vec.length;
        vec.forEach((v, i) => {
          const x = margin.l + i * (bw + gap);
          const hh = (Math.abs(v) / (2 * maxAbs)) * ph;
          const y = v >= 0 ? zeroY - hh : zeroY;
          ctx.fillStyle = v >= 0 ? "#2d9c9c" : "#c35a7f";
          ctx.fillRect(x, y, bw, hh);
          ctx.fillStyle = "#2f4a66";
          ctx.font = "11px Cascadia Mono, Consolas, monospace";
          ctx.fillText(`d${i}`, x, margin.t + ph + 12);
          ctx.fillText(fmtNum(v, 2), x, y - 4);
          hitZones.push({
            x,
            y: Math.min(y, zeroY),
            w: bw,
            h: Math.max(hh, 6),
            termKeys: blockStageTermMap[key] || chartTermGroups.blockInput,
          });
        });

        ctx.fillStyle = "#51657b";
        ctx.font = "12px LXGW WenKai, Source Han Sans SC, sans-serif";
        ctx.fillText(`当前阶段: ${key}`, margin.l, 12);
        setCanvasHitZones("blockVectorCanvas", hitZones);
      },
      520,
      220
    );
  };

  const renderPipeline = () => {
    const pipeline = byId("pipeline");
    const blockExplain = byId("blockExplain");
    if (!pipeline || !blockExplain) return;

    const html = stageDefs
      .map((s, i) => {
        const isActive = i === blockState.step;
        const vec = blockState.vectors[s.key];
        return `
          <div class="step-card ${isActive ? "active" : ""}">
            <div class="step-title">${s.title}</div>
            <div class="vec">${fmtVec(vec)}</div>
          </div>
        `;
      })
      .join("");
    pipeline.innerHTML = html;
    blockExplain.textContent = stageExplain[blockState.step];
    const exp3Root = byId("exp3");
    if (exp3Root) annotateTechTerms(exp3Root);
    drawBlockEnergyChart();
    drawBlockVectorChart();
  };

  const resetBlock = () => {
    blockState.step = 0;
    blockState.vectors = runBlockMath(randomVec());
    renderPipeline();
  };

  const nextBlockStep = () => {
    blockState.step = (blockState.step + 1) % stageDefs.length;
    renderPipeline();
  };

  on("nextStepBtn", "click", nextBlockStep);
  on("resetBlockBtn", "click", () => {
    const autoPlayBtn = byId("autoPlayBtn");
    if (blockState.autoTimer) {
      clearInterval(blockState.autoTimer);
      blockState.autoTimer = null;
      if (autoPlayBtn) autoPlayBtn.textContent = "自动播放";
    }
    resetBlock();
  });

  on("autoPlayBtn", "click", () => {
    const autoPlayBtn = byId("autoPlayBtn");
    if (!autoPlayBtn) return;
    if (blockState.autoTimer) {
      clearInterval(blockState.autoTimer);
      blockState.autoTimer = null;
      autoPlayBtn.textContent = "自动播放";
      return;
    }
    autoPlayBtn.textContent = "停止";
    blockState.autoTimer = setInterval(nextBlockStep, 780);
  });

  /* ---------------------- 实验 4A: 训练双图 ---------------------- */
  const scaleToLr = (x) => 10 ** (-5 + (Number(x) / 100) * 3);

  const updateTrainLabels = () => {
    const lrScale = byId("lrScale");
    const batchSize = byId("batchSize");
    const gradAccum = byId("gradAccum");
    const dropoutRate = byId("dropoutRate");
    const lrValue = byId("lrValue");
    const batchValue = byId("batchValue");
    const accumValue = byId("accumValue");
    const dropoutValue = byId("dropoutValue");
    if (
      !lrScale ||
      !batchSize ||
      !gradAccum ||
      !dropoutRate ||
      !lrValue ||
      !batchValue ||
      !accumValue ||
      !dropoutValue
    ) {
      return;
    }
    lrValue.textContent = scaleToLr(lrScale.value).toExponential(2);
    batchValue.textContent = batchSize.value;
    accumValue.textContent = gradAccum.value;
    dropoutValue.textContent = (Number(dropoutRate.value) / 100).toFixed(2);
  };

  const getScheduledLr = (i, steps, baseLr, warmupEnd) => {
    const minLr = baseLr / 10;
    if (i < warmupEnd) return (baseLr * (i + 1)) / (warmupEnd + 1);
    const ratio = (i - warmupEnd) / Math.max(steps - warmupEnd - 1, 1);
    const coeff = 0.5 * (1 + Math.cos(Math.PI * ratio));
    return minLr + coeff * (baseLr - minLr);
  };

  const drawLossChart = () => {
    withCanvas("lossCanvas", (ctx, w, h) => {
      const train = trainState.train;
      const val = trainState.val;
      if (!train.length) {
        setCanvasHitZones("lossCanvas", []);
        drawEmptyChart(ctx, w, h, "点击“运行 220 步”生成 loss 曲线");
        return;
      }

      const margin = { l: 44, r: 16, t: 16, b: 28 };
      const pw = w - margin.l - margin.r;
      const ph = h - margin.t - margin.b;
      const all = [...train, ...val];
      const minY = Math.min(...all) * 0.95;
      const maxY = Math.max(...all) * 1.05;

      ctx.fillStyle = "#fffdf6";
      ctx.fillRect(0, 0, w, h);
      drawGrid(ctx, margin.l, margin.t, pw, ph, 4);

      const toXY = (idx, v) => {
        const x = margin.l + (idx / Math.max(train.length - 1, 1)) * pw;
        const y = margin.t + ((maxY - v) / Math.max(maxY - minY, 1e-6)) * ph;
        return [x, y];
      };

      const drawLine = (arr, color) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        arr.forEach((v, i) => {
          const [x, y] = toXY(i, v);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      };

      drawLine(train, "#1f9a9a");
      drawLine(val, "#d35f29");

      ctx.fillStyle = "#4c6078";
      ctx.font = "12px Cascadia Mono, Consolas, monospace";
      ctx.fillText("train", margin.l + 8, margin.t + 12);
      ctx.fillText("val", margin.l + 60, margin.t + 12);
      ctx.fillStyle = "#1f9a9a";
      ctx.fillRect(margin.l - 10, margin.t + 5, 8, 3);
      ctx.fillStyle = "#d35f29";
      ctx.fillRect(margin.l + 44, margin.t + 5, 8, 3);

      ctx.fillStyle = "#5a6d80";
      ctx.fillText(fmtNum(minY, 2), 6, margin.t + ph);
      ctx.fillText(fmtNum(maxY, 2), 6, margin.t + 10);
      setCanvasHitZones("lossCanvas", [
        {
          x: margin.l,
          y: margin.t,
          w: pw,
          h: ph,
          termKeys: chartTermGroups.lossCurve,
        },
      ]);
    });
  };

  const drawLrChart = () => {
    withCanvas(
      "lrCanvas",
      (ctx, w, h) => {
        const lrs = trainState.lrs;
        if (!lrs.length) {
          setCanvasHitZones("lrCanvas", []);
          drawEmptyChart(ctx, w, h, "运行训练后显示 LR 计划");
          return;
        }

        const margin = { l: 50, r: 16, t: 16, b: 28 };
        const pw = w - margin.l - margin.r;
        const ph = h - margin.t - margin.b;
        const minY = Math.min(...lrs) * 0.95;
        const maxY = Math.max(...lrs) * 1.05;

        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, w, h);
        drawGrid(ctx, margin.l, margin.t, pw, ph, 4);

        const warmupX = margin.l + (trainState.warmupEnd / Math.max(lrs.length - 1, 1)) * pw;
        ctx.fillStyle = "rgba(211, 95, 41, 0.1)";
        ctx.fillRect(margin.l, margin.t, warmupX - margin.l, ph);

        const toXY = (idx, v) => {
          const x = margin.l + (idx / Math.max(lrs.length - 1, 1)) * pw;
          const y = margin.t + ((maxY - v) / Math.max(maxY - minY, 1e-12)) * ph;
          return [x, y];
        };

        ctx.strokeStyle = "#2d7aa7";
        ctx.lineWidth = 2;
        ctx.beginPath();
        lrs.forEach((v, i) => {
          const [x, y] = toXY(i, v);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();

        ctx.strokeStyle = "#c35f3a";
        ctx.beginPath();
        ctx.moveTo(warmupX, margin.t);
        ctx.lineTo(warmupX, margin.t + ph);
        ctx.stroke();

        ctx.fillStyle = "#52667b";
        ctx.font = "12px Cascadia Mono, Consolas, monospace";
        ctx.fillText("warmup", Math.max(margin.l + 2, warmupX - 40), margin.t + 12);
        ctx.fillText("cosine decay", Math.min(w - 110, warmupX + 8), margin.t + 12);
        ctx.fillText(maxY.toExponential(1), 6, margin.t + 10);
        ctx.fillText(minY.toExponential(1), 6, margin.t + ph);
        setCanvasHitZones("lrCanvas", [
          {
            x: margin.l,
            y: margin.t,
            w: Math.max(0, warmupX - margin.l),
            h: ph,
            termKeys: chartTermGroups.lrWarmup,
          },
          {
            x: warmupX,
            y: margin.t,
            w: Math.max(0, margin.l + pw - warmupX),
            h: ph,
            termKeys: chartTermGroups.lrCosine,
          },
        ]);
      },
      520,
      220
    );
  };

  const simulateTrain = () => {
    const lrScale = byId("lrScale");
    const batchSize = byId("batchSize");
    const gradAccum = byId("gradAccum");
    const dropoutRate = byId("dropoutRate");
    const trainSummary = byId("trainSummary");
    if (!lrScale || !batchSize || !gradAccum || !dropoutRate || !trainSummary) return;

    const baseLr = scaleToLr(lrScale.value);
    const batch = Number(batchSize.value);
    const accum = Number(gradAccum.value);
    const dropout = Number(dropoutRate.value) / 100;

    const steps = 220;
    const warmupEnd = Math.max(10, Math.round(steps * 0.12));
    const train = [];
    const val = [];
    const lrs = [];
    let loss = 3.2 + Math.random() * 0.15;

    for (let i = 0; i < steps; i += 1) {
      const lrNow = getScheduledLr(i, steps, baseLr, warmupEnd);
      lrs.push(lrNow);

      const progress = i / (steps - 1);
      const lrQuality = Math.exp(-Math.abs(Math.log10(lrNow) - Math.log10(6e-4)) * 0.85);
      const batchQuality = Math.min(1.2, 0.75 + Math.log2(batch) / 8 + accum / 20);
      const decay = 0.009 * lrQuality * batchQuality * (1 - dropout * 0.35) * (1 - progress * 0.55);
      const jitter = (Math.random() - 0.5) * (0.045 + (1 - lrQuality) * 0.04);
      const unstable = lrNow > 3e-3 ? Math.sin(i / 4) * (lrNow / 3e-3) * 0.018 : 0;

      loss = loss * (1 - decay) + jitter + unstable;
      loss = Math.max(0.62, loss);

      const valGap =
        0.1 + dropout * 0.18 + (batch < 16 ? 0.06 : 0) + Math.abs(Math.random() - 0.5) * 0.04;
      train.push(loss);
      val.push(loss + valGap);
    }

    trainState.train = train;
    trainState.val = val;
    trainState.lrs = lrs;
    trainState.warmupEnd = warmupEnd;
    drawLossChart();
    drawLrChart();

    const lastTrain = train[train.length - 1];
    const lastVal = val[val.length - 1];
    let note = "学习率接近常用区间，下降趋势较稳。";
    if (baseLr > 3e-3) note = "学习率偏大，曲线波动明显。";
    if (baseLr < 8e-5) note = "学习率偏小，下降速度偏慢。";
    trainSummary.textContent =
      `最终 train=${fmtNum(lastTrain)}, val=${fmtNum(lastVal)}。${note} 对照 train.py:231 与 train.py:255。`;
    const exp4Root = byId("exp4");
    if (exp4Root) annotateTechTerms(exp4Root);
  };

  on("lrScale", "input", updateTrainLabels);
  on("batchSize", "input", updateTrainLabels);
  on("gradAccum", "input", updateTrainLabels);
  on("dropoutRate", "input", updateTrainLabels);
  on("runTrainSimBtn", "click", simulateTrain);
  on("resetTrainSimBtn", "click", () => {
    trainState.train = [];
    trainState.val = [];
    trainState.lrs = [];
    const trainSummary = byId("trainSummary");
    if (trainSummary) trainSummary.textContent = "";
    drawLossChart();
    drawLrChart();
  });

  /* ---------------------- 实验 4B: 采样概率图 ---------------------- */
  const BOS = "<bos>";
  const EOS = "<eos>";
  const graph = {};

  const addEdge = (a, b) => {
    if (!graph[a]) graph[a] = {};
    graph[a][b] = (graph[a][b] || 0) + 1;
  };

  [
    "nano gpt reads many examples",
    "nano gpt predicts next token",
    "the model learns patterns from data",
    "attention lets each token look left",
    "small steps make the loss go down",
    "training repeats forward backward update",
    "temperature changes randomness of outputs",
    "top k keeps only likely words",
  ].forEach((line) => {
    const seq = [BOS, ...line.split(" "), EOS];
    for (let i = 0; i < seq.length - 1; i += 1) addEdge(seq[i], seq[i + 1]);
  });

  const candidateDist = (lastWord, temperature, topK) => {
    const nextMap = graph[lastWord] || graph[BOS];
    const entries = Object.entries(nextMap);
    const total = sum(entries.map(([, c]) => c));
    let probs = entries.map(([word, c]) => ({ word, p: c / Math.max(total, 1) }));

    probs = probs.map((x) => ({ word: x.word, p: x.p ** (1 / Math.max(temperature, 1e-6)) }));
    const z = sum(probs.map((x) => x.p)) || 1;
    probs = probs.map((x) => ({ word: x.word, p: x.p / z }));
    probs.sort((a, b) => b.p - a.p);
    probs = probs.slice(0, topK);
    const z2 = sum(probs.map((x) => x.p)) || 1;
    return probs.map((x) => ({ word: x.word, p: x.p / z2 }));
  };

  const sampleWord = (dist) => {
    let r = Math.random();
    for (let i = 0; i < dist.length; i += 1) {
      r -= dist[i].p;
      if (r <= 0) return dist[i].word;
    }
    return dist[dist.length - 1].word;
  };

  const drawCandidateChart = () => {
    withCanvas(
      "candidateCanvas",
      (ctx, w, h) => {
        const data = sampleState.dist;
        const hitZones = [];
        if (!data.length) {
          setCanvasHitZones("candidateCanvas", []);
          drawEmptyChart(ctx, w, h, "生成一步后显示候选概率图");
          return;
        }
        const margin = { l: 36, r: 14, t: 14, b: 38 };
        const pw = w - margin.l - margin.r;
        const ph = h - margin.t - margin.b;
        const maxV = Math.max(...data.map((d) => d.p), 1e-6);
        ctx.fillStyle = "#fffdf6";
        ctx.fillRect(0, 0, w, h);
        drawGrid(ctx, margin.l, margin.t, pw, ph, 4);

        const gap = Math.max(6, pw * 0.025);
        const bw = (pw - gap * (data.length - 1)) / data.length;
        data.forEach((d, i) => {
          const x = margin.l + i * (bw + gap);
          const hh = (d.p / maxV) * ph;
          const y = margin.t + ph - hh;
          ctx.fillStyle = i === 0 ? "#d35f29" : "#2f8fbd";
          ctx.fillRect(x, y, bw, hh);
          ctx.fillStyle = "#334c63";
          ctx.font = "10px Cascadia Mono, Consolas, monospace";
          ctx.fillText(d.word.slice(0, 6), x, margin.t + ph + 12);
          ctx.fillText(`${(d.p * 100).toFixed(1)}%`, x, y - 4);
          hitZones.push({
            x,
            y,
            w: bw,
            h: hh,
            termKeys: chartTermGroups.sampling,
          });
        });
        ctx.strokeStyle = "#bfc8cf";
        ctx.strokeRect(margin.l, margin.t, pw, ph);
        setCanvasHitZones("candidateCanvas", hitZones);
      },
      520,
      220
    );
  };

  const updateSampleLabels = () => {
    const temperatureRange = byId("temperatureRange");
    const topKRange = byId("topKRange");
    const temperatureValue = byId("temperatureValue");
    const topKValue = byId("topKValue");
    if (!temperatureRange || !topKRange || !temperatureValue || !topKValue) return;
    temperatureValue.textContent = (Number(temperatureRange.value) / 10).toFixed(1);
    topKValue.textContent = topKRange.value;
  };

  const syncSeed = () => {
    const seedInput = byId("seedInput");
    const genOutput = byId("genOutput");
    if (!seedInput || !genOutput) return;
    const seed = seedInput.value.trim().toLowerCase();
    sampleState.seq = seed ? seed.split(/\s+/) : ["nano"];
    genOutput.textContent = sampleState.seq.join(" ");
  };

  const renderCandidates = (dist) => {
    const candidateView = byId("candidateView");
    if (!candidateView) return;
    sampleState.dist = dist;
    candidateView.innerHTML = dist
      .map((x) => `<div class="candidate-row"><span>${escapeHtml(x.word)}</span><span>${(x.p * 100).toFixed(1)}%</span></div>`)
      .join("");
    const exp4Root = byId("exp4");
    if (exp4Root) annotateTechTerms(exp4Root);
    drawCandidateChart();
  };

  const generateOne = () => {
    const temperatureRange = byId("temperatureRange");
    const topKRange = byId("topKRange");
    const genOutput = byId("genOutput");
    if (!temperatureRange || !topKRange || !genOutput) return;
    if (!sampleState.seq.length) syncSeed();
    const temperature = Number(temperatureRange.value) / 10;
    const topK = Number(topKRange.value);
    const last = sampleState.seq[sampleState.seq.length - 1];
    const dist = candidateDist(last, temperature, topK);
    renderCandidates(dist);

    let next = sampleWord(dist);
    if (next === EOS) next = ".";
    sampleState.seq.push(next);
    genOutput.textContent = sampleState.seq.join(" ");
  };

  on("genOneBtn", "click", generateOne);
  on("genTenBtn", "click", () => {
    for (let i = 0; i < 10; i += 1) generateOne();
  });
  on("genResetBtn", "click", () => {
    const candidateView = byId("candidateView");
    syncSeed();
    sampleState.dist = [];
    if (candidateView) candidateView.innerHTML = "";
    drawCandidateChart();
  });
  on("seedInput", "change", syncSeed);
  on("temperatureRange", "input", updateSampleLabels);
  on("topKRange", "input", updateSampleLabels);

  const chartCanvasIds = [
    "architectureBigCanvas",
    "gptArchitectureCanvas",
    "dataPipelineCanvas",
    "tensorShapeCanvas",
    "headGridCanvas",
    "paramBreakdownCanvas",
    "flopsCanvas",
    "trainTimelineCanvas",
    "memoryCanvas",
    "charFreqCanvas",
    "attnHeatmapCanvas",
    "blockEnergyCanvas",
    "blockVectorCanvas",
    "lossCanvas",
    "lrCanvas",
    "candidateCanvas",
  ];

  const getCanvasHitZone = (canvasId, x, y) =>
    (canvasHitZones[canvasId] || []).find(
      (zone) => x >= zone.x && x <= zone.x + zone.w && y >= zone.y && y <= zone.y + zone.h
    );

  const bindChartTermInteractions = () => {
    chartCanvasIds.forEach((canvasId) => {
      const canvas = byId(canvasId);
      if (!canvas) return;

      canvas.addEventListener("click", (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const zone = getCanvasHitZone(canvasId, x, y);
        if (zone && zone.termKeys && zone.termKeys.length) highlightGlossaryTerms(zone.termKeys);
      });

      canvas.addEventListener("mousemove", (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        canvas.style.cursor = getCanvasHitZone(canvasId, x, y) ? "pointer" : "default";
      });

      canvas.addEventListener("mouseleave", () => {
        canvas.style.cursor = "default";
      });
    });
  };

  const redrawAllCharts = () => {
    drawArchitectureCharts();
    drawCharFreqChart();
    const focusIndex = byId("focusIndex");
    drawAttentionHeatmap(Number((focusIndex && focusIndex.value) || 0));
    drawBlockEnergyChart();
    drawBlockVectorChart();
    drawLossChart();
    drawLrChart();
    drawCandidateChart();
  };

  /* ---------------------- 初始化 ---------------------- */
  if (hasEl("termList") && hasEl("termSearch") && hasEl("termCount")) renderGlossary();
  if (hasEl("archLayer")) syncArchStateFromInputs();
  bindChartTermInteractions();
  annotateTechTerms(document.body);
  if (hasEl("charInput")) buildCharDemo();
  if (hasEl("attnInput") && hasEl("focusIndex")) applyAttentionInput();
  if (hasEl("pipeline")) resetBlock();
  if (hasEl("lrScale")) updateTrainLabels();
  if (hasEl("lossCanvas")) drawLossChart();
  if (hasEl("lrCanvas")) drawLrChart();
  if (hasEl("temperatureRange")) updateSampleLabels();
  if (hasEl("seedInput")) syncSeed();
  if (hasEl("candidateCanvas")) drawCandidateChart();

  window.addEventListener("resize", redrawAllCharts);
})();
