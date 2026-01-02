"""
此脚本用于将原始文本数据集（如莎士比亚全集）转化为模型可训练的二进制格式。
它实现了 LLM 流程中的第一步：Tokenization（分词与编码）。
"""

import os
import requests
import numpy as np

# --- 第一阶段：数据获取 ---
# 定义本地存储路径
data_dir = os.path.join('data', 'shakespeare_char')
input_file_path = os.path.join(data_dir, 'input.txt')

# 如果本地没有数据，从网络下载莎士比亚全集
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path', 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

# 读取原始文本内容
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"数据文件长度（字符数）: {len(data):,}")

# --- 第二阶段：构建词表 (Vocabulary) ---
# 获取文本中所有出现的唯一字符
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("所有的字符集合:", ''.join(chars))
print(f"词表大小 (Vocab Size): {vocab_size}")

# 创建字符到数字的映射字典 (Encoder)
stoi = { ch:i for i,ch in enumerate(chars) }
# 创建数字到字符的映射字典 (Decoder)
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    """编码函数：输入字符串，返回对应的数字 ID 列表"""
    return [stoi[c] for c in s] 

def decode(l):
    """解码函数：输入数字列表，返回还原后的字符串"""
    return ''.join([itos[i] for i in l])

# --- 第三阶段：数据集划分 ---
# 将整个文本编码为整数序列
n = len(data)
train_data = data[:int(n*0.9)] # 前 90% 用于训练
val_data = data[int(n*0.9):]   # 后 10% 用于验证

# 将列表转化为 numpy 数组，使用 uint16 以节省内存（因为 65 < 65535）
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"训练集 token 数量: {len(train_ids):,}")
print(f"验证集 token 数量: {len(val_ids):,}")

# --- 第四阶段：保存为二进制文件 (Binary Files) ---
# 使用二进制格式存储是为了在训练时能通过内存映射（Memory Map）极速读取数据
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

# 顺便保存元数据（词表信息），供后续 sample.py 推理时解码使用
import pickle
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("数据准备完成！生成的 .bin 和 meta.pkl 已存入:", data_dir)