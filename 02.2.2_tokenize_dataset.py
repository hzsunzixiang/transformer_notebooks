"""02.2.2 Tokenize Dataset — 对整个数据集进行分词
使用 DistilBERT tokenizer 的 map() 方法批量处理 emotion 数据集。
"""
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from utils import get_device

# ============================================================
# 1. 加载数据集和 tokenizer
# ============================================================
print("加载 emotion 数据集...")
emotions = load_dataset("emotion")

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# ============================================================
# 2. 定义分词函数
# ============================================================

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


# 测试 tokenize 函数
print("\n" + "=" * 60)
print("Tokenizing the Whole Dataset")
print("=" * 60)

print("分词示例 (前2条):")
print(tokenize(emotions["train"][:2]))

# 特殊 token 对照表
tokens2ids = list(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids))
data = sorted(tokens2ids, key=lambda x: x[-1])
df_tokens = pd.DataFrame(data, columns=["Special Token", "Special Token ID"])
print(f"\n特殊 token:\n{df_tokens.T}")

# ============================================================
# 3. 对整个数据集进行分词
# ============================================================
print("\n对整个数据集进行分词...")
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(f"分词后列名: {emotions_encoded['train'].column_names}")

# ============================================================
# 4. 查看分词结果
# ============================================================
print("\n" + "=" * 60)
print("查看分词后的数据")
print("=" * 60)

# 查看前 3 条样本
for i in range(3):
    sample = emotions_encoded["train"][i]
    print(f"\n--- 样本 {i} ---")
    print(f"  原文: {sample['text']}")
    print(f"  标签: {sample['label']}")
    print(f"  input_ids ({len(sample['input_ids'])} tokens): {sample['input_ids'][:20]}...")
    print(f"  解码还原: {tokenizer.decode(sample['input_ids'], skip_special_tokens=True)}")
    print(f"  attention_mask: {sample['attention_mask'][:20]}...")

# 数据集统计
print(f"\n数据集各 split 大小:")
for split in emotions_encoded:
    print(f"  {split}: {len(emotions_encoded[split])} 条")

# ============================================================
# 5. 查看每个 token 的向量 (hidden state embedding)
# ============================================================
print("\n" + "=" * 60)
print("Token Embeddings — 每个 token 的向量表示")
print("=" * 60)

# 加载预训练模型
print("加载 DistilBERT 模型...")
device = get_device()
print(f"使用设备: {device}")
model = AutoModel.from_pretrained(model_ckpt).to(device)
model.eval()

# 取第 0 条样本做前向传播
sample_text = emotions["train"][0]["text"]
inputs = tokenizer(sample_text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
token_ids = inputs["input_ids"][0].tolist()
tokens = tokenizer.convert_ids_to_tokens(token_ids)

with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state 形状: [1, seq_len, hidden_dim]
hidden_states = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]

print(f"\n原文: {sample_text}")
print(f"Hidden state 形状: {hidden_states.shape}  (seq_len={hidden_states.shape[0]}, hidden_dim={hidden_states.shape[1]})")

for i, (tok, vec) in enumerate(zip(tokens, hidden_states)):
    vec_list = vec.tolist()
    print(f"\n  Token {i:2d}: {tok:15s}  (id={token_ids[i]})")
    print(f"    向量 (前10维): {[f'{v:.4f}' for v in vec_list[:10]]}")
    print(f"    向量 (后5维):  {[f'{v:.4f}' for v in vec_list[-5:]]}")
    print(f"    范数: {vec.norm().item():.4f}  均值: {vec.mean().item():.6f}  标准差: {vec.std().item():.4f}")
