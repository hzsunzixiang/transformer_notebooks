"""03.1 Scaled Dot-Product Attention — 从零实现注意力机制
手动实现 Scaled Dot-Product Attention 的每一步:
token embedding → Q/K/V → 点积打分 → softmax → 加权求和。
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from torch import nn
from transformers import AutoTokenizer, AutoConfig

# ============================================================
# 1. 准备输入: tokenize + embedding
# ============================================================
print("=" * 60)
print("Scaled Dot-Product Attention — 从零实现")
print("=" * 60)

model_ckpt = "bert-base-uncased"
text = "time flies like an arrow"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(f"\n输入文本: {text}")
print(f"input_ids: {inputs.input_ids}")

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
inputs_embeds = token_emb(inputs.input_ids)
print(f"Token embeddings shape: {inputs_embeds.size()}")
# shape: [batch_size=1, seq_len=5, hidden_size=768]

# ============================================================
# 2. 逐步计算 Scaled Dot-Product Attention
# ============================================================
print("\n" + "=" * 60)
print("Step-by-step Attention Computation")
print("=" * 60)

# 自注意力: Q = K = V = 输入嵌入
query = key = value = inputs_embeds

# Step 1: 计算注意力分数 (Q·K^T / sqrt(d_k))
dim_k = key.size(-1)
scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
print(f"\n注意力分数 scores shape: {scores.size()}")
# shape: [1, 5, 5] — 每对 token 之间的相似度

# Step 2: softmax 归一化得到注意力权重
weights = F.softmax(scores, dim=-1)
print(f"注意力权重 weights shape: {weights.size()}")
print(f"权重行和 (应全为1): {weights.sum(dim=-1)}")

# Step 3: 加权求和 (weights × V)
attn_outputs = torch.bmm(weights, value)
print(f"注意力输出 shape: {attn_outputs.shape}")

# ============================================================
# 3. 封装为函数
# ============================================================
print("\n" + "=" * 60)
print("Encapsulated Function")
print("=" * 60)


def scaled_dot_product_attention(query, key, value):
    """Scaled Dot-Product Attention
    Args:
        query: [batch, seq_len, dim]
        key:   [batch, seq_len, dim]
        value: [batch, seq_len, dim]
    Returns:
        [batch, seq_len, dim] — 注意力加权后的输出
    """
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


output = scaled_dot_product_attention(query, key, value)
print(f"函数输出 shape: {output.shape}")
print(f"与逐步计算结果一致: {torch.allclose(output, attn_outputs)}")

# ============================================================
# 4. 可视化注意力权重 (heatmap)
# ============================================================
print("\n" + "=" * 60)
print("Attention Weights Visualization")
print("=" * 60)

tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
w = weights[0].detach().numpy()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(w, cmap="Blues", vmin=0, vmax=w.max())
ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, fontsize=11)
ax.set_yticklabels(tokens, fontsize=11)
ax.set_xlabel("Key (被关注的 token)")
ax.set_ylabel("Query (发起关注的 token)")
ax.set_title("Scaled Dot-Product Attention Weights")

# 在每个格子中标注数值
for i in range(len(tokens)):
    for j in range(len(tokens)):
        color = "white" if w[i, j] > w.max() * 0.6 else "black"
        ax.text(j, i, f"{w[i, j]:.3f}", ha="center", va="center",
                fontsize=9, color=color)

fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig("03.1_attention_weights.png", dpi=150)
plt.show()
print("图片已保存: 03.1_attention_weights.png")
