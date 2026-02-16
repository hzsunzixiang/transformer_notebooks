"""03.4 The Decoder — 解码器与因果掩码
实现 Decoder 中的 Causal (Masked) Self-Attention:
使用下三角掩码确保每个 token 只能关注它之前的 token (自回归)。
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from torch import nn
from transformers import AutoTokenizer, AutoConfig

# ============================================================
# 准备输入
# ============================================================
model_ckpt = "bert-base-uncased"
text = "time flies like an arrow"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained(model_ckpt)

inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
inputs_embeds = token_emb(inputs.input_ids)

tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
print(f"输入文本: {text}")
print(f"Tokens: {tokens}")

# ============================================================
# 1. 因果掩码 (Causal Mask) — 下三角矩阵
# ============================================================
print("\n" + "=" * 60)
print("Causal Mask — 因果掩码 (下三角矩阵)")
print("=" * 60)

seq_len = inputs.input_ids.size(-1)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
print(f"\n掩码矩阵 ({seq_len}×{seq_len}):")
print(mask[0])
print("\n含义: mask[i][j]=1 表示 token i 可以关注 token j")
print("       mask[i][j]=0 表示 token i 不能关注 token j (未来的 token)")

# ============================================================
# 2. 带掩码的注意力分数
# ============================================================
print("\n" + "=" * 60)
print("Masked Attention Scores")
print("=" * 60)

query = key = value = inputs_embeds
dim_k = key.size(-1)
scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)

print(f"\n原始注意力分数:")
print(scores[0].detach().numpy().round(3))

# 将掩码为 0 的位置设为 -inf，softmax 后变为 0
scores_masked = scores.masked_fill(mask == 0, -float("inf"))
print(f"\n掩码后的注意力分数 (上三角为 -inf):")
print(scores_masked[0].detach().numpy().round(3))

# ============================================================
# 3. 封装: 带掩码的 Scaled Dot-Product Attention
# ============================================================
print("\n" + "=" * 60)
print("Masked Scaled Dot-Product Attention")
print("=" * 60)


def scaled_dot_product_attention(query, key, value, mask=None):
    """Scaled Dot-Product Attention (支持因果掩码)
    Args:
        query, key, value: [batch, seq_len, dim]
        mask: [1, seq_len, seq_len] 或 None
    """
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights.bmm(value)


# 编码器模式 (无掩码): 每个 token 可以关注所有 token
encoder_output = scaled_dot_product_attention(query, key, value)
print(f"编码器 (无掩码) 输出 shape: {encoder_output.shape}")

# 解码器模式 (有掩码): 每个 token 只能关注之前的 token
decoder_output = scaled_dot_product_attention(query, key, value, mask=mask)
print(f"解码器 (有掩码) 输出 shape: {decoder_output.shape}")

# ============================================================
# 4. 可视化: 编码器 vs 解码器的注意力权重对比 (heatmap)
# ============================================================
print("\n" + "=" * 60)
print("编码器 vs 解码器 — 注意力权重对比")
print("=" * 60)

# 编码器注意力权重
enc_weights = F.softmax(scores, dim=-1)[0].detach().numpy()
# 解码器注意力权重
dec_weights = F.softmax(scores_masked, dim=-1)[0].detach().numpy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, data, title in [
    (axes[0], enc_weights, "Encoder Attention (双向)"),
    (axes[1], dec_weights, "Decoder Attention (因果掩码)"),
]:
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=max(enc_weights.max(), dec_weights.max()))
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=10)
    ax.set_yticklabels(tokens, fontsize=10)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title(title, fontsize=12)
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            color = "white" if data[i, j] > data.max() * 0.6 else "black"
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color=color)

fig.colorbar(im, ax=axes, shrink=0.8)
plt.suptitle("Encoder vs Decoder — Attention Weights", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("03.4_encoder_vs_decoder_attention.png", dpi=150, bbox_inches="tight")
plt.show()
print("图片已保存: 03.4_encoder_vs_decoder_attention.png")

print("\n关键区别:")
print("  编码器: 'arrow' 可以关注 'time' (双向)")
print("  解码器: 'arrow' 只能关注 'time','flies','like','an','arrow' (单向)")
print("  这就是 GPT 等自回归模型的核心: 生成下一个词时不能偷看未来的词")
