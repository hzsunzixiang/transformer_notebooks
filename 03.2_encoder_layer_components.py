"""03.2 Multi-Head Attention + Feed-Forward + LayerNorm — 编码器层组件
从零实现 Transformer 编码器的三大组件:
1. AttentionHead + MultiHeadAttention (多头注意力)
2. FeedForward (前馈网络)
3. TransformerEncoderLayer (含 LayerNorm 和残差连接)
"""
import torch
import torch.nn.functional as F
from math import sqrt
from torch import nn
from transformers import AutoTokenizer, AutoConfig

# ============================================================
# 1. 准备输入
# ============================================================
model_ckpt = "bert-base-uncased"
text = "time flies like an arrow"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained(model_ckpt)

inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
inputs_embeds = token_emb(inputs.input_ids)
print(f"输入文本: {text}")
print(f"Token embeddings: {inputs_embeds.size()}")
print(f"BERT config: hidden_size={config.hidden_size}, "
      f"num_attention_heads={config.num_attention_heads}, "
      f"intermediate_size={config.intermediate_size}")


# ============================================================
# 2. Scaled Dot-Product Attention (复用 03.1 的实现)
# ============================================================
def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


# ============================================================
# 3. 单头注意力 AttentionHead
# ============================================================
print("\n" + "=" * 60)
print("AttentionHead — 单头注意力")
print("=" * 60)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs


head_dim = config.hidden_size // config.num_attention_heads
attn_head = AttentionHead(config.hidden_size, head_dim)
head_output = attn_head(inputs_embeds)
print(f"单头输出 shape: {head_output.size()}")
print(f"  head_dim = hidden_size / num_heads = {config.hidden_size} / {config.num_attention_heads} = {head_dim}")

# ============================================================
# 4. 多头注意力 MultiHeadAttention
# ============================================================
print("\n" + "=" * 60)
print("MultiHeadAttention — 多头注意力")
print("=" * 60)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


multihead_attn = MultiHeadAttention(config)
attn_output = multihead_attn(inputs_embeds)
print(f"多头输出 shape: {attn_output.size()}")
print(f"  {config.num_attention_heads} 个头, 每头 {head_dim} 维, "
      f"拼接后 {config.num_attention_heads}×{head_dim}={config.hidden_size} 维")

# ============================================================
# 5. 前馈网络 FeedForward
# ============================================================
print("\n" + "=" * 60)
print("FeedForward — 前馈网络")
print("=" * 60)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_output)
print(f"FeedForward 输出 shape: {ff_outputs.size()}")
print(f"  升维: {config.hidden_size} → {config.intermediate_size} → {config.hidden_size}")

# ============================================================
# 6. Transformer 编码器层 (含 LayerNorm + 残差连接)
# ============================================================
print("\n" + "=" * 60)
print("TransformerEncoderLayer — 编码器层 (Pre-LN)")
print("=" * 60)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


encoder_layer = TransformerEncoderLayer(config)
layer_output = encoder_layer(inputs_embeds)
print(f"输入 shape:  {inputs_embeds.shape}")
print(f"输出 shape:  {layer_output.size()}")
print(f"形状不变，但每个 token 的表示已经融合了全局上下文信息")
