"""03.3 Full Transformer Encoder + Classification Head — 完整编码器与分类头
将 Positional Embeddings、多层 EncoderLayer 组装为完整 TransformerEncoder,
再加上分类头实现 TransformerForSequenceClassification。
"""
import torch
import torch.nn.functional as F
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
print(f"输入文本: {text}")
print(f"input_ids: {inputs.input_ids}")


# ============================================================
# 复用前面定义的组件 (忠于原文的逐步构建过程)
# ============================================================
def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_state = self.layer_norm_1(x)
        x = x + self.attention(hidden_state)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


# ============================================================
# 1. Positional Embeddings — 位置嵌入
# ============================================================
print("\n" + "=" * 60)
print("Positional Embeddings — 位置嵌入")
print("=" * 60)


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size,
                                             config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


embedding_layer = Embeddings(config)
emb_output = embedding_layer(inputs.input_ids)
print(f"Embeddings 输出 shape: {emb_output.size()}")
print(f"  = token_embedding + position_embedding → LayerNorm → Dropout")
print(f"  max_position_embeddings = {config.max_position_embeddings}")

# ============================================================
# 2. 完整 TransformerEncoder — 组装多层编码器
# ============================================================
print("\n" + "=" * 60)
print("TransformerEncoder — 完整编码器")
print("=" * 60)


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config)
                                     for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x


encoder = TransformerEncoder(config)
encoder_output = encoder(inputs.input_ids)
print(f"Encoder 输出 shape: {encoder_output.size()}")
print(f"  共 {config.num_hidden_layers} 层 TransformerEncoderLayer")

# ============================================================
# 3. 加分类头 — TransformerForSequenceClassification
# ============================================================
print("\n" + "=" * 60)
print("Adding a Classification Head — 分类头")
print("=" * 60)


class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder(x)[:, 0, :]  # select hidden state of [CLS] token
        x = self.dropout(x)
        x = self.classifier(x)
        return x


config.num_labels = 3
encoder_classifier = TransformerForSequenceClassification(config)
cls_output = encoder_classifier(inputs.input_ids)
print(f"分类输出 shape: {cls_output.size()}")
print(f"  取 [CLS] token (位置0) 的隐藏状态 → Dropout → Linear({config.hidden_size}, {config.num_labels})")
print(f"  输出 {config.num_labels} 个类别的 logits: {cls_output.detach()}")
