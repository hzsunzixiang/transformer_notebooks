"""02.2.1 Tokenization Methods — 三种分词方法对比
演示字符级分词、词级分词、子词分词 (WordPiece) 的原理与区别。
"""
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

text = "Tokenizing text is a core task of NLP."
print(f"示例文本: {text}")

# ============================================================
# 1. Character Tokenization — 字符级分词
# ============================================================
print("\n" + "=" * 60)
print("Character Tokenization (字符级分词)")
print("=" * 60)

tokenized_text = list(text)
print(f"字符列表: {tokenized_text}")

# 构建字符→ID 映射
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(f"字符词表 ({len(token2idx)} 个字符): {token2idx}")

# 将字符映射为 ID
input_ids = [token2idx[token] for token in tokenized_text]
print(f"input_ids: {input_ids}")

# One-hot 编码演示
categorical_df = pd.DataFrame(
    {"Name": ["Bumblebee", "Optimus Prime", "Megatron"],
     "Label ID": [0, 1, 2]})
print(f"\n类别数据:\n{categorical_df}")
print(f"\nOne-hot 编码:\n{pd.get_dummies(categorical_df['Name'])}")

input_ids_tensor = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids_tensor, num_classes=len(token2idx))
print(f"\nOne-hot 张量形状: {one_hot_encodings.shape}")
print(f"Token: {tokenized_text[0]}")
print(f"Tensor index: {input_ids_tensor[0]}")
print(f"One-hot: {one_hot_encodings[0]}")

# ============================================================
# 2. Word Tokenization — 词级分词
# ============================================================
print("\n" + "=" * 60)
print("Word Tokenization (词级分词)")
print("=" * 60)

tokenized_text = text.split()
print(f"词列表: {tokenized_text}")

# ============================================================
# 3. Subword Tokenization — 子词分词 (WordPiece)
# ============================================================
print("\n" + "=" * 60)
print("Subword Tokenization (子词分词 - WordPiece)")
print("=" * 60)

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

encoded_text = tokenizer(text)
print(f"编码结果: {encoded_text}")

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(f"tokens: {tokens}")

print(f"还原文本: {tokenizer.convert_tokens_to_string(tokens)}")
print(f"词表大小: {tokenizer.vocab_size}")
print(f"最大上下文长度: {tokenizer.model_max_length}")
print(f"模型输入字段: {tokenizer.model_input_names}")
