"""02.3.1 Extract Features — 提取隐藏状态特征
加载预训练 DistilBERT，对 emotion 数据集提取 [CLS] token 的 hidden state，
保存为 .npz 文件供后续脚本使用。
"""
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from utils import get_device

# ============================================================
# 1. 准备数据和模型
# ============================================================
print("加载 emotion 数据集并分词...")
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
emotions = load_dataset("emotion")


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

# ============================================================
# 2. 加载预训练模型
# ============================================================
print("\n" + "=" * 60)
print("Using Pretrained Models (加载预训练模型)")
print("=" * 60)

device = get_device()
print(f"使用设备: {device}")
model = AutoModel.from_pretrained(model_ckpt).to(device)

# ============================================================
# 3. 提取隐藏状态 (单条示例)
# ============================================================
print("\n" + "=" * 60)
print("Extracting the Last Hidden States")
print("=" * 60)

text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape: {inputs['input_ids'].size()}")

inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(f"Last hidden state shape: {outputs.last_hidden_state.size()}")
print(f"[CLS] token shape: {outputs.last_hidden_state[:, 0].size()}")

# ============================================================
# 4. 提取整个数据集的隐藏状态
# ============================================================
print("\n" + "=" * 60)
print("Creating a Feature Matrix")
print("=" * 60)


def extract_hidden_states(batch):
    inputs = {k: v.to(device) for k, v in batch.items()
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])
print("提取所有隐藏状态 (这可能需要几分钟)...")
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
print(f"新增列: {emotions_hidden['train'].column_names}")

X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
print(f"X_train shape: {X_train.shape}, X_valid shape: {X_valid.shape}")

# ============================================================
# 5. 保存特征到磁盘，供 02.3.2 / 02.3.3 使用
# ============================================================
output_file = "emotion_features.npz"
np.savez(output_file, X_train=X_train, X_valid=X_valid,
         y_train=y_train, y_valid=y_valid)
print(f"\n特征已保存到 {output_file}")
