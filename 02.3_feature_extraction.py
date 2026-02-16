"""02.3 Feature Extraction — 特征提取方法
使用预训练 DistilBERT 提取隐藏状态，用 UMAP 可视化，用 LogisticRegression 分类。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# 5. UMAP 可视化
# ============================================================
print("\n" + "=" * 60)
print("Visualizing the Training Set (UMAP)")
print("=" * 60)

from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

# 缩放特征到 [0, 1]
X_scaled = MinMaxScaler().fit_transform(X_train)
# UMAP 降维
mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
X_2d = mapper.embedding_

df_emb = pd.DataFrame(X_2d, columns=["X", "Y"])
df_emb["label"] = y_train

labels = emotions["train"].features["label"].names

fig, axes = plt.subplots(2, 3, figsize=(7, 5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
for i, (label, cmap) in enumerate(zip(labels, cmaps)):
    df_emb_sub = df_emb.query(f"label == {i}")
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                   gridsize=20, linewidths=(0,))
    axes[i].set_title(label)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.tight_layout()
plt.show()

# ============================================================
# 6. 训练 LogisticRegression 分类器
# ============================================================
print("\n" + "=" * 60)
print("Training a Simple Classifier (LogisticRegression)")
print("=" * 60)

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
lr_score = lr_clf.score(X_valid, y_valid)
print(f"LogisticRegression accuracy: {lr_score:.4f}")

# Dummy 基线
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_score = dummy_clf.score(X_valid, y_valid)
print(f"DummyClassifier (most_frequent) accuracy: {dummy_score:.4f}")

# 混淆矩阵
y_preds = lr_clf.predict(X_valid)

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

plot_confusion_matrix(y_preds, y_valid, labels)
