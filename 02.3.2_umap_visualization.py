"""02.3.2 UMAP Visualization — 用 UMAP 可视化训练集特征
从 emotion_features.npz 加载特征，UMAP 降维后按情感类别绘制 hexbin 图。
依赖: 先运行 02.3.1_extract_features.py 生成 emotion_features.npz
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 1. 加载特征
# ============================================================
print("加载特征文件 emotion_features.npz ...")
data = np.load("emotion_features.npz")
X_train = data["X_train"]
y_train = data["y_train"]
print(f"X_train shape: {X_train.shape}")

# 获取标签名称
emotions = load_dataset("emotion")
labels = emotions["train"].features["label"].names
print(f"标签: {labels}")

# ============================================================
# 2. UMAP 降维
# ============================================================
print("\n" + "=" * 60)
print("Visualizing the Training Set (UMAP)")
print("=" * 60)

# 缩放特征到 [0, 1]
X_scaled = MinMaxScaler().fit_transform(X_train)
# UMAP 降维到 2 维
print("UMAP 降维中 (这可能需要一分钟)...")
mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
X_2d = mapper.embedding_

df_emb = pd.DataFrame(X_2d, columns=["X", "Y"])
df_emb["label"] = y_train

# ============================================================
# 3. 分类别绘制 hexbin 图
# ============================================================
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
plt.savefig("emotion_umap.png", dpi=150, bbox_inches="tight")
print("图片已保存为 emotion_umap.png")
plt.show()
