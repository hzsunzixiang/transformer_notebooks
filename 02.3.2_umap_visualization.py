"""02.3.2 UMAP Visualization — 用 UMAP 可视化训练集特征
从 emotion_features.npz 加载特征，UMAP 降维后按情感类别绘制 hexbin 图。
依赖: 先运行 02.3.1_extract_features.py 生成 emotion_features.npz

注意: 在 ARM Mac 上 matplotlib 与 numba/UMAP 共存于同一进程会导致 segfault
(libomp 冲突)。因此 UMAP 降维在子进程中执行，主进程只负责绘图。
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# ============================================================
# 1. 在子进程中执行 UMAP 降维 (避免 matplotlib 与 numba 冲突)
# ============================================================
UMAP_CACHE = "emotion_umap_2d.npy"

if not os.path.exists(UMAP_CACHE):
    print("UMAP 降维中 (子进程执行，避免 libomp 冲突)...")
    ret = subprocess.run([
        sys.executable, "-c", """
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP

data = np.load("emotion_features.npz")
X_scaled = MinMaxScaler().fit_transform(data["X_train"])
print(f"X_scaled shape: {X_scaled.shape}")
print("UMAP fitting (n_jobs=1, metric=cosine)...")
mapper = UMAP(n_components=2, metric="cosine", n_jobs=1).fit(X_scaled)
np.save("emotion_umap_2d.npy", mapper.embedding_)
print(f"完成，保存到 emotion_umap_2d.npy, shape={mapper.embedding_.shape}")
"""
    ])
    if ret.returncode != 0:
        print("UMAP 子进程失败！", file=sys.stderr)
        sys.exit(1)
else:
    print(f"发现缓存文件 {UMAP_CACHE}，跳过 UMAP 降维")

# ============================================================
# 2. 加载降维结果
# ============================================================
print("\n" + "=" * 60)
print("Visualizing the Training Set (UMAP)")
print("=" * 60)

X_2d = np.load(UMAP_CACHE)
data = np.load("emotion_features.npz")
y_train = data["y_train"]
print(f"UMAP 结果: {X_2d.shape}")

# 获取标签名称
emotions = load_dataset("emotion")
labels = emotions["train"].features["label"].names

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
