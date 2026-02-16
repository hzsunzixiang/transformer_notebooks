"""02.3.3 Simple Classifier — LogisticRegression + DummyClassifier 基线
从 emotion_features.npz 加载特征，训练分类器并绘制混淆矩阵。
依赖: 先运行 02.3.1_extract_features.py 生成 emotion_features.npz

注意: 在 ARM Mac 上 sklearn + matplotlib + datasets 共存于同一进程会导致
libomp 冲突 segfault。因此 sklearn 训练在子进程中执行，主进程只负责绘图。
"""
import os
import sys
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# ============================================================
# 1. 在子进程中训练分类器 (避免 libomp 冲突)
# ============================================================
RESULT_CACHE = "classifier_results.npz"

if not os.path.exists(RESULT_CACHE):
    print("在子进程中训练分类器 (避免 libomp 冲突)...")
    ret = subprocess.run([
        sys.executable, "-c", """
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from datasets import load_dataset

data = np.load("emotion_features.npz")
X_train, X_valid = data["X_train"], data["X_valid"]
y_train, y_valid = data["y_train"], data["y_valid"]
print(f"X_train: {X_train.shape}, X_valid: {X_valid.shape}")

# 获取标签名称
emotions = load_dataset("emotion")
labels = emotions["train"].features["label"].names

# LogisticRegression
print("Training LogisticRegression (max_iter=3000)...")
lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
lr_score = lr_clf.score(X_valid, y_valid)
print(f"LogisticRegression accuracy: {lr_score:.4f}")

# DummyClassifier 基线
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_score = dummy_clf.score(X_valid, y_valid)
print(f"DummyClassifier accuracy: {dummy_score:.4f}")
print(f"提升: {lr_score - dummy_score:.4f} ({(lr_score/dummy_score - 1)*100:.1f}%)")

# 保存结果
y_preds = lr_clf.predict(X_valid)
np.savez("classifier_results.npz",
         y_preds=y_preds, y_valid=y_valid,
         lr_score=np.array(lr_score),
         dummy_score=np.array(dummy_score))
# 标签单独存 json（字符串数组）
with open("classifier_labels.json", "w") as f:
    json.dump(labels, f)
print("结果已保存")
"""
    ])
    if ret.returncode != 0:
        print("子进程失败！", file=sys.stderr)
        sys.exit(1)
else:
    print(f"发现缓存文件 {RESULT_CACHE}，跳过训练")

# ============================================================
# 2. 加载结果并打印
# ============================================================
print("\n" + "=" * 60)
print("Confusion Matrix")
print("=" * 60)

results = np.load(RESULT_CACHE)
y_preds = results["y_preds"]
y_valid = results["y_valid"]
lr_score = float(results["lr_score"])
dummy_score = float(results["dummy_score"])

with open("classifier_labels.json") as f:
    labels = json.load(f)

print(f"LogisticRegression accuracy: {lr_score:.4f}")
print(f"DummyClassifier accuracy: {dummy_score:.4f}")

# ============================================================
# 3. 绘制混淆矩阵
# ============================================================
cm = confusion_matrix(y_valid, y_preds, normalize="true")
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
plt.title("Normalized confusion matrix")
plt.savefig("emotion_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("图片已保存为 emotion_confusion_matrix.png")
plt.show()
