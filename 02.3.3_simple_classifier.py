"""02.3.3 Simple Classifier — LogisticRegression + DummyClassifier 基线
从 emotion_features.npz 加载特征，训练分类器并绘制混淆矩阵。
依赖: 先运行 02.3.1_extract_features.py 生成 emotion_features.npz
"""
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# ============================================================
# 1. 加载特征
# ============================================================
print("加载特征文件 emotion_features.npz ...")
data = np.load("emotion_features.npz")
X_train = data["X_train"]
X_valid = data["X_valid"]
y_train = data["y_train"]
y_valid = data["y_valid"]
print(f"X_train: {X_train.shape}, X_valid: {X_valid.shape}")

# 获取标签名称
emotions = load_dataset("emotion")
labels = emotions["train"].features["label"].names

# ============================================================
# 2. 训练 LogisticRegression
# ============================================================
print("\n" + "=" * 60)
print("Training a Simple Classifier (LogisticRegression)")
print("=" * 60)

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
lr_score = lr_clf.score(X_valid, y_valid)
print(f"LogisticRegression accuracy: {lr_score:.4f}")

# ============================================================
# 3. DummyClassifier 基线对比
# ============================================================
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_score = dummy_clf.score(X_valid, y_valid)
print(f"DummyClassifier (most_frequent) accuracy: {dummy_score:.4f}")
print(f"提升: {lr_score - dummy_score:.4f} ({(lr_score/dummy_score - 1)*100:.1f}%)")

# ============================================================
# 4. 混淆矩阵
# ============================================================
print("\n" + "=" * 60)
print("Confusion Matrix")
print("=" * 60)

y_preds = lr_clf.predict(X_valid)


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.savefig("emotion_confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("图片已保存为 emotion_confusion_matrix.png")
    plt.show()


plot_confusion_matrix(y_preds, y_valid, labels)
