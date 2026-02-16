"""02.1 Dataset Exploration — 数据集探索
加载 Hugging Face emotion 数据集，探索结构，转为 DataFrame，绘制类别分布和推文长度分布。
"""
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from huggingface_hub import list_datasets

# ============================================================
# 1. 列出 Hub 上的数据集
# ============================================================
print("=" * 60)
print("A First Look at Hugging Face Datasets")
print("=" * 60)

all_datasets = [ds.id for ds in list_datasets(limit=100)]
print(f"获取了 {len(all_datasets)} 个数据集示例")
print(f"前 10 个: {all_datasets[:10]}")

# ============================================================
# 2. 加载 emotion 数据集
# ============================================================
print("\n加载 emotion 数据集...")
emotions = load_dataset("emotion")
print(emotions)

train_ds = emotions["train"]
print(f"\n训练集大小: {len(train_ds)}")
print(f"第一条样本: {train_ds[0]}")
print(f"列名: {train_ds.column_names}")
print(f"特征: {train_ds.features}")
print(f"前5条: {train_ds[:5]}")

# ============================================================
# 3. Sidebar: 从本地/远程文件加载数据集 (演示)
# ============================================================
print("\n" + "=" * 60)
print("Sidebar: 从远程 CSV 加载数据集")
print("=" * 60)
dataset_url = "https://huggingface.co/datasets/transformersbook/emotion-train-split/raw/main/train.txt"
emotions_remote = load_dataset("csv", data_files=dataset_url, sep=";",
                               names=["text", "label"])
print(f"远程加载的数据集: {emotions_remote}")

# ============================================================
# 4. 转换为 Pandas DataFrame
# ============================================================
print("\n" + "=" * 60)
print("From Datasets to DataFrames")
print("=" * 60)

emotions.set_format(type="pandas")
df = emotions["train"][:]
print(df.head())


def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)


df["label_name"] = df["label"].apply(label_int2str)
print("\n添加 label_name 列:")
print(df.head())

# ============================================================
# 5. 类别分布
# ============================================================
print("\n" + "=" * 60)
print("Looking at the Class Distribution")
print("=" * 60)

df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

# ============================================================
# 6. 推文长度分布
# ============================================================
print("\n" + "=" * 60)
print("How Long Are Our Tweets?")
print("=" * 60)

df["Words Per Tweet"] = df["text"].str.split().apply(len)
print(df.boxplot("Words Per Tweet", by="label_name", grid=False,
                 showfliers=False, color="black"))
plt.suptitle("")
plt.xlabel("")
plt.show()

# 重置格式，以便后续使用
emotions.reset_format()
