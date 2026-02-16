"""02.4 Fine-Tuning — 微调 DistilBERT
使用 Trainer API 对 DistilBERT 进行端到端微调，实现情感分类。
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from utils import get_device

# ============================================================
# 1. 准备数据
# ============================================================
print("加载 emotion 数据集并分词...")
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
emotions = load_dataset("emotion")


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
labels = emotions["train"].features["label"].names

# ============================================================
# 2. 加载预训练分类模型
# ============================================================
print("\n" + "=" * 60)
print("Loading a Pretrained Model for Sequence Classification")
print("=" * 60)

device = get_device()
print(f"使用设备: {device}")

num_labels = 6
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))

# ============================================================
# 3. 定义评估指标
# ============================================================

def compute_metrics(pred):
    labels_ids = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels_ids, preds, average="weighted")
    acc = accuracy_score(labels_ids, preds)
    return {"accuracy": acc, "f1": f1}


# ============================================================
# 4. 训练模型
# ============================================================
print("\n" + "=" * 60)
print("Training the Model with Trainer API")
print("=" * 60)

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    eval_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,  # 设为 True 可推送到 Hub
    log_level="error")

trainer = Trainer(
    model=model, args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    tokenizer=tokenizer)

print("开始训练 (2 epochs)...")
trainer.train()

# ============================================================
# 5. 查看训练结果 — 混淆矩阵
# ============================================================
print("\n" + "=" * 60)
print("Confusion Matrix")
print("=" * 60)

preds_output = trainer.predict(emotions_encoded["validation"])
print(f"Metrics: {preds_output.metrics}")

y_preds = np.argmax(preds_output.predictions, axis=1)
y_valid = np.array(emotions_encoded["validation"]["label"])


def plot_confusion_matrix(y_preds, y_true, plot_labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=plot_labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


plot_confusion_matrix(y_preds, y_valid, labels)
