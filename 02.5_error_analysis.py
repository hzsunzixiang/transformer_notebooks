"""02.5 Error Analysis & Inference — 错误分析与模型推理
对微调后的模型进行逐样本损失分析，检查高/低损失样本，并用 pipeline 做推理。
"""
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.functional import cross_entropy
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, pipeline)
from utils import get_device

# ============================================================
# 1. 准备数据 & 训练模型 (复用 02.4 逻辑)
# ============================================================
print("加载 emotion 数据集并分词...")
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
emotions = load_dataset("emotion")
labels = emotions["train"].features["label"].names


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

device = get_device()
print(f"使用设备: {device}")

num_labels = 6
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))


def compute_metrics(pred):
    labels_ids = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels_ids, preds, average="weighted")
    acc = accuracy_score(labels_ids, preds)
    return {"accuracy": acc, "f1": f1}


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
    push_to_hub=False,
    log_level="error",
    report_to="none")

trainer = Trainer(
    model=model, args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    processing_class=tokenizer)

print("开始训练 (2 epochs)...")
trainer.train()

# ============================================================
# 2. Error Analysis — 逐样本损失分析
# ============================================================
print("\n" + "=" * 60)
print("Error Analysis (错误分析)")
print("=" * 60)


def forward_pass_with_label(batch):
    """对每个 batch 计算损失和预测标签"""
    inputs = {k: v.to(device) for k, v in batch.items()
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device),
                             reduction="none")
    return {"loss": loss.cpu().numpy(),
            "predicted_label": pred_label.cpu().numpy()}


# 转为 PyTorch 格式并计算损失
emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])
emotions_encoded["validation"] = emotions_encoded["validation"].map(
    forward_pass_with_label, batched=True, batch_size=16)


def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)


# 创建分析 DataFrame
emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = df_test["predicted_label"].apply(label_int2str)

# 损失最高的样本 — 可能是标注错误或困难样本
print("\n损失最高的 10 个样本 (可能的标注错误):")
print(df_test.sort_values("loss", ascending=False).head(10).to_string())

# 损失最低的样本 — 模型最自信的预测
print("\n损失最低的 10 个样本 (最自信的预测):")
print(df_test.sort_values("loss", ascending=True).head(10).to_string())

# ============================================================
# 3. Pipeline Inference — 用 pipeline 做推理
# ============================================================
print("\n" + "=" * 60)
print("Saving & Using the Model (保存与推理)")
print("=" * 60)

# 注意: push_to_hub 设为 False，这里用公开的模型演示 pipeline
model_id = "transformersbook/distilbert-base-uncased-finetuned-emotion"
classifier = pipeline("text-classification", model=model_id)

custom_tweet = "I saw a movie today and it was really good."
print(f"\n输入文本: {custom_tweet}")
preds = classifier(custom_tweet, return_all_scores=True)

preds_df = pd.DataFrame(preds[0])
print(f"\n预测结果:\n{preds_df.to_string(index=False)}")

plt.bar(labels, 100 * preds_df["score"], color="C0")
plt.title(f'"{custom_tweet}"')
plt.ylabel("Class probability (%)")
plt.show()
