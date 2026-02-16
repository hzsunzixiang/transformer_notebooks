"""01.1 Text Classification — 文本情感分类
使用 distilbert-base-uncased-finetuned-sst-2-english 模型判断文本的正面/负面情感。
"""
from transformers import pipeline, set_seed
import pandas as pd
import transformers, datasets
transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()

text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

print("=" * 60)
print("Text Classification (情感分类)")
print("=" * 60)
print(f"\n输入文本:\n{text}\n")

classifier = pipeline("text-classification")
outputs = classifier(text)
df = pd.DataFrame(outputs)
print("分类结果:")
print(df.to_string(index=False))
