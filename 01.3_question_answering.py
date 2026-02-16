"""01.3 Question Answering — 阅读理解式问答
给定上下文和问题，模型从文本中抽取答案片段。
"""
from transformers import pipeline
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

question = "What does the customer want?"

print("=" * 60)
print("Question Answering (阅读理解问答)")
print("=" * 60)
print(f"\n上下文:\n{text}\n")
print(f"问题: {question}\n")

reader = pipeline("question-answering")
outputs = reader(question=question, context=text)
df = pd.DataFrame([outputs])
print("答案:")
print(df.to_string(index=False))
