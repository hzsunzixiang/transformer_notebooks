"""01.4 Summarization — 自动摘要
将长文本压缩为简短摘要。
"""
from transformers import pipeline
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
print("Summarization (自动摘要)")
print("=" * 60)
print(f"\n原文:\n{text}\n")

summarizer = pipeline("summarization")
outputs = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
print(f"摘要:\n{outputs[0]['summary_text']}")
