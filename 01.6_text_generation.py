"""01.6 Text Generation — 文本生成
给定前文（客户投诉 + 客服开头），模型自动续写客服回复。
"""
from transformers import pipeline, set_seed
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

response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response

print("=" * 60)
print("Text Generation (文本生成)")
print("=" * 60)
print(f"\n提示词 (prompt):\n{prompt}\n")
print("-" * 60)
print("模型续写:\n")

set_seed(42)
generator = pipeline("text-generation")
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])
