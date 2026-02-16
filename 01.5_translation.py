"""01.5 Translation — 机器翻译 (英 → 德)
使用 Helsinki-NLP/opus-mt-en-de 模型将英文翻译为德文。
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
print("Translation (机器翻译: English → German)")
print("=" * 60)
print(f"\n英文原文:\n{text}\n")

translator = pipeline("translation_en_to_de",
                       model="Helsinki-NLP/opus-mt-en-de")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(f"德文翻译:\n{outputs[0]['translation_text']}")
