"""01.2 Named Entity Recognition — 命名实体识别
识别文本中的人名(PER)、组织(ORG)、地点(LOC)、其他实体(MISC)。
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

print("=" * 60)
print("Named Entity Recognition (命名实体识别)")
print("=" * 60)
print(f"\n输入文本:\n{text}\n")

ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
df = pd.DataFrame(outputs)
print("识别到的实体:")
print(df.to_string(index=False))
