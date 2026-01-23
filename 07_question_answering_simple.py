# Question Answering with Transformers (Simplified Version)
# This script demonstrates extractive QA without Haystack/Elasticsearch dependencies

import os
import subprocess
from utils import *
setup_chapter()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# Part 1: Load and Explore SQuAD Dataset
# =============================================================================
print("=" * 60)
print("Part 1: Loading SQuAD Dataset")
print("=" * 60)

from datasets import load_dataset
import pandas as pd

# Load SQuAD dataset (standard QA benchmark)
squad = load_dataset("squad")
print(f"Dataset splits: {list(squad.keys())}")
print(f"Train size: {len(squad['train'])}, Validation size: {len(squad['validation'])}")

# Examine structure
print("\nExample question-answer pair:")
example = squad["train"][0]
print(f"Question: {example['question']}")
print(f"Context: {example['context'][:200]}...")
print(f"Answer: {example['answers']['text'][0]}")

# Create dataframes for analysis
dfs = {}
for split in ["train", "validation"]:
    df = squad[split].to_pandas()
    df["answers.text"] = df["answers"].apply(lambda x: x["text"])
    df["answers.answer_start"] = df["answers"].apply(lambda x: x["answer_start"])
    dfs[split] = df

# Question type distribution
print("\nQuestion Type Distribution:")
counts = {}
question_types = ["What", "How", "Who", "When", "Where", "Why", "Which"]
for q in question_types:
    count = dfs["train"]["question"].str.startswith(q).sum()
    if count > 0:
        counts[q] = count

import matplotlib.pyplot as plt
pd.Series(counts).sort_values().plot.barh()
plt.title("Frequency of Question Types in SQuAD")
plt.xlabel("Count")
plt.tight_layout()
plt.show()

# =============================================================================
# Part 2: Tokenization for QA
# =============================================================================
print("\n" + "=" * 60)
print("Part 2: Tokenization for Question Answering")
print("=" * 60)

from transformers import AutoTokenizer

model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

question = "How much music can this hold?"
context = """An MP3 is about 1 MB/minute, so about 6000 hours depending on file size."""

inputs = tokenizer(question, context, return_tensors="pt")
print(f"\nTokenized input:")
print(tokenizer.decode(inputs["input_ids"][0]))

# Show token structure
input_df = pd.DataFrame.from_dict(tokenizer(question, context), orient="index")
print(f"\nInput structure:")
print(input_df)

# =============================================================================
# Part 3: QA Model Inference
# =============================================================================
print("\n" + "=" * 60)
print("Part 3: Question Answering Model")
print("=" * 60)

import torch
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)

with torch.no_grad():
    outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits

print(f"Input IDs shape: {inputs.input_ids.size()}")
print(f"Start logits shape: {start_logits.size()}")
print(f"End logits shape: {end_logits.size()}")

# Visualize start/end scores
import numpy as np

s_scores = start_logits.detach().numpy().flatten()
e_scores = end_logits.detach().numpy().flatten()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 6), sharex=True)
colors = ["C0" if s != np.max(s_scores) else "C1" for s in s_scores]
ax1.bar(x=tokens, height=s_scores, color=colors)
ax1.set_ylabel("Start Scores")
ax1.set_title("Start and End Token Scores (orange = highest)")
colors = ["C0" if s != np.max(e_scores) else "C1" for s in e_scores]
ax2.bar(x=tokens, height=e_scores, color=colors)
ax2.set_ylabel("End Scores")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Extract answer
start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits) + 1
answer_span = inputs["input_ids"][0][start_idx:end_idx]
answer = tokenizer.decode(answer_span)
print(f"\nQuestion: {question}")
print(f"Answer: {answer}")

# =============================================================================
# Part 4: Using the Pipeline API
# =============================================================================
print("\n" + "=" * 60)
print("Part 4: Question Answering Pipeline")
print("=" * 60)

from transformers import pipeline

pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Get top-k answers
results = pipe(question=question, context=context, top_k=3)
print("\nTop 3 answers:")
for i, res in enumerate(results):
    print(f"  {i+1}. '{res['answer']}' (score: {res['score']:.4f})")

# Handle impossible questions
result = pipe(
    question="Why is there no data?", 
    context=context, 
    handle_impossible_answer=True
)
print(f"\nImpossible question result: {result}")

# =============================================================================
# Part 5: Handling Long Contexts with Sliding Window
# =============================================================================
print("\n" + "=" * 60)
print("Part 5: Handling Long Contexts")
print("=" * 60)

# Many QA contexts exceed model's max length (512 tokens)
long_context = squad["train"][100]["context"]
long_question = squad["train"][100]["question"]

# Check token length distribution
def compute_input_length(row):
    inputs = tokenizer(row["question"], row["context"])
    return len(inputs["input_ids"])

sample_df = dfs["train"].sample(1000, random_state=42)
sample_df["n_tokens"] = sample_df.apply(compute_input_length, axis=1)

fig, ax = plt.subplots()
sample_df["n_tokens"].hist(bins=50, grid=False, ec="C0", ax=ax)
plt.xlabel("Number of tokens in question-context pair")
ax.axvline(x=512, ymin=0, ymax=1, linestyle="--", color="C1", 
           label="Maximum sequence length (512)")
plt.legend()
plt.ylabel("Count")
plt.title("Token Length Distribution in SQuAD")
plt.show()

# Demonstrate sliding window tokenization
tokenized_example = tokenizer(
    long_question, long_context,
    return_overflowing_tokens=True, 
    max_length=100, 
    stride=25
)

print(f"\nLong context split into {len(tokenized_example['input_ids'])} windows:")
for idx, window in enumerate(tokenized_example["input_ids"]):
    print(f"  Window #{idx}: {len(window)} tokens")

# =============================================================================
# Part 6: Evaluate on SQuAD Examples
# =============================================================================
print("\n" + "=" * 60)
print("Part 6: Evaluation Examples")
print("=" * 60)

# Test on several SQuAD examples
test_examples = [
    squad["validation"][i] for i in [0, 10, 50, 100, 200]
]

print("\nQA Results on SQuAD validation samples:")
for ex in test_examples:
    result = pipe(question=ex["question"], context=ex["context"])
    ground_truth = ex["answers"]["text"][0]
    print(f"\nQ: {ex['question'][:80]}...")
    print(f"  Predicted: {result['answer']}")
    print(f"  Ground truth: {ground_truth}")
    print(f"  Score: {result['score']:.4f}")

# =============================================================================
# Part 7: Fine-tuning QA Model (Demo)
# =============================================================================
print("\n" + "=" * 60)
print("Part 7: Fine-tuning Setup (Demo)")
print("=" * 60)

from transformers import TrainingArguments, Trainer, DefaultDataCollator

# Prepare a small subset for demo
small_train = squad["train"].select(range(100))
small_val = squad["validation"].select(range(50))

# Tokenize with answer positions
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        
        sequence_ids = inputs.sequence_ids(i)
        
        # Find start and end of context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        # If answer is not fully in context, label is (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Find token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

print("Tokenizing dataset...")
tokenized_train = small_train.map(
    preprocess_function, 
    batched=True, 
    remove_columns=small_train.column_names
)
tokenized_val = small_val.map(
    preprocess_function, 
    batched=True, 
    remove_columns=small_val.column_names
)

print(f"Tokenized train size: {len(tokenized_train)}")
print(f"Tokenized val size: {len(tokenized_val)}")

# Training arguments (demo - short training)
training_args = TrainingArguments(
    output_dir="qa-model-demo",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    push_to_hub=False,
    logging_steps=10,
    report_to="none",  # Disable TensorBoard to avoid tf compatibility issues
)

# Initialize fresh model for fine-tuning
from transformers import AutoModelForQuestionAnswering
finetune_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=finetune_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,  # Updated from deprecated 'tokenizer'
    data_collator=DefaultDataCollator(),
)

print("\nStarting fine-tuning (1 epoch demo)...")
trainer.train()

print("\n" + "=" * 60)
print("Question Answering Tutorial Complete!")
print("=" * 60)
print("""
Summary:
1. Loaded and explored SQuAD dataset
2. Understood QA tokenization (question + context)
3. Learned how start/end logits work
4. Used pipeline API for easy inference
5. Handled long contexts with sliding window
6. Set up fine-tuning workflow

For production QA systems, consider:
- RAG (Retrieval Augmented Generation)
- Dense passage retrieval
- Semantic search with embeddings
""")
