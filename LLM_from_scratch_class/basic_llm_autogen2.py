"""
Script to demonstrate basic LLM training on a tiny dataset.
Dataset is fixed to:
["he boy", "she girl"]
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

from LLM_from_scratch_class.GPTModel import GPTModel

class LLMDataset(Dataset):
    def __init__(self, texts, word_to_id, max_len):
        self.texts = texts
        self.word_to_id = word_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = torch.tensor(text_to_token_ids(self.texts[idx], self.word_to_id)[:self.max_len + 1])
        x = tokens[:-1]
        y = tokens[1:]
        return x, y

# -----------------------------
# 1) Tiny dataset (as requested)
# -----------------------------
data = ["he boy", "she girl"]

VOCAB_SIZE = 4
CONTEXT_LEN = 1
BATCH_SIZE = 2
EPOCHS = 100

torch.manual_seed(123)

print("data:", data)
print("type(data):", type(data))


# -----------------------------
# 2) Build vocabulary + mappings
# -----------------------------
vocab = set()
for sentence in data:
    for word in sentence.split():
        vocab.add(word)

vocab = sorted(vocab)
word_to_id = {word: idx for idx, word in enumerate(vocab)}
id_to_word = {idx: word for idx, word in enumerate(vocab)}

print("sorted vocab:", vocab)
print("word_to_id:", word_to_id)
print("id_to_word:", id_to_word)


def text_to_token_ids(text, mapping):
    tokens = text.split()
    return [mapping[token] for token in tokens]


def token_ids_to_text(token_ids, mapping):
    return " ".join([mapping[token_id] for token_id in token_ids])


example_text = data[0]
print("example text:", example_text)
print("example token ids:", text_to_token_ids(example_text, word_to_id))


# -----------------------------
# 3) Dataset + DataLoader
# -----------------------------
train_dataset = LLMDataset(
    data,
    word_to_id,
    CONTEXT_LEN
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("num batches:", len(train_dataloader))
example_input_output = next(iter(train_dataloader))
print("example batch:", example_input_output)


# -----------------------------
# 4) Model + optimizer + loss
# -----------------------------
model = GPTModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")

param_count = defaultdict(int)
for name, param in model.named_parameters():
    top_level = name.split(".")[0]
    param_count[top_level] += param.numel()

for k, v in param_count.items():
    print(f"{k:20s}: {v:,}")


# -----------------------------
# 5) Training loop
# -----------------------------
def train_one_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader, start=1):
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"batch: {batch_idx} loss: {loss.item():.6f}")


for epoch in range(EPOCHS):
    print("-" * 30)
    print(f"Epoch: {epoch + 1}")
    train_one_epoch(train_dataloader, model, loss_fn, optimizer)

print("Done training.")
model.eval()


# -----------------------------
# 6) Simple generation helper
# -----------------------------
def generate(start_text):
    # Keep input aligned with dataset vocab (lowercase "he"/"she")
    token_id = word_to_id[start_text]
    x = torch.tensor([token_id])

    with torch.no_grad():
        y_pred = model(x)

    next_token_id = int(round(y_pred.item()))
    # Clamp to valid vocab range to avoid out-of-range index
    next_token_id = max(0, min(next_token_id, len(id_to_word) - 1))

    return start_text + " " + id_to_word[next_token_id]


print("Generated text:", generate("he"))
print("Generated text:", generate("she"))