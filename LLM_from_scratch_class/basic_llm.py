'''
SCript to demonstrate basic LLM working
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import UserDict, defaultdict

from LLM_from_scratch_class.GPTModel import GPTModel
#from LLM_from_scratch_class.LLMDataset import LLMDataset


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

# list of strings
data = ["he boy",
        "she girl"]

VOCAB_SIZE = 4 # no of unique words in the data
CONTEXT_LEN = 1 # how many previous words to consider for predicting the next word
BATCH_SIZE = 2
EPOCHS = 100

print("type of data : ", type(data))

_ = torch.manual_seed(123)

text_data = data
print("test data : ", text_data)

example_data = text_data[0]
print(example_data)

vocab =set()
for sentence in data:
    for word in sentence.split():
        vocab.add(word)
print("vocab : ", vocab)
vocab = sorted(vocab)
print("sorted vocab : ", vocab)

# code to read the set vocab with individual elements index
# Creating word to index mapping
word_to_id = {word: idx for idx, word in enumerate(vocab)}
print("vocab to index : ", word_to_id)
print("vocab to index type : ", type(word_to_id))

# Creating index to word mapping
id_to_word = {idx: word for idx, word in enumerate(vocab)}
print("index to vocab : ", id_to_word)
print("index to vocab type : ", type(id_to_word))


# Converting text to tokens
def text_to_token_ids(text, word_to_id):
    tokens = text.split()
    return [word_to_id[token] for token in tokens]

# Converting tokens to text
def token_ids_to_text(token_ids, id_to_word):
    return ' '.join([id_to_word[token_id] for token_id in token_ids])

print(example_data)
text_to_token_ids(example_data, word_to_id)

text_to_token_ids(example_data, word_to_id)[:CONTEXT_LEN]
text_to_token_ids(example_data, word_to_id)[1:CONTEXT_LEN + 1]

# create dataset + dataloader
train_dataset = LLMDataset(
    text_data,
    word_to_id,
    CONTEXT_LEN
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(len(train_dataloader))

example_input_output = next(iter(train_dataloader))

print(example_input_output)

import re
formatted = [re.sub(r'\s+', ' ', str(t)) for t in example_input_output]
print("[" + ",\n ".join(formatted) + "]")

# Example Input
example_input_output[0][0]

# Example Output (shifted by one token)
example_input_output[1][0]

model = GPTModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {trainable_params:,}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

param_count = defaultdict(int)

for name, param in model.named_parameters():
    top_level = name.split('.')[0]
    param_count[top_level] += param.numel()

for k, v in param_count.items():
    print(f"{k:20s}: {v:,}")


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        y_predicted = model(X)
        loss = loss_fn(y_predicted, y.float())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"batch: {batch + 1} loss: {loss:>7f}")


for epoch in range(EPOCHS):
    print("-------------------------------")
    print(f"Epoch: {epoch+1}")
    train(train_dataloader, model, loss_fn, optimizer)
    print("-------------------------------")
print("Done!")

_ = model.eval()

def generate(start_text):
    token_id = word_to_id[start_text]
    x = torch.tensor([token_id])
    with torch.no_grad():
        y_predicted = model(x)
    next_token_id = round(y_predicted.item())
    return start_text + " " + id_to_word[next_token_id]

text = generate("he")
print("Generated text:", text)

text = generate("she")
print("Generated text:", text)
