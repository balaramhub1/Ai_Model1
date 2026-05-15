'''
Script to demonstrate a basic model training process using a simple linear regression model on synthetic data.
'''

import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from LLM_from_scratch_class_V0.GPTModel import GPTModel


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

data = [
  "She composes songs and practices piano daily.",
  "He reads books and explores the nearby caves.",
  "He reads novel and climbs mountains every weekend.",
  "She composes songs and writes novels.",
  "He reads newspaper and solves complex puzzles.",
  "She composes music and organizes exhibitions regularly.",
  "He reads books and builds small wooden models.",
  "He reads books and participates in local science fairs.",
  "She composes songs and curates art projects.",
  "She composes tunes and designs jewelry for her friends.",
  "He reads everyday and documents wildlife photography trips.",
  "She composes harmonies and experiments with digital music.",
  "He reads novel and trains for local marathons.",
  "She composes soundtracks and collaborates with creative filmmakers.",
  "He reads newspaper and studies navigation using maps and stars.",
  "She composes rhythms and teaches music."
]

VOCAB_SIZE = 70
CONTEXT_LEN = 6
EMB_DIM = 24
BATCH_SIZE = 4
EPOCHS = 100

print("raw data : ", data)

_ = torch.manual_seed(123)

def separate_dots(s):
    """
    Normalize sentence-ending periods by treating '.' as a standalone token.

    This function inserts spaces around literal dot characters so downstream
    tokenization (e.g., `split()`) can separate punctuation from words.

    Args:
        s (str): Input text string, typically a sentence.

    Returns:
        str: Text where each '.' is surrounded by spaces, with leading/trailing
        whitespace removed.

    Example:
        "She writes music." -> "She writes music ."
    """
    # Match literal '.' characters and replace each with " . "
    s = re.sub(r"\.", " . ", s)

    # Remove extra leading/trailing whitespace introduced by replacement
    return s.strip()

text_data = [separate_dots(x) for x in data]

print("test data : ", text_data)

example_data = text_data[0]
print("Sample data: ",example_data)

vocab = set()
for text in text_data:
    for word in text.split():
        vocab.add(word)

vocab = sorted(vocab)
print("Size of vocab : ",len(vocab))

# Display token:word mappings
for i, word in enumerate(vocab):
    print(f"{i}: {word}")

# build word -> index mapping
word_to_id = {word: idx for idx, word in enumerate(vocab)}

# build id -> word mapping
id_to_word = {idx: word for idx, word in enumerate(vocab)}

def text_to_token_ids(text, word_to_id):
    """
    Convert a whitespace-tokenized text string into integer token IDs.

    The function assumes every token in `text` exists in `word_to_id`.
    It is typically used after text normalization/preprocessing and vocab creation.

    Args:
        text (str): Input text (e.g., "She composes songs .").
        word_to_id (dict[str, int]): Mapping from token string to integer ID.

    Returns:
        list[int]: Token IDs in the same order as the input tokens.

    Raises:
        KeyError: If any token in `text` is not present in `word_to_id`.

    Example:
        text_to_token_ids("he reads .", {"he": 0, "reads": 1, ".": 2})
        [0, 1, 2]
    """
    # Split text into whitespace-separated tokens.
    tokens = text.split()

    # Map each token to its vocabulary ID.
    return [word_to_id[t] for t in tokens]


def token_ids_to_text(token_ids, id_to_word):
    """
    Convert a sequence of token IDs back into a space-separated text string.

    This is the inverse operation of `text_to_token_ids` when `id_to_word`
    is the exact reverse mapping of `word_to_id`.

    Args:
        token_ids (list[int]): Sequence of token IDs.
        id_to_word (dict[int, str]): Mapping from integer ID to token string.

    Returns:
        str: Decoded text formed by joining tokens with spaces.

    Raises:
        KeyError: If any ID in `token_ids` is not present in `id_to_word`.

    Example:
        token_ids_to_text([0, 1, 2], {0: "he", 1: "reads", 2: "."})
        "he reads ."
    """
    # Convert each token ID into its corresponding token string.
    words = [id_to_word[id] for id in token_ids]

    # Join tokens into a single text string.
    return " ".join(words)

print("Sample data: ",example_data)

text_to_token_ids(example_data, word_to_id)

print(f"Text to Token id's for :- '{example_data}' is : ", text_to_token_ids(example_data, word_to_id))

text_to_token_ids(example_data, word_to_id)[:CONTEXT_LEN + 1]

print(f"Text to Token id's from index 0 till [context length + 1] i.e {CONTEXT_LEN+1} is : ", text_to_token_ids(example_data, word_to_id)[:CONTEXT_LEN + 1])

# shifted by 1 to get the target token ids
# this is the expected output as the target token ids should be the next token ids for the input token ids
text_to_token_ids(example_data, word_to_id)[1:CONTEXT_LEN + 1]

print(f"Text to Token id's from index 1 till [context length + 1] i.e {CONTEXT_LEN+1} is : ", text_to_token_ids(example_data, word_to_id)[1:CONTEXT_LEN + 1])

# create dataset + dataloader
train_dataset = LLMDataset(
    text_data,
    word_to_id,
    CONTEXT_LEN
)
print("Training Dataset : ",train_dataset)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
print("Training Data Loader : ",train_dataloader)

print(len(train_dataloader))

# Batch of Input and Output data, one batch is of 4 sentence
example_input_output = next(iter(train_dataloader))
print("Example input/output : ",example_input_output)
'''
Example:
Input = [ 2, 11, 55,  3, 46, 45]
Output = [11, 55,  3, 46, 45, 14]
when input is 2, we train model to return 11, when input is 11, we train model to return 55

Input are : 
example_input_output[0][0], example_input_output[0][1], example_input_output[0][2], example_input_output[0][3]

Output are : 
example_input_output[1][0], example_input_output[1][1], example_input_output[1][2], example_input_output[1][3]
'''

# Example Input
print("First Input : ",example_input_output[0][0])

# Example Output (shifted by one token)
print("First Output : ",example_input_output[1][0])



# Instantiate the GPT-style language model.
#
# This single call triggers `GPTModel.__init__()`, which registers the model
# with PyTorch's `nn.Module` system and allocates three learnable components:
#
#   tok_emb      : nn.Embedding(VOCAB_SIZE, EMB_DIM)
#                  Maps each integer token ID to a dense vector of size EMB_DIM.
#
#   pos_emb      : nn.Embedding(CONTEXT_LEN, EMB_DIM)
#                  Maps each sequence position (0 to CONTEXT_LEN-1) to a dense
#                  vector of size EMB_DIM, encoding positional information.
#
#   output_layer : nn.Linear(EMB_DIM, VOCAB_SIZE, bias=False)
#                  Projects the combined token+position embedding back to a
#                  score (logit) for every token in the vocabulary.
#
# All weights are randomly initialized at this point using the state of the
# torch RNG, which was fixed earlier via `torch.manual_seed(123)` to ensure
# reproducible experiments.
#
# After instantiation, `model` is a fully registered `nn.Module`, so:
#   - `model.parameters()` exposes all learnable tensors to the optimizer.
#   - `model(x)` automatically invokes `model.forward(x)` during training
#     and inference.
model = GPTModel()

# Configure the AdamW optimizer to adjust model weights during training.
#
# AdamW (Adam with decoupled Weight decay) is a variant of the Adam optimizer
# that separates L2 regularization (weight decay) from the gradient-based
# parameter update. This often leads to better generalization than standard Adam.
#
# Arguments:
#   model.parameters() : yields all learnable tensors registered in `GPTModel`,
#                        including tok_emb, pos_emb, and output_layer weights.
#   lr=0.0003          : learning rate — controls how large each parameter update
#                        step is. A small value like 0.0003 (3e-4) is a common
#                        and stable starting point for transformer-style models.
#   weight_decay=0.1   : L2 regularization strength — penalizes large weight
#                        values to reduce overfitting. Applied independently of
#                        the gradient update in AdamW.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.1)

# Define the loss function for next-token prediction (multi-class classification).
#
# `CrossEntropyLoss` is the standard loss for language modeling. It internally
# combines two operations:
#   1. `LogSoftmax` : converts raw logits to log-probabilities.
#   2. `NLLLoss`    : computes the negative log-likelihood against the true class.
#
# Expects:
#   - input  : raw logits of shape (batch_size * seq_len, VOCAB_SIZE)
#              produced by `GPTModel.forward()`.
#   - target : integer class IDs of shape (batch_size * seq_len,)
#              representing the true next token at each position.
#
# Note: `CrossEntropyLoss` is preferred over `MSELoss` for token prediction
# because it treats the problem as classification over a vocabulary, producing
# proper probability distributions and sharper gradients.
loss_fn = nn.CrossEntropyLoss()


# Count the total number of scalar values across all trainable parameters in the model.
#
# Breakdown:
#   model.parameters()   : yields every parameter tensor registered in `GPTModel`
#                          (tok_emb, pos_emb, output_layer weights).
#   p.requires_grad      : filters to only parameters that will receive gradient
#                          updates during backpropagation. Parameters with
#                          `requires_grad=False` are frozen and excluded.
#   p.numel()            : returns the total number of scalar elements in tensor `p`.
#                          e.g. a tensor of shape (70, 24) has 70 * 24 = 1,680 elements.
#   sum(...)             : accumulates the element counts across all trainable tensors.
#
# Result: a single integer representing the model's total trainable capacity.
# Useful for comparing model sizes, debugging architecture changes, and
# estimating memory requirements before training.
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {trainable_params:,}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Initialise a dictionary that accumulates parameter counts grouped by
# top-level module name. `defaultdict(int)` automatically initialises any
# missing key to 0, so no explicit key-existence check is needed when
# incrementing counts.
param_count = defaultdict(int)

# Iterate over all named parameters in the model.
# `model.named_parameters()` yields (name, tensor) pairs where `name` is a
# dot-separated string reflecting the module hierarchy, e.g.:
#   "tok_emb.weight"      -> top-level module: "tok_emb"
#   "pos_emb.weight"      -> top-level module: "pos_emb"
#   "output_layer.weight" -> top-level module: "output_layer"
for name, param in model.named_parameters():

    # Extract only the first segment of the parameter path (before the first dot)
    # to identify which top-level submodule the parameter belongs to.
    # e.g. "tok_emb.weight".split('.')[0] -> "tok_emb"
    top_level = name.split('.')[0]

    # Accumulate the total scalar element count for this top-level module.
    # `param.numel()` returns the number of individual values in the tensor.
    # e.g. tok_emb weight of shape (70, 24) contributes 1,680 elements.
    param_count[top_level] += param.numel()

# Print a formatted summary table of parameter counts per top-level module.
# This gives a clear breakdown of where model capacity is concentrated,
# useful for architecture debugging and comparing model variants.
#
# Format:
#   {k:20s}  left-aligns the module name in a 20-character wide column.
#   {v:,}    formats the count with thousands separators for readability.
for k, v in param_count.items():
    print(f"{k:20s}: {v:,}")

def train(dataloader, model, loss_fn, optimizer):
    """
    Execute one full training pass (one epoch) over the provided dataloader.

    For each mini-batch, the function:
    1. Runs a forward pass to produce logits.
    2. Reshapes logits and targets to the shape expected by `CrossEntropyLoss`.
    3. Computes the loss.
    4. Backpropagates gradients.
    5. Updates model parameters.
    6. Clears gradients for the next iteration.

    Args:
        dataloader  (DataLoader) : Yields (X, y) batches from `LLMDataset`.
                                   X shape: (batch_size, seq_len)
                                   y shape: (batch_size, seq_len)
        model       (nn.Module)  : The GPT-style model to train.
        loss_fn     (nn.Module)  : Loss function; expected to be `CrossEntropyLoss`.
        optimizer   (Optimizer)  : Parameter update rule; expected to be `AdamW`.

    Returns:
        None. Prints per-batch loss to stdout.
    """
    # Switch model to training mode.
    # Enables train-time behavior for layers such as dropout and batch norm.
    model.train()

    # Iterate over mini-batches; `batch` is the 0-based batch index.
    # X: input token IDs, y: target (next) token IDs — both shape (batch_size, seq_len).
    for batch, (X, y) in enumerate(dataloader):

        # Forward pass: produce raw logits for each token position.
        # logits shape: (batch_size, seq_len, VOCAB_SIZE)
        logits = model(X)

        # Reshape tensors to satisfy CrossEntropyLoss input contract:
        #   - logits.flatten(0, 1) : merges batch and seq dims
        #                            -> (batch_size * seq_len, VOCAB_SIZE)
        #   - y.flatten()          : flattens target IDs to 1-D
        #                            -> (batch_size * seq_len,)
        # CrossEntropyLoss expects class scores per row and integer class IDs as targets.
        loss = loss_fn(logits.flatten(0, 1), y.flatten())

        # Backpropagation: compute gradients of loss w.r.t. all model parameters.
        loss.backward()

        # Apply computed gradients to update model parameters using AdamW rules.
        optimizer.step()

        # Zero out gradients to prevent accumulation into the next batch iteration.
        optimizer.zero_grad()

        # Log batch index (1-based) and current scalar loss value.
        # `loss:>7f` right-aligns the float in a 7-character wide field.
        print(f"batch: {batch + 1} loss: {loss:>7f}")

for epoch in range(EPOCHS):
    print("-------------------------------")
    print(f"Epoch: {epoch+1}")
    train(train_dataloader, model, loss_fn, optimizer)
    print("-------------------------------")
print("Done!")


# Switch the model to evaluation/inference mode.
#
# `model.eval()` tells PyTorch modules to use eval-time behavior:
#   - Dropout layers (if present) stop randomly dropping activations.
#   - BatchNorm layers (if present) use stored running statistics instead of
#     per-batch statistics.
#
# This is the correct mode before generating text or running validation.
#
# The return value is the model itself; assigning to `_` is a convention that
# means "intentionally ignore this return value" while keeping the side effect
# (changing the model mode).
_ = model.eval()

def generate(start_text):
    """
    Autoregressively generate a fixed-length continuation from a start prompt.

    The function:
    1. Encodes `start_text` into token IDs.
    2. Repeatedly predicts the next token using the current context window.
    3. Appends each predicted token to the running sequence.
    4. Decodes final token IDs back to text.

    Generation uses greedy decoding (`argmax`), meaning it always picks the
    highest-logit token at each step.

    Args:
        start_text (str): Prompt text to seed generation. Tokens must exist in
            `word_to_id` and should be whitespace-tokenized in the same style
            used during training.

    Returns:
        str: Generated text with total length up to `CONTEXT_LEN` tokens
        (prompt + predicted tokens).
    """
    # Convert prompt text into integer token IDs.
    token_ids = text_to_token_ids(start_text, word_to_id)

    # Compute how many tokens to generate to reach CONTEXT_LEN total tokens.
    num_new_tokens = CONTEXT_LEN - len(token_ids)

    # Create a batch dimension so input shape becomes (1, current_seq_len).
    idx = torch.tensor(token_ids).unsqueeze(0)

    # Generate one token at a time.
    for _ in range(num_new_tokens):
        # Keep only the most recent CONTEXT_LEN tokens as model input context.
        idx_cond = idx[:, -CONTEXT_LEN:]

        # Inference mode: disable gradient tracking for speed/memory efficiency.
        with torch.no_grad():
            # logits shape: (1, seq_len, VOCAB_SIZE)
            logits = model(idx_cond)

        # Select logits for the last time step only (next-token distribution).
        # shape becomes: (1, VOCAB_SIZE)
        logits = logits[:, -1, :]

        # Greedy decoding: pick highest-probability token ID.
        # keepdim=True keeps shape as (1, 1) for easy concatenation.
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append predicted token ID to the running sequence along time dimension.
        idx = torch.cat((idx, idx_next), dim=1)

    # Remove batch dimension and convert IDs back to a Python list.
    idx = idx.view(-1).tolist()

    # Decode token IDs into readable text.
    text = token_ids_to_text(idx, id_to_word)
    return text

text = generate("He reads")
print("Generated text:", text)

text = generate("She composes")
print("Generated text:", text)
