'''
Script to demonstrate a basic model training process using a simple linear regression model on synthetic data.
'''

import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

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
