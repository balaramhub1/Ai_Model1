'''
Class to handle the dataset for training the language model. It reads the text data, tokenizes it, and prepares it for training by creating input-target pairs.
'''
import torch
from torch.utils.data import Dataset

from LLM_from_scratch_class.basic_llm import text_to_token_ids


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
