
# Detailed Explanation of `LLM_from_scratch_class_Embedding_model/model.py`

`LLM_from_scratch_class_Embedding_model/model.py` is a **toy embedding learner** built from scratch using NumPy.  
Instead of using Word2Vec/GloVe libraries, it manually builds a context-based similarity signal and then adjusts 2D word vectors so similar words move closer and dissimilar words move farther apart.

---

## 1) Imports and Dataset

import numpy as np

Imports NumPy for vector operations and numerical math.

```python
corpus = [
    "he is a boy",     "she is a girl",
    "he is like him",  "she is like her",
    "boy is like him", "girl is like her",
]
```

Defines a tiny handcrafted corpus with simple semantic patterns:
- `he` co-occurs with `boy`, `him`
- `she` co-occurs with `girl`, `her`
- `boy` and `girl` appear in similar contexts

---

## 2) Vocabulary and Index Mapping

vocab = sorted(set(word for s in corpus for word in s.split()))
w2i = {w: i for i, w in enumerate(vocab)}


- Extracts all unique words from the corpus.
- Sorts them to make ordering deterministic.
- Creates `w2i` (`word -> index`) mapping so words can be represented as numeric IDs.

Deterministic sorting helps reproducibility and easier debugging.

---

## 3) Build Context Profiles


context_of = {w: set() for w in vocab}

Initializes an empty context set for each word.

for sentence in corpus:
    words = sentence.split()
    for w1 in words:
        for w2 in words:
            if w1 != w2:
                context_of[w1].add(w2)


For each sentence:
- Every word `w1` gets every other word `w2` in the same sentence as context.
- Self-context is excluded (`w1 != w2`).
- `set` removes duplicates automatically.

Result: `context_of[word]` stores words that co-occur with `word`.

---

## 4) Pairwise Context Similarity Targets

pairs = []
for i in range(len(vocab)):
    for j in range(i + 1, len(vocab)):
        word_a = vocab[i]
        word_b = vocab[j]
        shared = context_of[word_a].intersection(context_of[word_b])
        all_context = context_of[word_a].union(context_of[word_b])
        similarity = len(shared) / len(all_context)
        pairs.append((i, j, similarity))

Builds one training record per unique word pair `(i, j)`:

- `shared`: intersection of context sets
- `all_context`: union of context sets
- `similarity = |intersection| / |union|` (Jaccard similarity)

So each pair gets a score in `[0, 1]`:
- `1` means highly similar contexts
- `0` means very different contexts

`pairs` is the supervision signal for training.

---

## 5) `Embedding` Class

### Constructor (`__init__`)

self.learning_rate = learning_rate
self.n_epochs = n_epochs
self.embeddings = np.random.randn(n_words, 2) * 0.5

- Stores hyperparameters.
- Initializes embedding matrix of shape `(n_words, 2)`.
- Each word gets a 2D vector.
- Random init is scaled by `0.5` for moderate starting values.

---

### Training Method (`train`)

for epoch in range(self.n_epochs):
    for i, j, sim in pairs:
        diff = self.embeddings[i] - self.embeddings[j]
        distance = np.sqrt(np.sum(diff ** 2)) + 1e-8
        direction = diff / distance

For each pair:
- `diff`: vector between embeddings
- `distance`: Euclidean distance
- `+ 1e-8`: numerical stability (avoid division by zero)
- `direction`: unit vector along pair direction

#### If contexts are similar (`sim > 0.5`), pull together

self.embeddings[i] -= self.learning_rate * sim * direction
self.embeddings[j] += self.learning_rate * sim * direction

#### Else, push apart

self.embeddings[i] += self.learning_rate * (1 - sim) * direction
self.embeddings[j] -= self.learning_rate * (1 - sim) * direction

This creates a simple force-like update rule:
- attract similar pairs
- repel dissimilar pairs

---

### `get_embedding`

def get_embedding(self, index):
    return self.embeddings[index]

Returns the vector for a given word index.

---

## 6) Utility Distance Function

def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

Computes Euclidean distance between two vectors.  
(Defined as utility; not used later in this script.)

---

## 7) Train and Print Results

np.random.seed(123)
model = Embedding(n_words=len(vocab))
model.train(pairs)

- Sets random seed for reproducibility.
- Creates model with one vector per vocabulary word.
- Runs training for default `n_epochs=1000`.

print("\nEmbeddings:")
for word in ["he", "she", "boy", "girl"]:
    v = model.get_embedding(w2i[word])
    print(f"  {word:7s} -> [{v[0]:+.4f}, {v[1]:+.4f}]")

Prints final 2D embeddings for selected words in formatted output.

---

## Conceptual Summary

This file demonstrates the core distributional NLP idea:

> Words that appear in similar contexts should have similar embeddings.

Pipeline:
1. Build co-occurrence contexts
2. Compute pairwise context similarity (Jaccard)
3. Train vectors using pull/push updates
4. Inspect resulting coordinates

---

## Limitations (expected for a toy model)

- Very small corpus
- Sentence-level context only (no sliding window)
- Hard threshold (`sim > 0.5`) is heuristic
- No formal global loss tracking
- 2D vectors are for intuition, not production NLP quality
```