"""
Toy context-based embedding learner (NumPy only).

This script demonstrates a minimal, from-scratch approach to learning 2D word
embeddings from a tiny corpus using co-occurrence context similarity.

Pipeline:
1. Build a vocabulary from the corpus.
2. For each word, collect the set of words that appear with it (context profile).
3. Compute pairwise Jaccard similarity between word context profiles.
4. Train 2D embeddings:
   - pull similar word pairs closer
   - push dissimilar word pairs apart
5. Print learned vectors for selected words.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Small handcrafted corpus with simple relationships:
# - he <-> boy, him
# - she <-> girl, her
# - words sharing context should end up geometrically closer
corpus = [
    "he is a boy",     "she is a girl",
    "he is like him",  "she is like her",
    "boy is like him", "girl is like her",
]

# ---------------------------------------------------------------------------
# Build vocabulary
# ---------------------------------------------------------------------------
# Extract unique tokens and sort for deterministic index assignment.
vocab = sorted(set(word for s in corpus for word in s.split()))

# Word-to-index mapping for embedding lookup.
w2i = {w: i for i, w in enumerate(vocab)}

# ---------------------------------------------------------------------------
# Build context profiles
# ---------------------------------------------------------------------------
# For each word, store all other words that appear in the same sentence.
# Using sets avoids duplicate context entries.
context_of = {w: set() for w in vocab}
for sentence in corpus:
    words = sentence.split()
    for w1 in words:
        for w2 in words:
            if w1 != w2:
                context_of[w1].add(w2)

# ---------------------------------------------------------------------------
# Compute context similarity targets
# ---------------------------------------------------------------------------
# For each unique word pair (i, j), compute Jaccard similarity:
# similarity = |intersection(context_i, context_j)| / |union(context_i, context_j)|
pairs = []
for i in range(len(vocab)):
    for j in range(i + 1, len(vocab)):
        word_a = vocab[i]
        word_b = vocab[j]
        shared = context_of[word_a].intersection(context_of[word_b])
        all_context = context_of[word_a].union(context_of[word_b])
        similarity = len(shared) / len(all_context)
        pairs.append((i, j, similarity))


class Embedding:
    """
    Minimal 2D embedding model with heuristic pairwise updates.

    The model stores one 2D vector per vocabulary word and updates vectors using
    pairwise context similarity targets:
    - similar pairs are pulled together,
    - dissimilar pairs are pushed apart.
    """

    def __init__(self, n_words, learning_rate=0.05, n_epochs=1000):
        """
        Initialize model parameters.

        Args:
            n_words (int): Number of vocabulary entries.
            learning_rate (float, optional): Step size for updates. Defaults to 0.05.
            n_epochs (int, optional): Number of full passes over pair data. Defaults to 1000.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        # Randomly initialize 2D embeddings (shape: [n_words, 2]).
        self.embeddings = np.random.randn(n_words, 2) * 0.5

    def train(self, pairs):
        """
        Train embeddings using pairwise similarity-driven push/pull updates.

        Args:
            pairs (list[tuple[int, int, float]]):
                Each tuple is (word_index_i, word_index_j, similarity_score).

        Update rule:
            - If similarity > 0.5: pull vectors together proportionally to similarity.
            - Else: push vectors apart proportionally to (1 - similarity).
        """
        for epoch in range(self.n_epochs):
            for i, j, sim in pairs:
                # Vector from j to i.
                diff = self.embeddings[i] - self.embeddings[j]

                # Euclidean distance with epsilon for numeric stability.
                distance = np.sqrt(np.sum(diff ** 2)) + 1e-8

                # Unit direction vector.
                direction = diff / distance

                if sim > 0.5:  # similar contexts -> pull closer
                    self.embeddings[i] -= self.learning_rate * sim * direction
                    self.embeddings[j] += self.learning_rate * sim * direction
                else:  # different contexts -> push apart
                    self.embeddings[i] += self.learning_rate * (1 - sim) * direction
                    self.embeddings[j] -= self.learning_rate * (1 - sim) * direction

    def get_embedding(self, index):
        """
        Return embedding vector for a given word index.

        Args:
            index (int): Vocabulary index.

        Returns:
            np.ndarray: 2D embedding vector with shape (2,).
        """
        return self.embeddings[index]


def distance(a, b):
    """
    Compute Euclidean distance between two vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        float: Euclidean distance.
    """
    return np.sqrt(np.sum((a - b) ** 2))


# ---------------------------------------------------------------------------
# Create and train model
# ---------------------------------------------------------------------------
# Seed for reproducible random initialization and repeatable output.
np.random.seed(123)

model = Embedding(n_words=len(vocab))
model.train(pairs)

# ---------------------------------------------------------------------------
# Print selected embeddings
# ---------------------------------------------------------------------------
print("\nEmbeddings:")
for word in ["he", "she", "boy", "girl"]:
    v = model.get_embedding(w2i[word])
    print(f"  {word:7s} -> [{v[0]:+.4f}, {v[1]:+.4f}]")