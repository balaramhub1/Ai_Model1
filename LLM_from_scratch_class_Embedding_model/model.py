import numpy as np

# Data
corpus = [
    "he is a boy",     "she is a girl",
    "he is like him",  "she is like her",
    "boy is like him", "girl is like her",
]

# Build vocabulary
vocab = sorted(set(word for s in corpus for word in s.split()))
w2i = {w: i for i, w in enumerate(vocab)}

# Build context profile: for each word, which words appear with it?
context_of = {w: set() for w in vocab}
for sentence in corpus:
    words = sentence.split()
    for w1 in words:
        for w2 in words:
            if w1 != w2:
                context_of[w1].add(w2)

# Context similarity between every pair
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
    def __init__(self, n_words, learning_rate=0.05, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.embeddings = np.random.randn(n_words, 2) * 0.5

    def train(self, pairs):
        for epoch in range(self.n_epochs):
            for i, j, sim in pairs:
                diff = self.embeddings[i] - self.embeddings[j]
                distance = np.sqrt(np.sum(diff ** 2)) + 1e-8
                direction = diff / distance
                if sim > 0.5:  # similar contexts -> pull closer
                    self.embeddings[i] -= self.learning_rate * sim * direction
                    self.embeddings[j] += self.learning_rate * sim * direction
                else:  # different contexts -> push apart
                    self.embeddings[i] += self.learning_rate * (1 - sim) * direction
                    self.embeddings[j] -= self.learning_rate * (1 - sim) * direction

    def get_embedding(self, index):
        return self.embeddings[index]

def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Create and train model
np.random.seed(123)
model = Embedding(n_words=len(vocab))
model.train(pairs)

# Print embeddings
print("\nEmbeddings:")
for word in ["he", "she", "boy", "girl"]:
    v = model.get_embedding(w2i[word])
    print(f"  {word:7s} -> [{v[0]:+.4f}, {v[1]:+.4f}]")
