# Embedding Model

A tiny word embedding model built from scratch with NumPy. Trains on a small corpus and learns to place words in 2D space - words that appear in similar contexts end up close together.

## How It Works

1. Build a 6-sentence corpus using he/she/boy/girl/him/her
2. For each word, collect all other words that appear in the same sentence (context profile)
3. Compute Jaccard similarity between every pair of context profiles (shared / union)
4. Train 2D embeddings: pull similar-context words closer, push different-context words apart

The `Embedding` class exposes `train(pairs)` for training and `get_embedding(index)` for retrieving vectors. A standalone `distance(a, b)` function computes Euclidean distance between vectors.

## Run

```bash
python3 model.py
```

## Output

```
Embeddings:
  he      -> [-101.7751, -163.4795]
  she     -> [+108.9409, +171.5504]
  boy     -> [-101.7599, -163.4553]
  girl    -> [+108.8153, +171.4747]
```

Similar words have small Euclidean distance (he-boy = 0.03, she-girl = 0.15), while different pairs are far apart (he-she = 395.79).
