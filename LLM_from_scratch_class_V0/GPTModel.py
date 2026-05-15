'''
Class for GPT model, including the architecture and forward pass.
'''
import torch
import torch.nn as nn

VOCAB_SIZE = 70
CONTEXT_LEN = 6
EMB_DIM = 24
BATCH_SIZE = 4
EPOCHS = 100

class GPTModel(nn.Module):

# Initialization of the GPTModel class, setting up token and positional embeddings, and the output projection layer.
    def __init__(self):
        """
        Initialize the core layers of the GPT-style model.

        This constructor sets up:
        - a token embedding layer to convert token IDs into dense vectors,
        - a positional embedding layer to encode token positions within the context,
        - a linear output projection layer that maps hidden embeddings back to
          vocabulary-sized logits for next-token prediction.

        The layer dimensions are controlled by the module-level constants:
        - `VOCAB_SIZE`: number of unique tokens in the vocabulary,
        - `CONTEXT_LEN`: maximum sequence length handled by positional embeddings,
        - `EMB_DIM`: size of each embedding vector.
        """
        # Initialize the parent `nn.Module` so PyTorch can properly register
        # submodules, parameters, and internal state.
        super().__init__()

        # Learnable lookup table that maps each token ID in the vocabulary
        # to a dense embedding vector of size `EMB_DIM`.
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)

        # Learnable positional embedding table that provides a vector for each
        # position from 0 to `CONTEXT_LEN - 1`, allowing the model to encode
        # token order in the sequence.
        self.pos_emb = nn.Embedding(CONTEXT_LEN, EMB_DIM)

        # Final linear projection layer that converts each embedding vector
        # back into a score (logit) for every token in the vocabulary.
        # `bias=False` keeps the projection as a pure matrix multiplication.
        self.output_layer = nn.Linear(EMB_DIM, VOCAB_SIZE, bias=False)

# Defines the forward pass for `GPTModel`.
# In PyTorch, this method describes how input tensors flow through the model
# and is invoked automatically when the model instance is called, e.g. `model(x)`.
    def forward(self, in_idx):
        """
        Run the forward pass of the GPT-style model.

        This method:
        1. reads the input token IDs,
        2. looks up token embeddings,
        3. looks up positional embeddings for each sequence position,
        4. combines token and positional information,
        5. projects the combined embeddings to vocabulary-sized logits.

        Args:
            in_idx (torch.Tensor): Input token IDs of shape
                `(batch_size, seq_len)`.

        Returns:
            torch.Tensor: Output logits of shape
                `(batch_size, seq_len, VOCAB_SIZE)`, where each position contains
                unnormalized scores for every token in the vocabulary.
        """
        # Extract the sequence length from the input tensor shape.
        # `in_idx` is expected to be shaped as: (batch_size, seq_len)
        _, seq_len = in_idx.shape

        # Convert token IDs into dense token embeddings.
        # Result shape: (batch_size, seq_len, EMB_DIM)
        tok_embeds = self.tok_emb(in_idx)

        # Create position indices [0, 1, 2, ..., seq_len - 1] and look up their
        # corresponding positional embeddings.
        # Result shape: (seq_len, EMB_DIM)
        pos_embeds = self.pos_emb(torch.arange(seq_len))

        # Combine token embeddings with positional embeddings.
        # Broadcasting expands `pos_embeds` across the batch dimension.
        # Result shape: (batch_size, seq_len, EMB_DIM)
        x = tok_embeds + pos_embeds

        # Project each embedding vector to vocabulary-sized logits.
        # Result shape: (batch_size, seq_len, VOCAB_SIZE)
        logits = self.output_layer(x)

        # Return raw logits; loss functions such as CrossEntropyLoss typically
        # expect unnormalized scores rather than probabilities.
        return logits