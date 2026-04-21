import torch
import torch.nn as nn
from python_impl.toy_model.transformer import ToyTransformerBlock, ToyLayerNorm

class ToyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Token embedding: maps token ids -> dense vectors.
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # Positional embedding: tells the model where each token is in sequence.
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # Dropout right after adding token + position embeddings.
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of Transformer blocks (attention + feed-forward).
        self.trf_blocks = nn.Sequential(
            *[ToyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final normalization before projecting to vocabulary logits.
        self.final_norm = ToyLayerNorm(cfg["emb_dim"])

        # Output projection (language modeling head).
        # Converts hidden states to one logit per vocabulary token.
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx shape: [batch_size, num_tokens]
        batch_size, n_tokens = in_idx.shape

        # Convert token ids to vectors.
        tok_embeds = self.tok_emb(in_idx)

        # Build position ids [0, 1, 2, ...] for current sequence length.
        pos_embeds = self.pos_emb(torch.arange(n_tokens, device=in_idx.device))

        # Combine token meaning + token position.
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # logits shape: [batch_size, num_tokens, vocab_size]
        logits = self.out_head(x)
        return logits
