import torch
import torch.nn as nn

class ToyMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, drop_rate, n_heads, qkv_bias=False):
        super().__init__()
        # We can split attention into heads only if d_out is divisible by n_heads.
        assert (d_out % n_heads == 0), "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        # Per-head feature size.
        self.d_head = d_out // n_heads

        # Learned projections to Query/Key/Value spaces.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection after attention heads are concatenated.
        self.out_proj = nn.Linear(d_out, d_out)
        self.drop_out = nn.Dropout(drop_rate)

        # Causal mask: hides future tokens so token t cannot see t+1, t+2, ...
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # x shape: [batch_size, n_tokens, d_in]
        batch_size, n_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape to multi-head format: [batch, heads, tokens, d_head]
        keys = keys.view(batch_size, n_tokens, self.n_heads, self.d_head).transpose(1, 2)
        queries = queries.view(batch_size, n_tokens, self.n_heads, self.d_head).transpose(1, 2)
        values = values.view(batch_size, n_tokens, self.n_heads, self.d_head).transpose(1, 2)

        # Attention score for every token pair.
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]
        attn_scores.masked_fill_(mask_bool, float("-inf"))

        # Scaled softmax to get attention probabilities.
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.drop_out(attn_weights)

        # Weighted sum of values.
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, n_tokens,self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh( 
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3)) 
        ))

class ToyFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Standard Transformer FFN: expand -> nonlinearity -> project back.
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            nn.GELU(),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"]),
        )
    
    def forward(self, x):
        return self.layers(x)

class ToyLayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        # Normalize each token's feature vector along the last dimension.
        mean = x.mean(dim = -1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.bias

class ToyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = ToyMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            drop_rate=cfg["drop_rate"],
            n_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = ToyFeedForward(cfg)
        self.norm1 = ToyLayerNorm(cfg["emb_dim"])
        self.norm2 = ToyLayerNorm(cfg["emb_dim"])
        self.drop_out = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Pre-norm + residual for attention sub-layer.
        short_cut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_out(x)
        x = x + short_cut

        # Pre-norm + residual for feed-forward sub-layer.
        short_cut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_out(x)
        x = x + short_cut
        return x