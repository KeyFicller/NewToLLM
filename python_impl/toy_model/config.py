# Central config for the toy GPT-like model.
# Keeping these values in one place makes experiments easier.
ToyModelConfig = {
    # Number of unique tokens the tokenizer can output.
    "vocab_size" : 50257, # GPT-2 vocabulary size
    # Size of each token vector (hidden size of the model).
    "emb_dim" : 768, # Embedding dimension
    # Number of Transformer blocks stacked in the model.
    "n_layers" : 12, # Number of layers
    # Number of attention heads per block.
    "n_heads" : 12, # Number of attention heads
    # Dropout probability used to reduce overfitting.
    "drop_rate" : 0.1, # Dropout rate
    # Maximum number of tokens the model can see at once.
    "context_length" : 1024, # Maximum sequence length
    # Whether Q/K/V linear layers use a bias term.
    "qkv_bias" : False, # Whether to use bias in the QKV projection
}