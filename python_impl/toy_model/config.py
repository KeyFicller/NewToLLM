# Central config for the toy GPT-like model (GPT-2 124M shape).
# Values live in configs/toy_gpt2.json, shared with the C++ side.
#   vocab_size     number of unique tokens (GPT-2 vocabulary)
#   emb_dim        hidden size of the model
#   n_layers       number of Transformer blocks
#   n_heads        attention heads per block
#   drop_rate      dropout probability
#   context_length maximum sequence length
#   qkv_bias       whether Q/K/V projections use a bias
from python_impl.utils.runtime import SHARED_CONFIG

# Full-size config (ignores --small); use runtime.model_config() to honour it.
ToyModelConfig = dict(SHARED_CONFIG["model"])
