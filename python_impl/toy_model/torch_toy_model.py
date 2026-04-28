import torch
import python_impl.toy_model.config as cfg
import tiktoken
from python_impl.toy_model.model import ToyModel

# This text generate function simply choose the highest probability token for each step.
def generate_text_simple(model, idx, max_new_tokens, context_length):
    # Repeatedly predict one next token and append it to the sequence.
    for _ in range(max_new_tokens):
        # Keep only the last `context_length` tokens (model's max window).
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)

        # Use only the logits for the last position.
        logits = logits[:, -1, :]

        # Convert logits -> probabilities.
        probs = torch.softmax(logits, dim=-1)

        # Greedy decoding: pick the highest-probability token.
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        # Append new token id to current sequence.
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# This text generate function use temperature sampling and top-k sampling to generate text.
def generate_text_advanced(model, idx, max_new_tokens, context_length,
    temperature=0.0, top_k = None, eos_id = None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(-float('inf')), logits)
        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

def toy_model_torch():
    torch.manual_seed(123)
    # Build model from the config dictionary.
    model = ToyModel(cfg.ToyModelConfig)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")

    # tokenizer = tiktoken.get_encoding("gpt2")
    # batch = []
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"
    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # batch = torch.stack(batch, dim=0)

    # print("Input batch: ", batch)

    # out = model(batch)
    # print("Output shape: ", out.shape)
    # print("Output logits: ", out)

    input_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(input_context)
    print(f"[toy_model] Encoded prompt tokens: {encoded}")
    # Add batch dimension so shape becomes [1, seq_len].
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print(f"[toy_model] Encoded tensor shape: {tuple(encoded_tensor.shape)}")

    # Switch to evaluation mode for generation.
    model.eval()

    # Generate a few new tokens from the prompt.
    output_ids = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_length=cfg.ToyModelConfig["context_length"],
    )
    print(f"[toy_model] Generated token ids: {output_ids.tolist()}")
    print(f"[toy_model] Generated sequence length: {len(output_ids[0])}")

    # Convert token ids back to human-readable text.
    decoded_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    print(f"[toy_model] Decoded text: {decoded_text}")