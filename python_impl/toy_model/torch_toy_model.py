import logging

import torch
import python_impl.toy_model.config as cfg
import tiktoken
from python_impl.toy_model.model import ToyModel

logger = logging.getLogger(__name__)


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
    logger.info("Encoded prompt tokens: %s", encoded)
    # Add batch dimension so shape becomes [1, seq_len].
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    logger.info("Encoded tensor shape: %s", tuple(encoded_tensor.shape))

    # Switch to evaluation mode for generation.
    model.eval()

    # Generate a few new tokens from the prompt.
    output_ids = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_length=cfg.ToyModelConfig["context_length"],
    )
    logger.info("Generated token ids: %s", output_ids.tolist())
    logger.info("Generated sequence length: %d", len(output_ids[0]))

    # Convert token ids back to human-readable text.
    decoded_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    logger.info("Decoded text: %s", decoded_text)