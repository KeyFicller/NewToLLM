import torch
import torch.nn as nn
from python_impl.toy_model.torch_toy_model import ToyModel
from python_impl.toy_model.config import ToyModelConfig

import tiktoken
from python_impl.toy_model.torch_toy_model import generate_text_simple, generate_text_advanced

import python_impl.train.data_utils as data_utils

import logging

logger = logging.getLogger(__name__)

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_and_print_sample(model, tokenizer, device, start_context):
    # Generate text periodically to see qualitative training progress.
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, max_new_tokens=50, context_length=context_size)
    decoded = token_ids_to_text(token_ids, tokenizer)
    logger.info("[train] Generated text simple: %s", decoded.replace("\n", " "))

    with torch.no_grad():
        token_ids = generate_text_advanced(model, encoded, max_new_tokens=50, context_length=context_size, temperature=1.4, top_k=25, eos_id=None)
    decoded = token_ids_to_text(token_ids, tokenizer)
    logger.info("[train] Generated text advanced: %s", decoded.replace("\n", " "))
    
    model.train()

def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer
):
    # Track metrics over training for later plotting or analysis.
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for feature_batch, label_batch in train_loader:
            optimizer.zero_grad()
            loss = data_utils.calc_loss_batch(feature_batch, label_batch, model, device)
            loss.backward()

            optimizer.step()

            # Count how many tokens have been processed so far.
            tokens_seen += feature_batch.numel()
            global_step += 1

            # Evaluate every N optimization steps.
            if global_step % eval_freq == 0:
                train_loss, val_loss = data_utils.evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                logger.info(
                    "[train] Epoch %d, Step %d, Train Loss: %.4f, Val Loss: %.4f",
                    epoch + 1,
                    global_step,
                    train_loss,
                    val_loss,
                )

        # Qualitative sample generation during training.
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

def train_torch():
    cfg = ToyModelConfig.copy()
    # Make it runnable.
    cfg["context_length"] = 256

    torch.manual_seed(123)
    model = ToyModel(cfg)
    model.eval()

    # Prompt used for before/after training generation comparison.
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(model, text_to_token_ids(start_context, tokenizer), max_new_tokens=10, context_length=cfg["context_length"])
    generated_text = token_ids_to_text(token_ids, tokenizer)
    logger.info("[train] Generated text without training: %s", generated_text.replace("\n", " "))

    file_path = "assets/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = data_utils.create_data_loader(train_data, batch_size=2, max_length=cfg["context_length"], stride=cfg["context_length"], shuffle=True, drop_last=True, num_workers=0)
    val_loader = data_utils.create_data_loader(val_data, batch_size=2, max_length=cfg["context_length"], stride=cfg["context_length"], shuffle=False, drop_last=True, num_workers=0)

    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    logger.info(
        "[train] Train loader samples: %d, batches: %d, batch_size: %d",
        train_samples,
        train_batches,
        train_loader.batch_size,
    )
    logger.info(
        "[train] Validation loader samples: %d, batches: %d, batch_size: %d",
        val_samples,
        val_batches,
        val_loader.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Start training loop and collect loss curves.
    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=start_context,
        tokenizer=tokenizer
    )
    