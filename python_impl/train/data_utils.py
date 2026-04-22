import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        # Store each training sample as:
        # - feature: tokens [t0, t1, ... t(n-1)]
        # - label:   tokens [t1, t2, ... tn]
        # so the model learns "next-token prediction".
        self.features = []
        self.labels = []

        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            feature_chunk = token_ids[i: i + max_length]
            label_chunk = token_ids[i + 1: i + max_length + 1]
            self.features.append(torch.tensor(feature_chunk))
            self.labels.append(torch.tensor(label_chunk))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_data_loader(
    text,
    batch_size = 4,
    max_length = 256,
    stride = 128,
    shuffle = True,
    drop_last = True,
    num_workers = 0
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TextDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

def calc_loss_batch(feature_batch, label_batch, model, device):
    # feature_batch shape: [B, T], label_batch shape: [B, T]
    feature_batch = feature_batch.to(device)
    label_batch = label_batch.to(device)
    logits = model(feature_batch)
    # logits shape is [B, T, V]. Flatten to [B*T, V] to align with labels [B*T].
    loss = F.cross_entropy(logits.flatten(0, 1), label_batch.flatten())

    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0

    # Get batch size from data loader or specified number of batches.
    if (len(data_loader) == 0):
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    # Average loss over a subset (or all) batches.
    for batch_idx, (feature_batch, label_batch) in enumerate(data_loader):
        if batch_idx < num_batches:
            loss = calc_loss_batch(feature_batch, label_batch, model, device)
            total_loss += loss.item()
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss