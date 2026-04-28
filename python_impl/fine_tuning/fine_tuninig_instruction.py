from dis import Instruction
import json
import os

import tiktoken
from torch.utils import data
from python_impl.fine_tuning.utils import random_split, train_model_simple
from torch.utils.data import Dataset, DataLoader
import torch
from functools import partial
from python_impl.fine_tuning.utils import import_pretrained_model
from python_impl.toy_model.torch_toy_model import generate_text_advanced
from transformers import AutoTokenizer
from python_impl.train.data_utils import calc_loss_loader, calc_loss_batch
from pathlib import Path


class HFTokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def build_gpt2_tokenizer():
    try:
        return tiktoken.get_encoding("gpt2")
    except Exception as e:
        print(f"[fine_tuning] tiktoken gpt2 unavailable, fallback to local HF tokenizer: {e}")
        hf_tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        return HFTokenizerAdapter(hf_tokenizer)


def decide_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_file(path):
    with open(path, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    return data

def format_input(entry):
    instruction_text = ( 
        f"Below is an instruction that describes a task. " 
        f"Write a response that appropriately completes the request." 
        f"\n\n### Instruction:\n{entry['instruction']}" 
    ) 
    input_text = ( 
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else "" 
    ) 
    return instruction_text + input_text

def format_output(entry):
    response_test = f"\n\n### Response:\n{entry['output']}"
    return response_test


def build_prompt_for_generation(entry):
    return f"{format_input(entry)}\n\n### Response:\n"

class InstructionDataset(Dataset):
    def __init__(self, raw_data, tokenizer):
        self.data = raw_data
        self.encoded_text = []
        for entry in self.data:
            instruction_plus_input = format_input(entry)
            response = format_output(entry)
            full_text = instruction_plus_input + response
            self.encoded_text.append(tokenizer.encode(full_text))
    
    def __getitem__(self, index):
        return self.encoded_text[index]
    
    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None,device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def fine_tuning_instruction():
    data = load_file("assets/instruction-data.json")
    print("[fine_tuning] Number of entries: ", len(data))
    print("[fine_tuning] Instruction sample: ", format_input(data[0]))
    print("[fine_tuning] Response sample: ", format_output(data[0]))

    tDf, vDf, pDf = random_split(data, 0.85, 0.05, False)

    print("[fine_tuning] Number of train set: ", len(tDf))
    print("[fine_tuning] Number of validation set: ", len(vDf))
    print("[fine_tuning] Number of test set: ", len(pDf))

    # Test collate fn
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (inputs_1, inputs_2, inputs_3)
    print("[fine_tuning] Collate fn sample:", custom_collate_fn(batch))

    device = decide_device()

    customized_collate_fn = partial(
        custom_collate_fn, device=device, allowed_max_length=1024
    )

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    tokenizer = build_gpt2_tokenizer()
    tDt = InstructionDataset(tDf, tokenizer)
    tLoader = DataLoader(tDt, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=True, drop_last=True, num_workers=num_workers)
    vDt = InstructionDataset(vDf, tokenizer)
    vLoader = DataLoader(vDt, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=False, drop_last=False, num_workers=num_workers)
    pDt = InstructionDataset(pDf, tokenizer)
    pLoader = DataLoader(pDt, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=False, drop_last=False, num_workers=num_workers)

    # Output example of unfine_tuned model
    model = import_pretrained_model()
    model.eval()
    model.to(device)

    test_entry = pDf[0] if len(pDf) > 0 else data[0]
    prompt_text = build_prompt_for_generation(test_entry)
    expected_response = test_entry["output"]
    prompt_ids = tokenizer.encode(prompt_text)
    print("[fine_tuning] Test prompt: ", prompt_text)
    input_ids = torch.tensor(prompt_ids).unsqueeze(0).to(device)
    output_ids = generate_text_advanced(
        model,
        input_ids,
        max_new_tokens=50,
        context_length=1024,
        temperature=0.0,
        top_k=None,
        eos_id=50256,
    )
    full_output_text = tokenizer.decode(output_ids[0].tolist())
    generated_response = full_output_text[len(prompt_text):]
    print("[fine_tuning] Unfine-tuned model response: ", generated_response)
    print("[fine_tuning] Expected response: ", expected_response)

    model.eval()
    with torch.no_grad():
        val_loss = calc_loss_loader(vLoader, model, device, num_batches=5)
        test_loss = calc_loss_loader(pLoader, model, device, num_batches=5)
    print(f"[fine_tuning] Validation loss before fine-tuning: {val_loss:.4f}")
    print(f"[fine_tuning] Test loss before fine-tuning: {test_loss:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 1
    train_model_simple(
        model,
        tLoader,
        vLoader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        calc_loss_batch_fn=calc_loss_batch,
        calc_loss_loader_fn=calc_loss_loader,
    )

    model.eval()
    with torch.no_grad():
        val_loss = calc_loss_loader(vLoader, model, device, num_batches=5)
        test_loss = calc_loss_loader(pLoader, model, device, num_batches=5)
    print(f"[fine_tuning] Validation loss after fine-tuning: {val_loss:.4f}")
    print(f"[fine_tuning] Test loss after fine-tuning: {test_loss:.4f}")

    input_ids = torch.tensor(prompt_ids).unsqueeze(0).to(device)
    output_ids = generate_text_advanced(
        model,
        input_ids,
        max_new_tokens=50,
        context_length=1024,
        temperature=0.0,
        top_k=None,
        eos_id=50256,
    )
    full_output_text = tokenizer.decode(output_ids[0].tolist())
    generated_response = full_output_text[len(prompt_text):]
    print("[fine_tuning] Fine-tuned model response: ", generated_response)
    print("[fine_tuning] Expected response: ", expected_response)

    # Save the model
    temp_dir = Path(__file__).resolve().parents[1] / ".temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    model_path = temp_dir / "fine-tuned-model-instruction.pth"
    torch.save(model.state_dict(), model_path)