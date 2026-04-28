from dis import Instruction
import json
import os

import tiktoken
from torch.utils import data
from python_impl.fine_tuning.utils import random_split
from torch.utils.data import Dataset, DataLoader
import torch
from functools import partial

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    customized_collate_fn = partial(
        custom_collate_fn, device=device, allowed_max_length=1024
    )

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")
    tDt = InstructionDataset(tDf, tokenizer)
    tLoader = DataLoader(tDt, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=True, drop_last=True, num_workers=num_workers)
    vDt = InstructionDataset(vDf, tokenizer)
    vLoader = DataLoader(vDf, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=False, drop_last=False, num_workers=num_workers)
    pDt = InstructionDataset(pDf, tokenizer)
    pLoader = DataLoader(pDf, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=False, drop_last=False, num_workers=num_workers)

    
