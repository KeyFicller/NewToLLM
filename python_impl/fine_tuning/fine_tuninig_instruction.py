from dis import Instruction
import json
import os

from torch.utils import data
from python_impl.fine_tuning.utils import random_split
from torch.utils.data import Dataset

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

def fine_tuning_instruction():
    data = load_file("assets/instruction-data.json")
    print("Number of entries: ", len(data))
    print("Instruction sample: ", format_input(data[0]))
    print("Response sample: ", format_output(data[0]))

    tDf, vDf, pDf = random_split(data, 0.85, 0.05, False)

    print("Number of train set: ", len(tDf))
    print("Number of validation set: ", len(vDf))
    print("Number of test set: ", len(pDf))
