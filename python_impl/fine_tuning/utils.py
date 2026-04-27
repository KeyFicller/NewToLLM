import torch

from python_impl.toy_model.model import ToyModel
from python_impl.toy_model.config import ToyModelConfig
from pathlib import Path
import tiktoken
from python_impl.load_model.torch_load_model import generate_text_simple

def import_pretrained_model():
    temp_dir = Path(__file__).resolve().parents[1] / ".temp"
    state_dict_path = temp_dir / "toy_model_from_gpt2.pth"
    cfg = ToyModelConfig.copy()
    cfg["qkv_bias"] = True
    model = ToyModel(cfg)

    model.load_state_dict(torch.load(state_dict_path))

    return model

# Split the data to train, validate and test
def random_split(df, tFrac, vFrac, shuffle=True):
    # Shuffle
    if shuffle:
        df = df.sample(frac = 1, random_state=123).reset_index(drop=True)
    tEnd = int(len(df) * tFrac)
    vEnd = tEnd + int(len(df) * vFrac)
    
    return df[:tEnd], df[tEnd:vEnd], df[vEnd:]