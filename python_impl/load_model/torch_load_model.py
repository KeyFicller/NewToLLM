import tiktoken
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from python_impl.toy_model.config import ToyModelConfig
from python_impl.toy_model.model import ToyModel
from python_impl.toy_model.torch_toy_model import generate_text_simple

from pathlib import Path


def load_hf_gpt2_to_toy_model(hf_model: AutoModelForCausalLM) -> ToyModel:
    cfg = ToyModelConfig.copy()
    # GPT-2 uses bias in attention QKV projections.
    cfg["qkv_bias"] = True
    toy_model = ToyModel(cfg)

    sd = toy_model.state_dict()
    hf_sd = hf_model.state_dict()

    sd["tok_emb.weight"] = hf_sd["transformer.wte.weight"]
    sd["pos_emb.weight"] = hf_sd["transformer.wpe.weight"]
    sd["final_norm.scale"] = hf_sd["transformer.ln_f.weight"]
    sd["final_norm.bias"] = hf_sd["transformer.ln_f.bias"]
    sd["out_head.weight"] = hf_sd["lm_head.weight"]

    n_layers = cfg["n_layers"]
    for i in range(n_layers):
        toy_prefix = f"trf_blocks.{i}"
        hf_prefix = f"transformer.h.{i}"

        # LayerNorm before attention
        sd[f"{toy_prefix}.norm1.scale"] = hf_sd[f"{hf_prefix}.ln_1.weight"]
        sd[f"{toy_prefix}.norm1.bias"] = hf_sd[f"{hf_prefix}.ln_1.bias"]

        # Attention QKV (HF Conv1D: [in_dim, out_dim], Toy Linear: [out_dim, in_dim])
        c_attn_w = hf_sd[f"{hf_prefix}.attn.c_attn.weight"]
        c_attn_b = hf_sd[f"{hf_prefix}.attn.c_attn.bias"]
        q_w, k_w, v_w = torch.split(c_attn_w, cfg["emb_dim"], dim=1)
        q_b, k_b, v_b = torch.split(c_attn_b, cfg["emb_dim"], dim=0)

        sd[f"{toy_prefix}.att.W_query.weight"] = q_w.T
        sd[f"{toy_prefix}.att.W_query.bias"] = q_b
        sd[f"{toy_prefix}.att.W_key.weight"] = k_w.T
        sd[f"{toy_prefix}.att.W_key.bias"] = k_b
        sd[f"{toy_prefix}.att.W_value.weight"] = v_w.T
        sd[f"{toy_prefix}.att.W_value.bias"] = v_b

        # Attention output projection
        sd[f"{toy_prefix}.att.out_proj.weight"] = hf_sd[f"{hf_prefix}.attn.c_proj.weight"].T
        sd[f"{toy_prefix}.att.out_proj.bias"] = hf_sd[f"{hf_prefix}.attn.c_proj.bias"]

        # LayerNorm before feed-forward
        sd[f"{toy_prefix}.norm2.scale"] = hf_sd[f"{hf_prefix}.ln_2.weight"]
        sd[f"{toy_prefix}.norm2.bias"] = hf_sd[f"{hf_prefix}.ln_2.bias"]

        # Feed-forward projections
        sd[f"{toy_prefix}.ff.layers.0.weight"] = hf_sd[f"{hf_prefix}.mlp.c_fc.weight"].T
        sd[f"{toy_prefix}.ff.layers.0.bias"] = hf_sd[f"{hf_prefix}.mlp.c_fc.bias"]
        sd[f"{toy_prefix}.ff.layers.2.weight"] = hf_sd[f"{hf_prefix}.mlp.c_proj.weight"].T
        sd[f"{toy_prefix}.ff.layers.2.bias"] = hf_sd[f"{hf_prefix}.mlp.c_proj.bias"]

    toy_model.load_state_dict(sd)
    return toy_model


def load_public_model_to_toy_model():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    toy_model = load_hf_gpt2_to_toy_model(hf_model)

    hf_model.eval()
    toy_model.eval()

    prompt = "I love you , but you "
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        hf_logits = hf_model(**inputs).logits
        toy_logits = toy_model(inputs["input_ids"])

    max_abs_diff = (hf_logits - toy_logits).abs().max().item()
    print(f"[load_model] max_abs_diff(logits) = {max_abs_diff:.6e}")

    with torch.no_grad():
        output_ids = hf_model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )
    print(f"[load_model] HF model output: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")


    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # Switch to evaluation mode for generation.
    toy_model.eval()

    # Generate a few new tokens from the prompt.
    output_ids = generate_text_simple(
        model=toy_model,
        idx=encoded_tensor,
        max_new_tokens=30,
        context_length=1024, # TODO: get from config
    )
    print(f"[load_model] Toy model output: {tokenizer.decode(output_ids.squeeze(0).tolist())}")

    temp_dir = Path(__file__).resolve().parents[1] / ".temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = temp_dir / "toy_model_from_gpt2.pth"
    torch.save(toy_model.state_dict(), state_dict_path)

    # Export TorchScript for direct loading in LibTorch.
    scripted_model = torch.jit.script(toy_model)
    scripted_path = temp_dir / "toy_model_from_gpt2.pt"
    scripted_model.save(str(scripted_path))
    print(f"[load_model] Saved state_dict to: {state_dict_path}")
    print(f"[load_model] Saved TorchScript to: {scripted_path}")

def load_model_torch():
    load_public_model_to_toy_model()