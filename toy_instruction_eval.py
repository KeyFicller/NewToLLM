import html
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import torch

from python_impl.fine_tuning.fine_tuning_instruction import (
    build_gpt2_tokenizer,
    format_input,
)
from python_impl.toy_model.config import ToyModelConfig
from python_impl.toy_model.model import ToyModel
from python_impl.toy_model.torch_toy_model import generate_text_advanced
from python_impl.utils.torch_utils import decl_device


HOST = "127.0.0.1"
PORT = 8787
MODEL_PATH = Path("python_impl/.temp/fine-tuned-model-instruction.pth")

tokenizer = build_gpt2_tokenizer()
device = torch.device(decl_device())
model = None
model_error = None


def load_model():
    global model, model_error
    cfg = ToyModelConfig.copy()
    cfg["qkv_bias"] = True
    m = ToyModel(cfg)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model = m.to(device).eval()



def generate_response(instruction: str, user_input: str) -> str:
    entry = {"instruction": instruction, "input": user_input}
    prompt_text = f"{format_input(entry)}\n\n### Response:\n"
    input_ids = tokenizer.encode(prompt_text)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    output_ids = generate_text_advanced(
        model,
        input_tensor,
        max_new_tokens=120,
        context_length=1024,
        temperature=0.0,
        top_k=None,
        eos_id=50256,
    )
    full_output = tokenizer.decode(output_ids[0].tolist())
    return full_output[len(prompt_text):].strip()


def render_page(instruction: str, user_input: str, output: str) -> bytes:
    model_status = "loaded" if model is not None else f"failed: {model_error}"
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Fine-tuned Instruction Demo</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; max-width: 900px; }}
    textarea {{ width: 100%; box-sizing: border-box; }}
    .muted {{ color: #666; font-size: 13px; margin-bottom: 10px; }}
    .output {{ white-space: pre-wrap; background: #f6f6f6; padding: 12px; border-radius: 8px; min-height: 120px; }}
    button {{ padding: 8px 14px; }}
  </style>
</head>
<body>
  <h2>Fine-tuned Instruction Demo</h2>
  <div class="muted">Device: {html.escape(str(device))} | Model: {html.escape(model_status)}</div>
  <form method="post">
    <label><b>Instruction</b></label><br/>
    <textarea name="instruction" rows="5">{html.escape(instruction)}</textarea><br/><br/>
    <label><b>Input (optional)</b></label><br/>
    <textarea name="user_input" rows="4">{html.escape(user_input)}</textarea><br/><br/>
    <button type="submit">Generate</button>
  </form>
  <h3>Model Output</h3>
  <div class="output">{html.escape(output)}</div>
</body>
</html>"""
    return page.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        instruction = "Evaluate the following phrase by transforming it into the spelling given."
        user_input = "freind --> friend"
        output = ""
        body = render_page(instruction, user_input, output)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        form = parse_qs(raw)
        instruction = form.get("instruction", [""])[0].strip()
        user_input = form.get("user_input", [""])[0].strip()
        if instruction:
            output = generate_response(instruction, user_input)
        else:
            output = "[ERROR] Instruction cannot be empty."
        body = render_page(instruction, user_input, output)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    load_model()
    server = HTTPServer((HOST, PORT), Handler)
    print(f"[fine_tuning] open http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
