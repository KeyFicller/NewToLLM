import torch

from python_impl.toy_model.model import ToyModel
from python_impl.toy_model.config import ToyModelConfig
from pathlib import Path

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


def evaluate_model(model, tLoader, vLoader, device, eval_iter, calc_loss_loader_fn):
    model.eval()
    with torch.no_grad():
        tLoss = calc_loss_loader_fn(tLoader, model, device, num_batches=eval_iter)
        vLoss = calc_loss_loader_fn(vLoader, model, device, num_batches=eval_iter)
    model.train()
    return tLoss, vLoss


def train_model_simple(
    model,
    tLoader,
    vLoader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    calc_loss_batch_fn,
    calc_loss_loader_fn,
    calc_accuracy_loader_fn=None,
):
    tLosses, vLosses = [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in tLoader:
            optimizer.zero_grad()
            loss = calc_loss_batch_fn(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                tLoss, vLoss = evaluate_model(
                    model,
                    tLoader,
                    vLoader,
                    device,
                    eval_iter,
                    calc_loss_loader_fn,
                )
                tLosses.append(tLoss)
                vLosses.append(vLoss)
                print(
                    f"[fine_tuning] Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {tLoss:.3f}, Validate loss {vLoss:.3f}"
                )

        if calc_accuracy_loader_fn is not None:
            tAccuracy = calc_accuracy_loader_fn(
                tLoader,
                model,
                device,
                num_batches=eval_iter,
            )
            vAccuracy = calc_accuracy_loader_fn(
                vLoader,
                model,
                device,
                num_batches=eval_iter,
            )
            print(
                f"[fine_tuning] Train accuracy: {tAccuracy * 100:.2f}%, "
                f"Validate accuracy: {vAccuracy * 100:.2f}%"
            )

    return tLosses, vLosses