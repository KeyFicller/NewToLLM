import torch
import tiktoken
from torch.optim import optimizer
from python_impl.fine_tuning.utils import import_pretrained_model, random_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class ClassifyDataset(Dataset):
    def __init__(self, df, tokenizer, max_length = None, pad_token_id = 50256):
        self.encoded_texts = [tokenizer.encode(text) for text in df["Text"]]
        self.labels = df["Label"].tolist()

        if max_length == None:
            self.max_length = 0 
            for encoded_text in self.encoded_texts: 
                encoded_length = len(encoded_text) 
                if encoded_length > self.max_length: 
                    self.max_length = encoded_length 
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]
        
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts
        ]
    
    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label = self.labels[idx]
        return (torch.tensor(encoded, dtype= torch.long), torch.tensor(label, dtype=torch.long))
    
    def __len__(self):
        return len(self.encoded_texts)

# Sample on ham to make num of spam samples and num of ham samples equal
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state = 123
    )
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

# Load dataset from tsc file
def prepare_dataset():
    df = pd.read_csv("assets/SMSSpamCollection.tsv", sep="\t", header=None, names=["Label", "Text"])
    print("[fine_tuning] Raw data:\n", df["Label"].value_counts())

    balanced_df = create_balanced_dataset(df)
    print("[fine_tuning] Balanced data:\n", balanced_df["Label"].value_counts())

    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    return random_split(balanced_df, 0.7, 0.1)

# Accuracy evaluate.
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break

    return correct_predictions / num_examples

# Loss of a batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:,-1,:]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

# Loss of dataset
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches
    
# def intermediate_coding():
#     model = import_pretrained_model()

#     tDf, vDf, pDf = prepare_dataset()

#     tokenizer = tiktoken.get_encoding("gpt2")

#     tDt = ClassifyDataset(tDf, tokenizer, max_length=None)
#     vDt = ClassifyDataset(vDf, tokenizer, max_length = tDt.max_length)
#     pDt = ClassifyDataset(pDf, tokenizer, max_length = tDt.max_length)

#     num_workers = 0
#     batch_size = 8

#     tLoader = DataLoader(dataset= tDt, batch_size = batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
#     vLoader = DataLoader(dataset= vDt, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
#     pLoader = DataLoader(dataset= pDt, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
#     # Check data dimension
#     for input_batch, target_batch in tLoader:
#         pass
#     print("Input batch shape:", input_batch.shape)
#     print("Target batch shape:", target_batch.shape)

#     for param in model.parameters():
#         param.requires_grad = False

#     num_classes = 2
#     model.out_head = nn.Linear(768, num_classes)

#     for param in model.trf_blocks[-1].parameters():
#         param.requires_grad = True

#     for param in model.final_norm.parameters():
#         param.requires_grad = True

#     inputs = tokenizer.encode("Hello, how are you?")
#     inputs = torch.tensor(inputs).unsqueeze(0)
#     outputs = model(inputs)
#     # Check input and output shape
#     print("Input shape:", inputs.shape)
#     print("Output shape:", outputs.shape)
#     last_output = outputs[:, -1, :]
#     print("Last output :", last_output)

#     # Get label
#     label = torch.argmax(torch.softmax(last_output, dim=-1))
#     print("Class label: ", label.item())

#     # Accuracy of untrained
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Untrained accuracy, T:{cacl_accuracy_loader(tLoader, model, device)*100:.2f}%, V:{cacl_accuracy_loader(vLoader, model, device)*100:.2f}%, P:{calc_accuary_loader(pLoader, model, device)*100:.2f}%")

#     # Loss of untrained
#     with torch.no_grad():
#         train_loss = calc_loss_loader(tLoader, model, device, num_batches=5)
#         val_loss = calc_loss_loader(vLoader, model, device, num_batches=5)
#         test_loss = calc_loss_loader(pLoader, model, device, num_batches=5)
#     print(f"Initial lost, T:{train_loss:.3f}, V:{val_loss:.3f}, P:{test_loss:.3f}")

# Fine tuning on classify spam ham message
def train_classifier_simple(model,tLoader,vLoader,optimizer,device,num_epochs,eval_freq,eval_iter):
    tLosses, vLosses = [], []
    exmaples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in tLoader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            exmaples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                tLoss, vLoss = evaluate_model(model, tLoader, vLoader, device, eval_iter)
                tLosses.append(tLoss)
                vLosses.append(vLoss)
                print(f"[fine_tuning] Ep {epoch + 1} (Step {global_step: 06d}): "
                      f"Train loss {tLoss:.3f}, Validate loss {vLoss:.3f}")
        
        tAccuracy = calc_accuracy_loader(tLoader, model, device, num_batches=eval_iter)
        vAccuracy = calc_accuracy_loader(vLoader, model, device, num_batches=eval_iter)

        print(f"[fine_tuning] Train accuracy : {tAccuracy * 100: .2f}%, Validate accuracy: {vAccuracy * 100: .2f}%")

def evaluate_model(model, tLoader, vLoader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        tLoss = calc_loss_loader(tLoader, model, device, num_batches=eval_iter)
        vLoss = calc_loss_loader(vLoader, model, device, num_batches=eval_iter)
    model.train()
    return tLoss, vLoss

def fine_tuning_classify():
    model = import_pretrained_model()

    tDf, vDf, pDf = prepare_dataset()

    tokenizer = tiktoken.get_encoding("gpt2")

    tDt = ClassifyDataset(tDf, tokenizer, max_length=None)
    vDt = ClassifyDataset(vDf, tokenizer, max_length = tDt.max_length)
    pDt = ClassifyDataset(pDf, tokenizer, max_length = tDt.max_length)

    num_workers = 0
    batch_size = 8

    tLoader = DataLoader(dataset= tDt, batch_size = batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    vLoader = DataLoader(dataset= vDt, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    pLoader = DataLoader(dataset= pDt, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay = 0.1)
    num_epochs = 5
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_classifier_simple(model, tLoader, vLoader, optimizer, device, num_epochs, eval_freq=50, eval_iter=5)

    model.eval()
    print(f"[fine_tuning] Test accuracy : {calc_loss_loader(pLoader, model, device, num_batches=5) * 100:.2f}%")