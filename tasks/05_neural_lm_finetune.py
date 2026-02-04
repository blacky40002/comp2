import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import copy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score
import seville.tasks.utils_shared as utils

AUTHORS = ["primo_autore", "secondo_autore", "terzo_autore"]
AUTHOR_TO_ID = {a: i for i, a in enumerate(AUTHORS)}
ID_TO_AUTHOR = {i: a for i, a in enumerate(AUTHORS)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

class AuthorshipDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(ids, attention_mask=mask, labels=labels)
            total_loss += outputs.loss.item()
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels

def run(base_path, num_epochs=6, batch_size=16, lr=1e-5):
    print("TASK 5: DistilRoBERTa\n")
    raw = utils.load_dataset_flat(base_path)
    data = {}
    for split, items in raw.items():
        data[split] = [(t, AUTHOR_TO_ID[a]) for t, a in items]

    tok = RobertaTokenizer.from_pretrained("distilroberta-base")
    loaders = {}
    for split in data:
        texts, labels = zip(*data[split])
        ds = AuthorshipDataset(texts, labels, tok)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "training"))

    model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=len(AUTHORS)).to(DEVICE)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sch = get_linear_schedule_with_warmup(opt, 0, len(loaders["training"]) * num_epochs)

    train_losses, val_losses, val_accs = [], [], []
    best_acc = 0.0
    best_state = None
    best_epoch = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in loaders["training"]:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            opt.zero_grad()
            loss = model(ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            epoch_loss += loss.item()

        v_loss, v_acc, _, _ = evaluate(model, loaders["eval"])
        train_losses.append(epoch_loss / len(loaders["training"]))
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        print(f"Epoca {epoch}: Train Loss={train_losses[-1]:.4f}, Val Loss={v_loss:.4f}, Val Acc={v_acc:.4f}")

        if v_acc > best_acc:
            best_acc = v_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    print(f"\nMigliore epoca: {best_epoch} (Val Acc={best_acc:.4f})")
    model.load_state_dict(best_state)
    _, _, preds, true = evaluate(model, loaders["test"])
    p_names, t_names = [ID_TO_AUTHOR[i] for i in preds], [ID_TO_AUTHOR[i] for i in true]
    utils.print_evaluation(t_names, p_names, AUTHORS, "TEST SET")

    try:
        import utils_plot
        utils_plot.set_style()
        # uso val_accs due volte per compatibilità con la firma della funzione che aspetta anche F1
        utils_plot.plot_training_curves(train_losses, val_losses, val_accs, val_accs, "05_training_curves.png")
        utils_plot.plot_confusion_matrix(t_names, p_names, AUTHORS, "Task 5 CM", "05_confusion_matrix.png")
    except ImportError:
        pass

if __name__ == "__main__":
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run(os.path.join(main_dir, "dataset_authorship_finale"))
