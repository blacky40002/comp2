"""
Task 4: Fine-tuning RoBERTa-base per authorship attribution.

Requisiti:
    pip install transformers torch matplotlib

Output:
- Curve di loss training/validation per 6 epoche
- Metriche sul validation set per ogni epoca
- Valutazione finale sul test set
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

AUTHORS = ["primo autore", "secondo autore", "terzo autore"]
AUTHOR_TO_ID = {a: i for i, a in enumerate(AUTHORS)}
ID_TO_AUTHOR = {i: a for i, a in enumerate(AUTHORS)}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


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


def load_data(base_path):
    data = {"training": [], "test": [], "eval": []}
    split_map = {"training_set": "training", "test_set": "test", "eval_set": "eval"}

    for split_dir, split_name in split_map.items():
        split_path = os.path.join(base_path, split_dir)
        if not os.path.exists(split_path):
            continue

        for author in AUTHORS:
            author_path = os.path.join(split_path, author)
            if not os.path.exists(author_path):
                continue

            for filename in os.listdir(author_path):
                if not filename.endswith(".txt"):
                    continue

                with open(os.path.join(author_path, filename), "r", encoding="utf-8") as f:
                    text = f.read().strip()

                if text:
                    data[split_name].append((text, AUTHOR_TO_ID[author]))

    return data


def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1, all_preds, all_labels


def plot_curves(train_losses, val_losses, val_accs, val_f1s, output_path):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, train_losses, "b-o", label="Training")
    axes[0].plot(epochs, val_losses, "r-o", label="Validation")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, val_accs, "g-o")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(True)

    axes[2].plot(epochs, val_f1s, "m-o")
    axes[2].set_xlabel("Epoca")
    axes[2].set_ylabel("F1-Macro")
    axes[2].set_title("Validation F1-Macro")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Grafici salvati: {output_path}")


def run(base_path, num_epochs=6, batch_size=16, lr=2e-5, max_length=128):
    print("=" * 60)
    print("TASK 4: FINE-TUNING RoBERTa-base")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    print("\n[1/5] Caricamento dati...")
    data = load_data(base_path)
    print(f"  Training: {len(data['training'])}")
    print(f"  Validation: {len(data['eval'])}")
    print(f"  Test: {len(data['test'])}")

    print("\n[2/5] Preparazione dataset...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_texts, train_labels = zip(*data["training"])
    eval_texts, eval_labels = zip(*data["eval"])
    test_texts, test_labels = zip(*data["test"])

    train_ds = AuthorshipDataset(train_texts, train_labels, tokenizer, max_length)
    eval_ds = AuthorshipDataset(eval_texts, eval_labels, tokenizer, max_length)
    test_ds = AuthorshipDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    print("\n[3/5] Caricamento RoBERTa-base...")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=len(AUTHORS)
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    print(f"\n[4/5] Training ({num_epochs} epoche)...")
    print("-" * 60)

    train_losses, val_losses, val_accs, val_f1s = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, eval_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(
            f"Epoca {epoch}/{num_epochs}: "
            f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}"
        )

    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_curves(train_losses, val_losses, val_accs, val_f1s,
                os.path.join(output_dir, "training_curves.png"))

    print(f"\n[5/5] Valutazione TEST SET (dopo epoca {num_epochs})...")
    print("-" * 60)

    test_loss, test_acc, test_f1, test_preds, test_true = evaluate(model, test_loader)

    print(f"\nTEST SET:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1-Macro: {test_f1:.4f}")
    print("\n" + classification_report(test_true, test_preds,
          target_names=AUTHORS, zero_division=0))

    print("\nRIEPILOGO PER EPOCA:")
    print(f"{'Epoca':<8}{'Train Loss':<12}{'Val Loss':<12}{'Val Acc':<10}{'Val F1':<10}")
    print("-" * 52)
    for i in range(num_epochs):
        print(f"{i+1:<8}{train_losses[i]:<12.4f}{val_losses[i]:<12.4f}"
              f"{val_accs[i]:<10.4f}{val_f1s[i]:<10.4f}")

    print("\n" + "=" * 60)
    print("COMPLETATO!")
    print("=" * 60)

    return model, tokenizer


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(base_dir, "dataset_authorship_finale")

    if not os.path.exists(dataset_path):
        print(f"ERRORE: Dataset non trovato: {dataset_path}")
        return

    run(dataset_path, num_epochs=6, batch_size=16, lr=2e-5, max_length=128)


if __name__ == "__main__":
    main()
