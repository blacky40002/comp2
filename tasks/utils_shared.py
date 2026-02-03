"""Funzioni condivise per il caricamento dati e la valutazione."""

import os
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Regex unica per tutto il progetto
FILENAME_PATTERN = re.compile(r"^(training|test|eval)___(.+)___(\d+)")

def parse_filename(filename):
    clean_name = filename
    for ext in [".txt", ".conllu"]:
        if clean_name.endswith(ext):
            clean_name = clean_name[:-len(ext)]
            break
    m = FILENAME_PATTERN.match(clean_name)
    if m:
        return m.group(1), m.group(2)
    return None, None

def load_dataset_flat(dataset_dir):
    data = {"training": [], "test": [], "eval": []}
    if not os.path.exists(dataset_dir):
        return data
    for filename in sorted(os.listdir(dataset_dir)):
        split, author = parse_filename(filename)
        if not split: continue
        file_path = os.path.join(dataset_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text: data[split].append((text, author))
    return data

def print_evaluation(y_true, y_pred, labels=None, title="VALUTAZIONE"):
    """Stampa solo l'Accuracy e i dettagli per classe, rimuovendo le medie aggregate."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{title}:")
    print(f"  Accuracy: {acc:.4f}")
    
    # Genera il report e rimuovi le ultime righe (macro avg e weighted avg)
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    lines = report.split('\n')
    # Teniamo solo le intestazioni, i dati per classe e la riga dell'accuracy
    filtered_report = "\n".join([line for line in lines if "avg" not in line])
    print(filtered_report)
    
    return acc