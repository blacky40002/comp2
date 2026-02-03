"""Funzione condivisa per caricare il dataset flat."""

import os
import re

_PATTERN = re.compile(r"^(training|test|eval)___(.+)___\d+\.txt$")


def load_dataset(dataset_dir):
    """Carica il dataset flat. File: {split}___{autore}___{indice}.txt"""
    data = {"training": [], "test": [], "eval": []}
    for filename in sorted(os.listdir(dataset_dir)):
        m = _PATTERN.match(filename)
        if not m:
            continue
        with open(os.path.join(dataset_dir, filename), "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            data[m.group(1)].append((text, m.group(2)))
    return data
