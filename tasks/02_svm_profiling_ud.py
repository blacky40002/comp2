"""
Task 1: SVM lineare con features Profiling-UD.

Per preparare i dati:
1. Zippa tutti i file .txt da dataset_authorship_finale/
2. Carica su http://linguistic-profiling.italianlp.it/
3. Scarica il CSV e aggiorna il path sotto
"""

import os
import sys
# Aggiungi la root del progetto al path per permettere l'import di seville
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import re
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
import seville.tasks.utils_shared as utils

AUTHORS = ["primo_autore", "secondo_autore", "terzo_autore"]

def load_profiling_features(csv_path):
    """Carica features dal CSV di Profiling-UD."""
    dataset = {}
    feature_names = None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if feature_names is None:
                feature_names = row[1:]
                continue
            doc_id = os.path.basename(row[0])
            # Usa la funzione centralizzata per il parsing
            split, author = utils.parse_filename(doc_id)
            if split:
                dataset[doc_id] = {
                    "split": split,
                    "author": author,
                    "features": [float(x) for x in row[1:]],
                }
    return feature_names, dataset

def run(csv_path):
    print("=" * 60)
    print("TASK 2: SVM + PROFILING-UD")
    print("=" * 60)

    feature_names, dataset = load_profiling_features(csv_path)
    
    # Divisione dati manuale e robusta
    X_train, y_train = [], []
    X_test, y_test = [], []
    X_eval, y_eval = [], []
    
    for d in dataset.values():
        if d["split"] == "training":
            X_train.append(d["features"])
            y_train.append(d["author"])
        elif d["split"] == "test":
            X_test.append(d["features"])
            y_test.append(d["author"])
        elif d["split"] == "eval":
            X_eval.append(d["features"])
            y_eval.append(d["author"])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_eval, y_eval = np.array(X_eval), np.array(y_eval)

    if len(X_train) == 0:
        print("ERRORE: Training set vuoto!")
        return

    # 1. CROSS-VALIDATION sul Training Set (Parametro C di default)
    print("\n[1/4] Cross-validation (5-fold) sul Training Set...")
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, val_idx in splitter.split(X_train):
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_train[train_idx])
        X_va = scaler.transform(X_train[val_idx])
        m = LinearSVC(dual=False, max_iter=10000, random_state=42).fit(X_tr, y_train[train_idx])
        cv_scores.append(f1_score(y_train[val_idx], m.predict(X_va), average='macro'))
    print(f"  Media F1-Macro CV: {np.mean(cv_scores):.4f}")

    # 2. SELEZIONE MODELLO (Tuning di C sull'Eval Set)
    print("\n[2/4] Selezione iperparametri (C) sull'Eval Set...")
    best_acc = -1
    best_c = 1.0
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    for c in [0.01, 0.1, 1.0, 10.0, 20.0, 30.0,40.0]:
        m = LinearSVC(dual=False, C=c, max_iter=10000, random_state=42, class_weight="balanced").fit(X_train_scaled, y_train)
        score = accuracy_score(y_eval, m.predict(X_eval_scaled))
        print(f"  C={c:<5} -> Eval Accuracy={score:.4f}")
        if score > best_acc:
            best_acc = score
            best_c = c
    print(f"  Miglior C scelto: {best_c}")

    # 3. RETRAINING FINALE (Training + Eval)
    print("\n[3/4] Retraining finale su Training + Eval set...")
    X_full = np.concatenate([X_train, X_eval])
    y_full = np.concatenate([y_train, y_eval])
    
    final_scaler = MinMaxScaler()
    X_full_scaled = final_scaler.fit_transform(X_full)
    
    final_svm = LinearSVC(dual=False, C=best_c, max_iter=10000, random_state=42, class_weight="balanced")
    final_svm.fit(X_full_scaled, y_full)

    # 4. VALUTAZIONE FINALE sul Test Set
    print("\n[4/4] Valutazione finale sul TEST SET...")
    X_test_scaled = final_scaler.transform(X_test)
    test_pred = final_svm.predict(X_test_scaled)
    utils.print_evaluation(y_test, test_pred, labels=final_svm.classes_, title="RISULTATI TEST SET")

    # Plotting
    try:
        import utils_plot
        utils_plot.set_style()
        utils_plot.plot_confusion_matrix(y_test, test_pred, final_svm.classes_, 
                                       "Task 2 - Confusion Matrix", "02_confusion_matrix.png")
        for idx, author in enumerate(final_svm.classes_):
            utils_plot.plot_feature_importance(feature_names, final_svm.coef_[idx], 
                                             f"Top Features: {author}", f"02_feature_importance_{author}.png")
    except ImportError:
        pass

    return final_svm, final_scaler

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    possible_csv = [
        os.path.join(base_dir, "ProfilingUD documents parsed", "17566.csv"),
        os.path.join(base_dir, "profiling_ud_features.csv"),
    ]
    csv_path = next((p for p in possible_csv if os.path.exists(p)), None)
    if csv_path:
        run(csv_path)
    else:
        print("CSV non trovato.")

if __name__ == "__main__":
    main()
