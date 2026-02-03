"Task 4: SVM lineare con word embeddings."

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import sqlite3
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import seville.tasks.utils_shared as utils

EMBEDDING_DIM = 128

def load_embeddings(db_path, texts):
    vocab = set()
    for t, a in texts:
        for w in t.lower().split(): vocab.add(w)
    
    embeddings = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for word in vocab:
        cursor.execute("SELECT * FROM store WHERE key = ?", (word,))
        row = cursor.fetchone()
        if row: embeddings[word] = np.array(row[1:129], dtype=np.float32)
    conn.close()
    return embeddings

def get_doc_vec(text, embeddings):
    vecs = [embeddings[w] for w in text.lower().split() if w in embeddings]
    if not vecs: return np.zeros(EMBEDDING_DIM)
    return np.sum(vecs, axis=0)

def run(dataset_path, db_path):
    print("TASK 4: SVM + EMBEDDINGS")
    print("-" * 60)

    data_raw = utils.load_dataset_flat(dataset_path)
    embeddings = load_embeddings(db_path, data_raw["training"] + data_raw["eval"] + data_raw["test"])

    # 1. VALIDAZIONE (Accuracy)
    print("\n[1/3] Training modello su Training set...")
    X_tr = np.array([get_doc_vec(t, embeddings) for t, a in data_raw["training"]])
    y_tr = np.array([a for t, a in data_raw["training"]])
    
    scaler = MinMaxScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    svm = LinearSVC(dual=False, max_iter=5000, class_weight="balanced", random_state=42).fit(X_tr_s, y_tr)

    X_ev = np.array([get_doc_vec(t, embeddings) for t, a in data_raw["eval"]])
    y_ev = np.array([a for t, a in data_raw["eval"]])
    X_ev_s = scaler.transform(X_ev)
    
    score = accuracy_score(y_ev, svm.predict(X_ev_s))
    print(f"  Accuracy su Eval set: {score:.4f}")

    # 2. RETRAINING
    print("\n[2/3] Retraining finale su Training + Eval set...")
    X_full = np.concatenate([X_tr, X_ev])
    y_full = np.concatenate([y_tr, y_ev])
    
    final_scaler = MinMaxScaler()
    X_full_s = final_scaler.fit_transform(X_full)
    final_svm = LinearSVC(dual=False, max_iter=5000, class_weight="balanced", random_state=42).fit(X_full_s, y_full)

    # 3. TEST FINALE
    print("\n[3/3] Valutazione finale sul TEST SET...")
    X_te = np.array([get_doc_vec(t, embeddings) for t, a in data_raw["test"]])
    y_te = np.array([a for t, a in data_raw["test"]])
    X_te_s = final_scaler.transform(X_te)
    
    test_pred = final_svm.predict(X_te_s)
    utils.print_evaluation(y_te, test_pred, labels=final_svm.classes_, title="RISULTATI TEST SET")

    try:
        import utils_plot
        utils_plot.set_style()
        utils_plot.plot_confusion_matrix(y_te, test_pred, final_svm.classes_, "Task 4 - Confusion Matrix", "04_confusion_matrix.png")
    except ImportError: pass

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run(os.path.join(base_dir, "dataset_authorship_finale"), os.path.join(base_dir, "ukwac128.sqlite"))

if __name__ == "__main__":
    main()