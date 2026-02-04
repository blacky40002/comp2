
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
    vocab = {w for t, _ in texts for w in t.lower().split()}
    emb = {}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    batch = list(vocab)
    for i in range(0, len(batch), 500):
        chunk = batch[i:i + 500]
        cur.execute(f"SELECT * FROM store WHERE key IN ({','.join('?' * len(chunk))})", chunk)
        for row in cur.fetchall():
            emb[row[0]] = np.array(row[1:EMBEDDING_DIM + 1], dtype=np.float32)
    conn.close()
    return emb


CONCAT_DIMS = {"min+max": 2, "mean+max+std": 3}

def doc_vector(text, emb, method="mean"):
    words = text.lower().split()
    vecs = [emb[w] for w in words if w in emb]
    if not vecs:
        n = CONCAT_DIMS.get(method, 1)
        return np.zeros(EMBEDDING_DIM * n)
    arr = np.array(vecs)
    if method == "max":
        return arr.max(axis=0)
    if method == "sum":
        return arr.sum(axis=0)
    if method == "min":
        return arr.min(axis=0)
    if method == "median":
        return np.median(arr, axis=0)
    if method == "min+max":
        return np.concatenate([arr.min(axis=0), arr.max(axis=0)])
    if method == "mean+max+std":
        return np.concatenate([arr.mean(axis=0), arr.max(axis=0), arr.std(axis=0)])
    return arr.mean(axis=0)


def run(dataset_path, db_path):

    data = utils.load_dataset_flat(dataset_path)
    all_texts = data["training"] + data["eval"] + data["test"]
    emb = load_embeddings(db_path, all_texts)
    print(f"Embeddings caricati: {len(emb)}")

    y_tr = np.array([a for _, a in data["training"]])
    y_ev = np.array([a for _, a in data["eval"]])

    # confronto mean / max / sum / min / median
    results = []
    print("\nValidazione aggregazioni")
    for method in ("mean", "max", "sum", "min", "median", "min+max", "mean+max+std"):
        X_tr = np.array([doc_vector(t, emb, method) for t, _ in data["training"]])
        scaler = MinMaxScaler()
        svm = LinearSVC(dual=False, max_iter=5000, class_weight="balanced", random_state=42)
        svm.fit(scaler.fit_transform(X_tr), y_tr)

        X_ev = np.array([doc_vector(t, emb, method) for t, _ in data["eval"]])
        acc = accuracy_score(y_ev, svm.predict(scaler.transform(X_ev)))
        print(f"  {method}: Acc={acc:.4f}")
        results.append({"name": method, "acc": acc})

    results.sort(key=lambda x: x["acc"], reverse=True)
    best = results[0]["name"]
    print(f"\nMigliore: {best} (Acc={results[0]['acc']:.4f})")

    # Retraining su training + eval
    full = data["training"] + data["eval"]
    X_full = np.array([doc_vector(t, emb, best) for t, _ in full])
    y_full = np.array([a for _, a in full])
    final_scaler = MinMaxScaler()
    final_svm = LinearSVC(dual=False, max_iter=5000, class_weight="balanced", random_state=42)
    final_svm.fit(final_scaler.fit_transform(X_full), y_full)

    #Test finale
    print("\n Valutazione TEST SET")
    X_te = np.array([doc_vector(t, emb, best) for t, _ in data["test"]])
    y_te = np.array([a for _, a in data["test"]])
    test_pred = final_svm.predict(final_scaler.transform(X_te))
    utils.print_evaluation(y_te, test_pred, labels=final_svm.classes_, title="RISULTATI TEST SET")

    try:
        import utils_plot
        utils_plot.set_style()
        plot_data = [{"name": r["name"], "f1_macro": r["acc"]} for r in results]
        utils_plot.plot_model_comparison(plot_data, "f1_macro", "Task 4 - Model Comparison", "04_model_comparison.png")
        utils_plot.plot_confusion_matrix(y_te, test_pred, final_svm.classes_, "Task 4 - Confusion Matrix", "04_confusion_matrix.png")
    except ImportError:
        pass


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run(os.path.join(base_dir, "dataset_authorship_finale"), os.path.join(base_dir, "ukwac128.sqlite"))


if __name__ == "__main__":
    main()
