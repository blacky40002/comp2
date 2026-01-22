import os
import sqlite3
from collections import Counter, namedtuple
import numpy as np
import spacy
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

Token = namedtuple("Token", ["word", "pos"])
Document = namedtuple("Document", ["tokens", "author", "split"])

EMBEDDING_DIM = 128
POS_CONTENT = ["ADJ", "NOUN", "VERB"]

# Carica modello spaCy (lazy loading)
_nlp = None


def get_nlp():
    """Carica il modello spaCy solo quando necessario."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Modello spaCy non trovato. Installalo con:")
            print("  python -m spacy download en_core_web_sm")
            raise
    return _nlp


def load_documents(dataset_path, min_tokens=10):
    documents = []
    split_map = {"training_set": "training", "test_set": "test", "eval_set": "eval"}

    for split_dir, split_name in split_map.items():
        split_path = os.path.join(dataset_path, split_dir)
        if not os.path.exists(split_path):
            continue

        for author in os.listdir(split_path):
            author_path = os.path.join(split_path, author)
            if not os.path.isdir(author_path):
                continue

            for filename in os.listdir(author_path):
                if not filename.endswith(".txt"):
                    continue

                file_path = os.path.join(author_path, filename)
                with open(file_path, "r", encoding="utf-8") as handle:
                    text = handle.read().strip()

                nlp = get_nlp()
                doc = nlp(text)
                tokens = [
                    Token(word=token.text.lower(), pos=token.pos_)
                    for token in doc
                    if not token.is_space and not token.is_punct
                ]

                if len(tokens) >= min_tokens:
                    documents.append(Document(tokens=tokens, author=author, split=split_name))

    return documents


def load_embeddings_from_sqlite(db_path, vocab):
    """Carica word embeddings da database SQLite per le parole nel vocabolario."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database embeddings non trovato: {db_path}")

    embeddings = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for word in vocab:
        cursor.execute("SELECT * FROM store WHERE key = ?", (word,))
        row = cursor.fetchone()
        if row:
            vector = np.array(row[1:129], dtype=np.float32)
            embeddings[word] = vector

    conn.close()
    return embeddings


def compute_embeddings_mean(vectors, dim):
    if not vectors:
        return np.zeros(dim)
    return np.mean(np.array(vectors), axis=0)


def aggregate_vectors(vectors, method, dim):
    if not vectors:
        return np.zeros(dim)
    arr = np.array(vectors)
    if method == "mean":
        return np.mean(arr, axis=0)
    if method == "max":
        return np.max(arr, axis=0)
    if method == "sum":
        return np.sum(arr, axis=0)
    return np.mean(arr, axis=0)


def document_embedding(tokens, embeddings, dim, config):
    method = config["method"]
    pos_filter = config.get("pos_filter")
    pos_sep = config.get("pos_sep", False)

    if pos_sep:
        if not pos_filter:
            return np.zeros(dim)
        parts = []
        for pos_tag in pos_filter:
            vectors = [
                embeddings[token.word]
                for token in tokens
                if token.word in embeddings and token.pos == pos_tag
            ]
            parts.append(aggregate_vectors(vectors, method, dim))
        return np.concatenate(parts, axis=0)

    vectors = [
        embeddings[token.word]
        for token in tokens
        if token.word in embeddings and (not pos_filter or token.pos in pos_filter)
    ]
    return aggregate_vectors(vectors, method, dim)


def extract_vocab(documents, min_freq=2):
    counter = Counter()
    for doc in documents:
        for token in doc.tokens:
            counter[token.word] += 1
    return [word for word, count in counter.items() if count >= min_freq]


def prepare_data(docs, embeddings, dim, config):
    X, y = [], []
    for doc in docs:
        X.append(document_embedding(doc.tokens, embeddings, dim, config))
        y.append(doc.author)
    return np.array(X), np.array(y)


def run(dataset_path, embeddings_db_path):
    documents = load_documents(dataset_path)
    train_docs = [d for d in documents if d.split == "training"]
    test_docs = [d for d in documents if d.split == "test"]
    eval_docs = [d for d in documents if d.split == "eval"]

    vocab = extract_vocab(documents)
    embeddings = load_embeddings_from_sqlite(embeddings_db_path, vocab)
    dim = EMBEDDING_DIM

    print(f"Vocabolario: {len(vocab)} parole, embeddings trovati: {len(embeddings)}")

    configs = [
        {"name": "All_Mean", "method": "mean"},
        {"name": "All_Max", "method": "max"},
        {"name": "All_Sum", "method": "sum"},
        {"name": "Content_Mean", "method": "mean", "pos_filter": POS_CONTENT},
        {
            "name": "Content_SepMean",
            "method": "mean",
            "pos_filter": POS_CONTENT,
            "pos_sep": True,
        },
    ]
    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config['name']}")
        X_train, y_train = prepare_data(train_docs, embeddings, dim, config)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        svm = LinearSVC(dual=False, max_iter=5000, class_weight="balanced", random_state=42)
        svm.fit(X_train, y_train)

        if not eval_docs:
            print("Eval set vuoto, impossibile selezionare il modello.")
            return

        X_eval, y_eval = prepare_data(eval_docs, embeddings, dim, config)
        X_eval = scaler.transform(X_eval)
        pred_eval = svm.predict(X_eval)
        eval_acc = accuracy_score(y_eval, pred_eval)
        eval_f1 = f1_score(y_eval, pred_eval, average="macro")

        results.append({
            "name": config["name"],
            "method": config["method"],
            "eval_f1": eval_f1,
            "eval_acc": eval_acc,
        })

        print(f"  Eval Accuracy: {eval_acc:.4f}")
        print(f"  Eval F1-Macro: {eval_f1:.4f}")

    print("\nRISULTATI VALIDATION")
    print(f"{'Configurazione':<20} {'Eval Acc':<12} {'Eval F1':<12}")
    print("-" * 50)
    for result in results:
        print(
            f"{result['name']:<20} {result['eval_acc']:.4f}       {result['eval_f1']:.4f}"
        )

    best = max(results, key=lambda x: x["eval_f1"])
    best_config = next(c for c in configs if c["name"] == best["name"])

    X_train, y_train = prepare_data(train_docs, embeddings, dim, best_config)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    svm = LinearSVC(dual=False, max_iter=5000, class_weight="balanced", random_state=42)
    svm.fit(X_train, y_train)

    print(f"Best method: {best['name']} (eval_f1={best['eval_f1']:.4f})")

    if test_docs:
        X_test, y_test = prepare_data(test_docs, embeddings, dim, best_config)
        X_test = scaler.transform(X_test)
        pred = svm.predict(X_test)
        print("\nTest:")
        print(f"  accuracy={accuracy_score(y_test, pred):.4f}")
        print(f"  f1_macro={f1_score(y_test, pred, average='macro'):.4f}")
        print(classification_report(y_test, pred, zero_division=0))

    return svm, scaler


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(base_dir, "dataset_authorship_finale")
    embeddings_db_path = os.path.join(base_dir, "ukwac128.sqlite")
    run(dataset_path, embeddings_db_path)


if __name__ == "__main__":
    main()
