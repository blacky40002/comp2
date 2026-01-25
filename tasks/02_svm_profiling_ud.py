"""
Task 1: SVM lineare con features Profiling-UD.

Per preparare i dati:
1. Zippa le cartelle training_set, test_set, eval_set da dataset_authorship_finale
2. Carica su http://linguistic-profiling.italianlp.it/
3. Scarica il CSV e aggiorna il path sotto
"""

import os
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.dummy import DummyClassifier

AUTHORS = ["primo_autore", "secondo_autore", "terzo_autore"]
AUTHOR_MAP = {
    "primo autore": "primo_autore",
    "secondo autore": "secondo_autore",
    "terzo autore": "terzo_autore",
    "primo_autore": "primo_autore",
    "secondo_autore": "secondo_autore",
    "terzo_autore": "terzo_autore",
}


def build_dataset_from_structure(base_path):
    """Costruisce dizionario dataset dalla struttura cartelle."""
    dataset = {}
    split_map = {"training_set": "training", "test_set": "test", "eval_set": "eval"}

    for split_dir, split_name in split_map.items():
        split_path = os.path.join(base_path, split_dir)
        if not os.path.exists(split_path):
            continue

        for author_dir in os.listdir(split_path):
            author_path = os.path.join(split_path, author_dir)
            if not os.path.isdir(author_path):
                continue

            author = AUTHOR_MAP.get(author_dir.lower(), author_dir)

            for filename in os.listdir(author_path):
                if not filename.endswith(".txt"):
                    continue

                doc_id = f"{split_dir}/{author_dir}/{filename}".replace(".txt", "")
                dataset[doc_id] = {
                    "split": split_name,
                    "author": author,
                    "features": None,
                }

    return dataset


def load_profiling_features(csv_path, dataset):
    """Carica features dal CSV di Profiling-UD e le aggiunge al dataset."""
    feature_names = None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if feature_names is None:
                feature_names = row[1:]
                continue

            filename = row[0]
            features = [float(x) for x in row[1:]]

            matched = False
            for doc_id in dataset:
                if doc_id in filename or filename in doc_id:
                    dataset[doc_id]["features"] = features
                    matched = True
                    break

            if not matched:
                for key, value in AUTHOR_MAP.items():
                    if key in filename.lower():
                        author = value
                        break
                else:
                    continue

                if "training" in filename.lower():
                    split = "training"
                elif "test" in filename.lower():
                    split = "test"
                elif "eval" in filename.lower():
                    split = "eval"
                else:
                    continue

                doc_id = filename.replace(".conllu", "").replace(".txt", "")
                dataset[doc_id] = {
                    "split": split,
                    "author": author,
                    "features": features,
                }

    return feature_names, dataset


def train_test_split(dataset):
    """Divide dataset in train, test, eval."""
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    eval_features, eval_labels = [], []

    for doc_id, doc in dataset.items():
        if doc["features"] is None:
            continue

        features = doc["features"]
        label = doc["author"]
        split = doc["split"]

        if split == "training":
            train_features.append(features)
            train_labels.append(label)
        elif split == "test":
            test_features.append(features)
            test_labels.append(label)
        elif split == "eval":
            eval_features.append(features)
            eval_labels.append(label)

    return (
        np.array(train_features),
        np.array(train_labels),
        np.array(test_features),
        np.array(test_labels),
        np.array(eval_features),
        np.array(eval_labels),
    )


def cross_validation_5fold(X_train, y_train):
    """Cross-validation 5-fold con scaling dentro ogni fold."""
    print("\nCROSS-VALIDATION 5-FOLD")
    print("=" * 50)

    splitter = KFold(n_splits=5, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []
    fold_accuracies = []
    fold_f1s = []

    for i, (train_idx, test_idx) in enumerate(splitter.split(X_train)):
        fold_X_train = X_train[train_idx]
        fold_y_train = y_train[train_idx]
        fold_X_test = X_train[test_idx]
        fold_y_test = y_train[test_idx]

        scaler = MinMaxScaler()
        fold_X_train = scaler.fit_transform(fold_X_train)
        fold_X_test = scaler.transform(fold_X_test)

        svm = LinearSVC(dual=False, max_iter=10000, random_state=42)
        svm.fit(fold_X_train, fold_y_train)
        fold_pred = svm.predict(fold_X_test)

        fold_acc = accuracy_score(fold_y_test, fold_pred)
        fold_f1 = f1_score(fold_y_test, fold_pred, average="macro")

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(fold_X_train, fold_y_train)
        baseline_acc = baseline.score(fold_X_test, fold_y_test)

        all_y_true.extend(fold_y_test.tolist())
        all_y_pred.extend(fold_pred.tolist())
        fold_accuracies.append(fold_acc)
        fold_f1s.append(fold_f1)

        print(f"Fold {i+1}: Acc={fold_acc:.4f}, F1={fold_f1:.4f}, Baseline={baseline_acc:.4f}")

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average="macro")

    print(f"\nOverall: Acc={overall_acc:.4f}, F1={overall_f1:.4f}")
    print(f"Mean: Acc={np.mean(fold_accuracies):.4f}±{np.std(fold_accuracies):.4f}, F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f}")

    return all_y_true, all_y_pred


def get_top_features(svm, feature_names, top_n=20):
    """Estrae le top N features per ogni autore."""
    print(f"\nTOP {top_n} FEATURES PER AUTORE")
    print("=" * 50)

    coefs = svm.coef_

    for idx, author in enumerate(svm.classes_):
        print(f"\n{author.upper()}:")
        class_coefs = coefs[idx]

        feature_importance = list(zip(feature_names, class_coefs))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        for i, (feat, coef) in enumerate(feature_importance[:top_n], 1):
            sign = "+" if coef > 0 else "-"
            print(f"  {i:2d}. {feat:<30} {sign}{abs(coef):.4f}")


def run(csv_path, base_path=None):
    """Pipeline completa."""
    print("=" * 60)
    print("TASK 1: SVM + PROFILING-UD")
    print("=" * 60)

    if base_path:
        dataset = build_dataset_from_structure(base_path)
        print(f"Documenti da struttura: {len(dataset)}")
    else:
        dataset = {}

    feature_names, dataset = load_profiling_features(csv_path, dataset)
    print(f"Features: {len(feature_names)}")
    print(f"Documenti con features: {sum(1 for d in dataset.values() if d['features'] is not None)}")

    X_train, y_train, X_test, y_test, X_eval, y_eval = train_test_split(dataset)

    print(f"\nDataset:")
    print(f"  Training: {len(X_train)}")
    print(f"  Test: {len(X_test)}")
    print(f"  Eval: {len(X_eval)}")

    if len(X_train) == 0:
        print("ERRORE: Training set vuoto!")
        return

    cross_validation_5fold(X_train, y_train)

    print("\n" + "=" * 50)
    print("VALUTAZIONE FINALE")
    print("=" * 50)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    svm = LinearSVC(dual=False, max_iter=10000, random_state=42)
    svm.fit(X_train_scaled, y_train)

    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
        test_pred = svm.predict(X_test_scaled)
        print(f"\nTEST SET:")
        print(f"  Accuracy: {accuracy_score(y_test, test_pred):.4f}")
        print(f"  F1-Macro: {f1_score(y_test, test_pred, average='macro'):.4f}")
        print(classification_report(y_test, test_pred, zero_division=0))

    if len(X_eval) > 0:
        X_eval_scaled = scaler.transform(X_eval)
        eval_pred = svm.predict(X_eval_scaled)
        print(f"\nEVAL SET:")
        print(f"  Accuracy: {accuracy_score(y_eval, eval_pred):.4f}")
        print(f"  F1-Macro: {f1_score(y_eval, eval_pred, average='macro'):.4f}")
        print(classification_report(y_eval, eval_pred, zero_division=0))

    get_top_features(svm, feature_names, top_n=20)

    return svm, scaler, feature_names


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(base_dir, "dataset_authorship_finale")

    possible_csv = [
        os.path.join(base_dir, "profiling_ud_features.csv"),
        os.path.join(base_dir, "16596.csv"),
    ]

    csv_path = None
    for path in possible_csv:
        if os.path.exists(path):
            csv_path = path
            break

    if csv_path is None:
        print("ERRORE: CSV Profiling-UD non trovato!")
        print("\nIstruzioni:")
        print("1. Zippa le cartelle in dataset_authorship_finale/")
        print("2. Carica su http://linguistic-profiling.italianlp.it/")
        print("3. Scarica il CSV e salvalo come profiling_ud_features.csv")
        return

    run(csv_path, dataset_path)


if __name__ == "__main__":
    main()
