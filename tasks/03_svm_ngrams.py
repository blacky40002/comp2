"""
Task 2: Classificatore basato su SVM lineari e n-grammi.

Utilizza n-grammi di:
- Caratteri
- Parole (forme)
- Lemmi
- Part-of-speech (POS)

Richiede: spaCy con modello inglese (python -m spacy download en_core_web_sm)

Output:
- Confronto diverse configurazioni sul validation set
- Valutazione miglior sistema su test set
"""

import os
from collections import defaultdict
import numpy as np
import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

AUTHORS = ["primo autore", "secondo autore", "terzo autore"]

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


class Token:
    """Token minimale per parola, lemma e POS."""

    def __init__(self, word, lemma, pos_tag):
        self.word = word
        self.lemma = lemma
        self.pos_tag = pos_tag

    def get_num_chars(self):
        return len(self.word)


class Sentence:
    """Frase con lista di token."""

    def __init__(self):
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)

    def get_words(self):
        return [token.word for token in self.tokens]

    def get_lemmas(self):
        return [token.lemma for token in self.tokens]

    def get_pos(self):
        return [token.pos_tag for token in self.tokens]

    def get_num_tokens(self):
        return len(self.tokens)

    def get_num_chars(self):
        if not self.tokens:
            return 0
        num_chars = sum(token.get_num_chars() for token in self.tokens)
        return num_chars + self.get_num_tokens() - 1


class Document:
    """Rappresenta un documento con frasi annotate."""

    def __init__(self, doc_id, author, split):
        self.doc_id = doc_id
        self.author = author
        self.split = split
        self.sentences = []
        self.text = ""

    def process_with_spacy(self, text):
        """Processa il testo con spaCy per estrarre token, lemmi e POS."""
        self.text = text
        nlp = get_nlp()
        doc = nlp(text)

        sentences = list(doc.sents) if doc.has_annotation("SENT_START") else []
        if not sentences:
            sentences = [doc]

        for sent in sentences:
            sentence = Sentence()
            for token in sent:
                if token.is_space:
                    continue
                sentence.add_token(
                    Token(token.text.lower(), token.lemma_.lower(), token.pos_)
                )
            if sentence.get_num_tokens() > 0:
                self.sentences.append(sentence)

    def num_tokens(self):
        return sum(sentence.get_num_tokens() for sentence in self.sentences)

    def num_chars(self):
        return sum(sentence.get_num_chars() for sentence in self.sentences)


def load_documents(base_path):
    """Carica documenti dalla struttura di cartelle."""
    datasets = {"training": [], "test": [], "eval": []}
    split_map = {"training_set": "training", "test_set": "test", "eval_set": "eval"}

    print("Caricamento documenti...")

    for split_dir, split_name in split_map.items():
        split_path = os.path.join(base_path, split_dir)

        if not os.path.exists(split_path):
            print(f"  Cartella non trovata: {split_path}")
            continue

        for author in AUTHORS:
            author_path = os.path.join(split_path, author)

            if not os.path.exists(author_path):
                continue

            doc_count = 0
            for filename in os.listdir(author_path):
                if not filename.endswith(".txt"):
                    continue

                file_path = os.path.join(author_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                except Exception as e:
                    continue

                if not text:
                    continue

                doc_id = f"{author}_{filename}"
                doc = Document(doc_id, author, split_name)
                doc.process_with_spacy(text)

                if doc.num_tokens() > 0:
                    datasets[split_name].append(doc)
                    doc_count += 1

            print(f"  {split_name}/{author}: {doc_count} documenti")

    return datasets


def extract_char_ngrams(text, n):
    """Estrae n-grammi di caratteri da una stringa."""
    ngrams = {}

    for i in range(len(text) - n + 1):
        ngram = f"CHAR_{n}_{text[i:i+n]}"
        ngrams[ngram] = ngrams.get(ngram, 0) + 1

    return ngrams


def extract_token_ngrams(tokens, n, prefix):
    """Estrae n-grammi da lista di token."""
    ngrams = {}

    for i in range(len(tokens) - n + 1):
        ngram = f"{prefix}_{n}_" + "_".join(tokens[i : i + n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1

    return ngrams


def extract_document_features(doc, config):
    """Estrae features da un documento secondo la configurazione."""
    token_features = {}
    char_features = {}

    for sentence in doc.sentences:
        words = sentence.get_words()
        lemmas = sentence.get_lemmas()
        pos_tags = sentence.get_pos()
        sentence_text = " ".join(words)

        # N-grammi di caratteri (per frase)
        if "char_ngrams" in config:
            for n in config["char_ngrams"]:
                features = extract_char_ngrams(sentence_text, n)
                for key, value in features.items():
                    char_features[key] = char_features.get(key, 0) + value

        # N-grammi di parole
        if "word_ngrams" in config:
            for n in config["word_ngrams"]:
                features = extract_token_ngrams(words, n, "WORD")
                for key, value in features.items():
                    token_features[key] = token_features.get(key, 0) + value

        # N-grammi di lemmi
        if "lemma_ngrams" in config:
            for n in config["lemma_ngrams"]:
                features = extract_token_ngrams(lemmas, n, "LEMMA")
                for key, value in features.items():
                    token_features[key] = token_features.get(key, 0) + value

        # N-grammi di POS
        if "pos_ngrams" in config:
            for n in config["pos_ngrams"]:
                features = extract_token_ngrams(pos_tags, n, "POS")
                for key, value in features.items():
                    token_features[key] = token_features.get(key, 0) + value

    # Normalizzazione
    if config.get("normalize", True):
        num_words = doc.num_tokens()
        num_chars = doc.num_chars()

        if num_words > 0:
            for key in token_features:
                token_features[key] = token_features[key] / num_words
        if num_chars > 0:
            for key in char_features:
                char_features[key] = char_features[key] / num_chars

    return {**token_features, **char_features}


def filter_rare_features(features_list, min_docs=2):
    """Filtra features che appaiono in meno di min_docs documenti."""
    feature_counts = defaultdict(int)
    for doc_features in features_list:
        for feature in doc_features.keys():
            feature_counts[feature] += 1

    filtered = []
    for doc_features in features_list:
        filtered.append(
            {f: v for f, v in doc_features.items() if feature_counts[f] >= min_docs}
        )

    num_before = len(feature_counts)
    num_after = len([f for f, c in feature_counts.items() if c >= min_docs])
    print(f"  Features filtrate: {num_before} -> {num_after}")

    return filtered


def prepare_data(documents, config, vectorizer=None, scaler=None, fit=True):
    """Prepara dati per training/test."""
    features_list = []
    labels = []

    for doc in documents:
        features_list.append(extract_document_features(doc, config))
        labels.append(doc.author)

    # Filtra features rare solo in training
    if fit and config.get("min_docs", 2) > 1:
        features_list = filter_rare_features(features_list, config["min_docs"])

    # Vectorize
    if fit:
        vectorizer = DictVectorizer()
        X = vectorizer.fit_transform(features_list)
    else:
        X = vectorizer.transform(features_list)

    y = np.array(labels)

    # Scale
    if fit:
        scaler = MaxAbsScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, vectorizer, scaler


def get_configurations():
    """Definisce le configurazioni da testare."""
    return [
        {
            "name": "Char_1-3",
            "char_ngrams": [1, 2, 3],
            "normalize": True,
            "min_docs": 2,
        },
        {
            "name": "Word_1-2",
            "word_ngrams": [1, 2],
            "normalize": True,
            "min_docs": 2,
        },
        {
            "name": "Lemma_1-2",
            "lemma_ngrams": [1, 2],
            "normalize": True,
            "min_docs": 2,
        },
        {
            "name": "POS_1-3",
            "pos_ngrams": [1, 2, 3],
            "normalize": True,
            "min_docs": 2,
        },
        {
            "name": "Word_Char",
            "word_ngrams": [1, 2],
            "char_ngrams": [2, 3],
            "normalize": True,
            "min_docs": 2,
        },
        {
            "name": "Lemma_POS",
            "lemma_ngrams": [1, 2],
            "pos_ngrams": [1, 2, 3],
            "normalize": True,
            "min_docs": 2,
        },
        {
            "name": "All_Features",
            "char_ngrams": [2, 3],
            "word_ngrams": [1, 2],
            "lemma_ngrams": [1, 2],
            "pos_ngrams": [1, 2, 3],
            "normalize": True,
            "min_docs": 3,
        },
    ]


def analyze_feature_importance(svm, vectorizer, top_n=15):
    """Analizza le features piu importanti."""
    feature_names = vectorizer.get_feature_names_out()
    coefs = svm.coef_

    print(f"\nTOP {top_n} FEATURES PER AUTORE:")
    print("-" * 50)

    for idx, author in enumerate(AUTHORS):
        print(f"\n{author.upper()}:")
        author_coefs = coefs[idx] if len(coefs.shape) > 1 else coefs

        pairs = list(zip(feature_names, author_coefs))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        for i, (feature, coef) in enumerate(pairs[:top_n], 1):
            direction = "+" if coef > 0 else "-"
            print(f"  {i:2d}. {feature:<35} {direction} {abs(coef):.4f}")


def run(base_path):
    """Pipeline completa per SVM con n-grammi."""
    print("=" * 70)
    print("TASK 2: SVM + N-GRAMMI")
    print("=" * 70)

    # 1. Caricamento dati
    print("\nFASE 1: CARICAMENTO DATI")
    print("-" * 40)

    datasets = load_documents(base_path)
    train_docs = datasets["training"]
    test_docs = datasets["test"]
    eval_docs = datasets["eval"]

    print(f"\nRiepilogo:")
    print(f"  Training: {len(train_docs)} documenti")
    print(f"  Test: {len(test_docs)} documenti")
    print(f"  Eval: {len(eval_docs)} documenti")

    if len(train_docs) == 0:
        print("ERRORE: Nessun documento di training!")
        return

    # 2. Test configurazioni
    print("\nFASE 2: TEST CONFIGURAZIONI (Validation Set)")
    print("-" * 40)

    configs = get_configurations()
    all_results = []

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config['name']}")

        try:
            X_train, y_train, vectorizer, scaler = prepare_data(
                train_docs, config, fit=True
            )
            print(f"  Matrice training: {X_train.shape}")

            svm = LinearSVC(
                dual=False, max_iter=10000, random_state=42, class_weight="balanced"
            )
            svm.fit(X_train, y_train)

            if not eval_docs:
                print("  ERRORE: Eval set vuoto, impossibile selezionare il modello.")
                return

            X_eval, y_eval, _, _ = prepare_data(
                eval_docs, config, vectorizer, scaler, fit=False
            )
            pred_eval = svm.predict(X_eval)

            eval_acc = accuracy_score(y_eval, pred_eval)
            eval_f1 = f1_score(y_eval, pred_eval, average="macro")

            all_results.append(
                {
                    "config_name": config["name"],
                    "eval_accuracy": eval_acc,
                    "eval_f1_macro": eval_f1,
                }
            )

            print(f"  Eval Accuracy: {eval_acc:.4f}")
            print(f"  Eval F1-Macro: {eval_f1:.4f}")

        except Exception as e:
            print(f"  ERRORE: {str(e)}")

    # 3. Confronto risultati
    print("\nFASE 3: CONFRONTO RISULTATI")
    print("-" * 40)

    print(f"\n{'Configurazione':<20} {'Eval Acc':<12} {'Eval F1':<12}")
    print("-" * 60)

    for r in all_results:
        print(
            f"{r['config_name']:<20} {r['eval_accuracy']:.4f}       {r['eval_f1_macro']:.4f}"
        )

    # 4. Selezione migliore e valutazione finale
    print("\nFASE 4: VALUTAZIONE MIGLIORE CONFIGURAZIONE")
    print("-" * 40)

    best = max(all_results, key=lambda x: x["eval_f1_macro"])
    best_config = next(c for c in configs if c["name"] == best["config_name"])

    print(f"\nMigliore: {best['config_name']}")
    print(f"  Eval Accuracy: {best['eval_accuracy']:.4f}")
    print(f"  Eval F1-Macro: {best['eval_f1_macro']:.4f}")

    # Training finale
    X_train, y_train, vectorizer, scaler = prepare_data(train_docs, best_config, fit=True)

    svm = LinearSVC(
        dual=False, max_iter=10000, random_state=42, class_weight="balanced"
    )
    svm.fit(X_train, y_train)

    # Valutazione su test
    if test_docs:
        X_test, y_test, _, _ = prepare_data(
            test_docs, best_config, vectorizer, scaler, fit=False
        )
        pred = svm.predict(X_test)

        print(f"\nTEST SET ({len(test_docs)} documenti):")
        print(f"  Accuracy: {accuracy_score(y_test, pred):.4f}")
        print(f"  F1-Macro: {f1_score(y_test, pred, average='macro'):.4f}")
        print(classification_report(y_test, pred, target_names=AUTHORS, zero_division=0))

    # Feature importance
    analyze_feature_importance(svm, vectorizer, top_n=15)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETATA!")
    print("=" * 70)

    return svm, vectorizer, scaler, all_results


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(base_dir, "dataset_authorship_finale")

    if not os.path.exists(dataset_path):
        print(f"ERRORE: Dataset non trovato: {dataset_path}")
        print("Esegui prima 01_data_preparation.py")
        return

    run(dataset_path)


if __name__ == "__main__":
    main()
