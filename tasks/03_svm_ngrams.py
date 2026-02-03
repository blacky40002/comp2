"Task 3: SVM lineare con n-grammi."

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import seville.tasks.utils_shared as utils

_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Installare modello: python -m spacy download en_core_web_sm")
            raise
    return _nlp

def extract_features(text, config):
    """Estrae features (n-grammi) da un testo."""
    doc = get_nlp()(text)
    words = [t.text.lower() for t in doc if not t.is_space]
    lemmas = [t.lemma_.lower() for t in doc if not t.is_space]
    pos = [t.pos_ for t in doc if not t.is_space]
    
    feats = {}
    
    # Character n-grams
    if "char_ngrams" in config:
        for n in config["char_ngrams"]:
            for i in range(len(text) - n + 1):
                k = f"CHAR_{n}_{text[i:i+n]}"
                feats[k] = feats.get(k, 0) + 1
                
    # Token n-grams (Word, Lemma, POS)
    for pfx, lst in [("WORD", words), ("LEMMA", lemmas), ("POS", pos)]:
        cfg_key = pfx.lower() + "_ngrams"
        if cfg_key in config:
            for n in config[cfg_key]:
                for i in range(len(lst) - n + 1):
                    k = f"{pfx}_{n}_" + "_".join(lst[i:i+n])
                    feats[k] = feats.get(k, 0) + 1
    
    # Normalizzazione basata sul numero di parole (per token feats) o caratteri (per char feats)
    # Per semplicità, dividiamo tutto per il numero di parole se presenti
    if len(words) > 0 and config.get("normalize", True):
        feats = {k: v / len(words) for k, v in feats.items()}
        
    return feats

def get_configurations():
    """Genera configurazioni incrementali per ogni tipo di n-gramma."""
    configs = []
    # Singoli tipi (Char, Word, Lemma, POS) incrementali da 1 a 6 (Char da 2)
    for pfx, key in [("Char", "char_ngrams"), ("Word", "word_ngrams"), 
                     ("Lemma", "lemma_ngrams"), ("POS", "pos_ngrams")]:
        start = 2 if pfx == "Char" else 1
        for n in range(start, 7):
            cfg = {"name": f"{pfx}_1-{n}", key: list(range(start, n + 1)), "normalize": True, "min_docs": 2}
            configs.append(cfg)
    
    # Configurazioni miste
    configs.extend([
        {"name": "Word_Char", "word_ngrams": [1, 2], "char_ngrams": [2, 3, 4], "normalize": True, "min_docs": 2},
        {"name": "Lemma_POS", "lemma_ngrams": [1, 2], "pos_ngrams": [1, 2, 3], "normalize": True, "min_docs": 2},
        {"name": "All_Features", "char_ngrams": [2, 3, 4], "word_ngrams": [1, 2], "lemma_ngrams": [1, 2], "pos_ngrams": [1, 2, 3], "normalize": True, "min_docs": 3},
    ])
    return configs

def run(base_path):
    print("=" * 70)
    print("TASK 3: SVM + N-GRAMMI")
    print("=" * 70)

    data_raw = utils.load_dataset_flat(base_path)
    if not data_raw["training"]:
        print("ERRORE: Dataset non caricato correttamente.")
        return

    configs = get_configurations()
    results = []
    print(f"\n[1/3] Validazione {len(configs)} configurazioni (Accuracy)...")
    
    # Pre-processiamo i documenti una volta sola per velocizzare? No, facciamo on-the-fly per semplicità
    # ma salviamo i risultati intermedi se necessario.
    
    for i, cfg in enumerate(configs, 1):
        print(f"  [{i}/{len(configs)}] {cfg['name']}...", end=" ", flush=True)
        try:
            X_tr_feats = [extract_features(t, cfg) for t, a in data_raw["training"]]
            y_tr = np.array([a for t, a in data_raw["training"]])
            
            vec = DictVectorizer()
            X_tr = vec.fit_transform(X_tr_feats)
            
            if X_tr.shape[1] == 0:
                print("Saltato (0 features)")
                continue
                
            scl = MaxAbsScaler()
            X_tr = scl.fit_transform(X_tr)
            
            svm = LinearSVC(dual=False, max_iter=5000, class_weight="balanced", random_state=42).fit(X_tr, y_tr)
            
            X_ev_feats = [extract_features(t, cfg) for t, a in data_raw["eval"]]
            y_ev = np.array([a for t, a in data_raw["eval"]])
            X_ev = scl.transform(vec.transform(X_ev_feats))
            
            score = accuracy_score(y_ev, svm.predict(X_ev))
            print(f"Accuracy: {score:.4f}")
            results.append({"name": cfg["name"], "acc": score, "cfg": cfg})
        except Exception as e:
            print(f"ERRORE: {e}")

    results.sort(key=lambda x: x["acc"], reverse=True)
    best = results[0]
    print(f"\nMigliore configurazione: {best['name']} (Acc={best['acc']:.4f})")

    print("\n[2/3] Retraining finale su Training + Eval set...")
    full_data = data_raw["training"] + data_raw["eval"]
    X_full_feats = [extract_features(t, best["cfg"]) for t, a in full_data]
    y_full = np.array([a for t, a in full_data])
    
    final_vec = DictVectorizer()
    X_full = final_vec.fit_transform(X_full_feats)
    final_scl = MaxAbsScaler()
    X_full = final_scl.fit_transform(X_full)
    
    final_svm = LinearSVC(dual=False, max_iter=5000, class_weight="balanced", random_state=42)
    final_svm.fit(X_full, y_full)

    print("\n[3/3] Valutazione finale sul TEST SET...")
    X_te_feats = [extract_features(t, best["cfg"]) for t, a in data_raw["test"]]
    y_te = np.array([a for t, a in data_raw["test"]])
    X_te = final_scl.transform(final_vec.transform(X_te_feats))
    
    test_pred = final_svm.predict(X_te)
    utils.print_evaluation(y_te, test_pred, labels=final_svm.classes_, title="RISULTATI TEST SET")

    try:
        import utils_plot
        utils_plot.set_style()
        plot_data = [{"name": r["name"], "f1_macro": r["acc"]} for r in results] # Usiamo acc come proxy
        utils_plot.plot_model_comparison(plot_data, "f1_macro", "Task 3 - Model Comparison", "03_model_comparison.png")
        utils_plot.plot_confusion_matrix(y_te, test_pred, final_svm.classes_, "Task 3 Confusion Matrix", "03_confusion_matrix.png")
    except:
        pass
def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run(os.path.join(base_dir, "dataset_authorship_finale"))

if __name__ == "__main__":
    main()
