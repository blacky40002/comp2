import os
import random
import re
from sklearn.model_selection import train_test_split

AUTHORS = ["primo autore", "secondo autore", "terzo autore"]
MIN_WORDS = 50
MAX_WORDS = 100
TEST_VAL_RATIO = 0.3
TEST_SIZE = 0.5


def reset_dir(path):
    if os.path.exists(path):
        import shutil

        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def extract_gutenberg_content(text):
    pattern_start = r"\*\*\* START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*"
    pattern_end = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK "

    match_start = re.search(pattern_start, text, re.IGNORECASE | re.DOTALL)
    match_end = re.search(pattern_end, text, re.IGNORECASE | re.DOTALL)

    if match_start and match_end:
        return text[match_start.end() : match_end.start()].strip()
    if match_start:
        return text[match_start.end() :].strip()
    if match_end:
        return text[: match_end.start()].strip()
    return text


def read_books(author_dir, filenames):
    books = []
    for filename in filenames:
        file_path = os.path.join(author_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue

        content = extract_gutenberg_content(text)
        if len(content) > 100:
            books.append(content)

    return books


def write_combined(books, output_path):
    if not books:
        return
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n\n" + "=" * 80 + "\n\n".join(books))


def split_books(input_root, output_training_dir, output_test_val_dir, ratio, seed):
    reset_dir(output_training_dir)
    reset_dir(output_test_val_dir)

    rng = random.Random(seed)
    stats = {}

    for author in AUTHORS:
        author_dir = os.path.join(input_root, author)
        try:
            txt_files = [f for f in os.listdir(author_dir) if f.endswith(".txt")]
        except (FileNotFoundError, OSError):
            stats[author] = {"training": 0, "test_val": 0}
            continue

        txt_files.sort()
        rng.shuffle(txt_files)

        if len(txt_files) >= 2:
            test_val_count = max(1, int(round(len(txt_files) * ratio)))
        else:
            test_val_count = 0

        test_val_files = txt_files[:test_val_count]
        training_files = txt_files[test_val_count:]

        train_books = read_books(author_dir, training_files)
        test_books = read_books(author_dir, test_val_files)

        base_name = f"{author.replace(' ', '_')}_tutti_i_libri.txt"
        write_combined(train_books, os.path.join(output_training_dir, base_name))
        write_combined(test_books, os.path.join(output_test_val_dir, base_name))

        stats[author] = {
            "training": len(training_files),
            "test_val": len(test_val_files),
        }

    return stats


def count_words(text):
    if not text or not text.strip():
        return 0
    text_clean = re.sub(r"\s+", " ", text.strip())
    return len(text_clean.split(" "))


def clean_paragraph(paragraph):
    if not paragraph:
        return ""
    paragraph = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", paragraph)
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    if re.match(r"^[^\w]*$", paragraph) or re.match(r"^\d+\.?\s*$", paragraph):
        return ""
    return paragraph


def paragraphs_from_text(text, min_words, max_words):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paragraphs = re.split(r"\n\s*\n", text)
    valid = []

    for raw in raw_paragraphs:
        paragraph = clean_paragraph(raw)
        if not paragraph:
            continue
        num_words = count_words(paragraph)
        if min_words <= num_words <= max_words:
            valid.append(paragraph)

    return valid


def paragraphs_from_file(file_path, min_words, max_words):
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return []

    return paragraphs_from_text(text, min_words, max_words)


def write_paragraphs(paragraphs, output_dir, prefix="par"):
    os.makedirs(output_dir, exist_ok=True)
    for i, paragraph in enumerate(paragraphs, 1):
        name = f"{prefix}-{i:03d}.txt"
        out_path = os.path.join(output_dir, name)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(paragraph)


def create_word_embedding_corpus(training_dir, test_val_dir, output_dir, min_words, max_words):
    embedding_dir = os.path.join(output_dir, "word_embedding_corpus")
    os.makedirs(embedding_dir, exist_ok=True)

    all_paragraphs = []
    by_author = {author: [] for author in AUTHORS}

    for source_dir in (training_dir, test_val_dir):
        for author in AUTHORS:
            name = f"{author.replace(' ', '_')}_tutti_i_libri.txt"
            file_path = os.path.join(source_dir, name)
            if not os.path.exists(file_path):
                continue

            paragraphs = paragraphs_from_file(file_path, min_words, max_words)
            all_paragraphs.extend(paragraphs)
            by_author[author].extend(paragraphs)

    full_corpus = os.path.join(embedding_dir, "corpus_completo.txt")
    with open(full_corpus, "w", encoding="utf-8") as handle:
        for paragraph in all_paragraphs:
            handle.write(paragraph.replace("\n", " ") + "\n")

    for author, paragraphs in by_author.items():
        author_name = f"corpus_{author.replace(' ', '_')}.txt"
        out_path = os.path.join(embedding_dir, author_name)
        with open(out_path, "w", encoding="utf-8") as handle:
            for paragraph in paragraphs:
                handle.write(paragraph.replace("\n", " ") + "\n")

    all_words = []
    for paragraph in all_paragraphs:
        all_words.extend(re.findall(r"\b\w+\b", paragraph.lower()))

    stats_path = os.path.join(embedding_dir, "corpus_statistics.txt")
    with open(stats_path, "w", encoding="utf-8") as handle:
        handle.write("STATISTICHE CORPUS WORD EMBEDDING\n")
        handle.write("=" * 37 + "\n\n")
        handle.write(f"Parametri estrazione: {min_words}-{max_words} parole/paragrafo\n\n")
        handle.write("TOTALI:\n")
        handle.write(f"  - Paragrafi: {len(all_paragraphs):,}\n")
        handle.write(f"  - Parole totali: {len(all_words):,}\n")
        handle.write(f"  - Vocabulary unico: {len(set(all_words)):,}\n\n")
        handle.write("PER AUTORE:\n")
        for author, paragraphs in by_author.items():
            handle.write(f"  - {author}: {len(paragraphs):,} paragrafi\n")

    return embedding_dir


def build_dataset(training_dir, test_val_dir, output_dir, min_words, max_words, test_size, create_embeddings):
    reset_dir(output_dir)

    if create_embeddings:
        create_word_embedding_corpus(training_dir, test_val_dir, output_dir, min_words, max_words)

    for split in ("training_set", "test_set", "eval_set"):
        for author in AUTHORS:
            os.makedirs(os.path.join(output_dir, split, author), exist_ok=True)

    training_stats = {}
    test_eval_stats = {}

    for author in AUTHORS:
        name = f"{author.replace(' ', '_')}_tutti_i_libri.txt"
        train_path = os.path.join(training_dir, name)
        paragraphs = paragraphs_from_file(train_path, min_words, max_words)
        training_stats[author] = len(paragraphs)
        if paragraphs:
            write_paragraphs(paragraphs, os.path.join(output_dir, "training_set", author), "par")

    for author in AUTHORS:
        name = f"{author.replace(' ', '_')}_tutti_i_libri.txt"
        test_val_path = os.path.join(test_val_dir, name)
        paragraphs = paragraphs_from_file(test_val_path, min_words, max_words)

        if len(paragraphs) >= 2:
            par_test, par_eval = train_test_split(paragraphs, test_size=test_size, random_state=42)
        else:
            par_test, par_eval = paragraphs, []

        test_eval_stats[author] = {
            "totale": len(paragraphs),
            "test": len(par_test),
            "eval": len(par_eval),
        }

        if par_test:
            write_paragraphs(par_test, os.path.join(output_dir, "test_set", author), "par")
        if par_eval:
            write_paragraphs(par_eval, os.path.join(output_dir, "eval_set", author), "par")

    return {
        "output_dir": output_dir,
        "training_stats": training_stats,
        "test_eval_stats": test_eval_stats,
    }


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_books = os.path.join(base_dir, "libri_training")
    output_training = os.path.join(base_dir, "libri_puliti_training")
    output_test_val = os.path.join(base_dir, "libri_test_val_puliti")
    output_dataset = os.path.join(base_dir, "dataset_authorship_finale")

    split_stats = split_books(
        input_books,
        output_training,
        output_test_val,
        ratio=TEST_VAL_RATIO,
        seed=42,
    )
    print("\nSTEP 1: SPLIT LIBRI (TRAINING vs TEST/VAL)")
    print(f"Percorso input libri: {input_books}")
    print(f"Output training: {output_training}")
    print(f"Output test/val: {output_test_val}")
    print(f"Ratio test/val: {TEST_VAL_RATIO:.2f}\n")
    for author, stats in split_stats.items():
        total = stats["training"] + stats["test_val"]
        ratio = stats["test_val"] / total if total > 0 else 0
        print(
            f"- {author}: {stats['training']} training, {stats['test_val']} test/val "
            f"(totale {total}, ratio effettivo {ratio:.2f})"
        )

    result = build_dataset(
        training_dir=output_training,
        test_val_dir=output_test_val,
        output_dir=output_dataset,
        min_words=MIN_WORDS,
        max_words=MAX_WORDS,
        test_size=TEST_SIZE,
        create_embeddings=True,
    )
    print(f"Step 2: dataset creato -> {result['output_dir']}")

    training_total = sum(result["training_stats"].values())
    test_total = sum(stats["test"] for stats in result["test_eval_stats"].values())
    eval_total = sum(stats["eval"] for stats in result["test_eval_stats"].values())
    print(f"Finale: training={training_total}, test={test_total}, eval={eval_total}")


if __name__ == "__main__":
    main()
