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
            with open(file_path, "r", encoding="utf-8-sig") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError) as e:
            print(f"ATTENZIONE: file saltato {file_path} ({e})")
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
    paragraph = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"
        r"\u00ad\u200b-\u200f\u2028-\u2029\ufeff\ufffd]",
        "",
        paragraph,
    )
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
        with open(file_path, "r", encoding="utf-8-sig") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError) as e:
        print(f"ATTENZIONE: file saltato {file_path} ({e})")
        return []

    return paragraphs_from_text(text, min_words, max_words)


def write_paragraphs(paragraphs, output_dir, split, author):
    """Scrive i paragrafi con naming convention SPLIT___AUTORE___ID.txt"""
    author_tag = author.replace(" ", "_")
    os.makedirs(output_dir, exist_ok=True)
    for i, paragraph in enumerate(paragraphs, 1):
        name = f"{split}___{author_tag}___{i:05d}.txt"
        out_path = os.path.join(output_dir, name)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(paragraph)


def build_dataset(training_dir, test_val_dir, output_dir, min_words, max_words, test_size):
    """Genera dataset flat: tutti i file in una sola directory.

    Nomi file: {split}___{autore}___{indice}.txt
    Il nome del file codifica split e autore, senza bisogno di sottocartelle.
    """
    reset_dir(output_dir)

    stats = {"training": {}, "test": {}, "eval": {}}

    for author in AUTHORS:
        author_tag = author.replace(" ", "_")
        name = f"{author_tag}_tutti_i_libri.txt"

        # Paragrafi di training
        train_path = os.path.join(training_dir, name)
        paragraphs = paragraphs_from_file(train_path, min_words, max_words)
        stats["training"][author] = len(paragraphs)
        if paragraphs:
            write_paragraphs(paragraphs, output_dir, "training", author)

        # Paragrafi di test/eval
        test_val_path = os.path.join(test_val_dir, name)
        tv_paragraphs = paragraphs_from_file(test_val_path, min_words, max_words)

        if len(tv_paragraphs) >= 2:
            par_test, par_eval = train_test_split(tv_paragraphs, test_size=test_size, random_state=42)
        else:
            par_test, par_eval = tv_paragraphs, []

        stats["test"][author] = len(par_test)
        stats["eval"][author] = len(par_eval)

        if par_test:
            write_paragraphs(par_test, output_dir, "test", author)
        if par_eval:
            write_paragraphs(par_eval, output_dir, "eval", author)

    return {"output_dir": output_dir, "stats": stats}


def load_dataset(dataset_dir):
    """Carica il dataset flat. File nella forma: {split}___{autore}___{indice}.txt

    Returns: {"training": [(testo, autore), ...], "test": [...], "eval": [...]}
    """
    data = {"training": [], "test": [], "eval": []}
    pattern = re.compile(r"^(training|test|eval)___(.+)___\d+\.txt$")
    split_map = {"training": "training", "test": "test", "eval": "eval"}

    for filename in sorted(os.listdir(dataset_dir)):
        m = pattern.match(filename)
        if not m:
            continue
        split_key = split_map[m.group(1)]
        author = m.group(2)

        with open(os.path.join(dataset_dir, filename), "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            data[split_key].append((text, author))

    return data


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
    )
    print(f"\nStep 2: dataset creato -> {result['output_dir']}")
    for split_name, author_counts in result["stats"].items():
        total = sum(author_counts.values())
        print(f"  {split_name}: {total} paragrafi")


if __name__ == "__main__":
    main()
