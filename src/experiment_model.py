"""
experiment_model.py - Compare classifier configurations fairly.

Each configuration is scored with stratified 5-fold cross-validation on the
same data, so a result is an average over 5 different train/test splits rather
than one lucky (or unlucky) split. The spread (std) is shown too: differences
smaller than the std are noise, not real improvements.

Usage (from the project root):
    python src/experiment_model.py
    python src/experiment_model.py --country us

Needs: scikit-learn (already installed if update_all.py works).
Results are also saved to data/processed/model_experiments.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import PROCESSED_DIR  # noqa: E402
from update_all import MIN_SAMPLES_PER_CLASS, load_jobs  # noqa: E402

OUTPUT_FILE = PROCESSED_DIR / "model_experiments.csv"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning as update_all.py: no repeated ads, no tiny classes."""
    data = df.dropna(subset=["description", "search_keyword"]).copy()
    data = data.drop_duplicates(subset=["company", "title", "description"])

    counts = data["search_keyword"].value_counts()
    data = data[data["search_keyword"].isin(counts[counts >= MIN_SAMPLES_PER_CLASS].index)]

    data["title"] = data["title"].fillna("")
    data["text_description"] = data["description"]
    data["text_title_description"] = data["title"] + " . " + data["description"]
    return data


def build_configs():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    def pipe(features, ngram, clf, sublinear=False):
        return make_pipeline(
            TfidfVectorizer(
                max_features=features,
                stop_words="english",
                ngram_range=ngram,
                sublinear_tf=sublinear,
            ),
            clf,
        )

    D, TD = "text_description", "text_title_description"

    return [
        # name, text column, pipeline
        ("A. baseline (current): 500 feat, LR",
         D, pipe(500, (1, 2), LogisticRegression(max_iter=1000))),
        ("B. 2000 feat, LR",
         D, pipe(2000, (1, 2), LogisticRegression(max_iter=1000))),
        ("C. 2000 feat, LR, balanced classes",
         D, pipe(2000, (1, 2), LogisticRegression(max_iter=1000, class_weight="balanced"))),
        ("D. 5000 feat, LR C=10, balanced, sublinear tf",
         D, pipe(5000, (1, 2), LogisticRegression(max_iter=2000, C=10, class_weight="balanced"),
                 sublinear=True)),
        ("E. 5000 feat, LinearSVC, balanced, sublinear tf",
         D, pipe(5000, (1, 2), LinearSVC(C=0.5, class_weight="balanced"), sublinear=True)),
        ("F. title + description, 2000 feat, LR, balanced",
         TD, pipe(2000, (1, 2), LogisticRegression(max_iter=1000, class_weight="balanced"))),
        ("G. title + description, 5000 feat, LinearSVC, balanced",
         TD, pipe(5000, (1, 2), LinearSVC(C=0.5, class_weight="balanced"), sublinear=True)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare classifier configurations.")
    parser.add_argument("--country", default="gb")
    args = parser.parse_args()

    from sklearn.model_selection import StratifiedKFold, cross_validate

    data = prepare(load_jobs(args.country))
    y = data["search_keyword"]
    print(f"Using {len(data):,} unique postings, {y.nunique()} classes.\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, column, model in build_configs():
        started = time.time()
        scores = cross_validate(
            model, data[column], y, cv=cv,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        )
        row = {
            "config": name,
            "accuracy": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "macro_f1": scores["test_macro_f1"].mean(),
            "seconds": time.time() - started,
        }
        results.append(row)
        print(f"{name:<58} acc {row['accuracy']:.2%} (+/- {row['accuracy_std']:.2%})  "
              f"macro-F1 {row['macro_f1']:.3f}  [{row['seconds']:.0f}s]")

    table = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    table.round(4).to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("Ranking by macro-F1 (treats every job family equally):")
    print("=" * 70)
    for rank, row in enumerate(table.itertuples(index=False), start=1):
        print(f"{rank}. {row.config}\n   accuracy {row.accuracy:.2%} (+/- {row.accuracy_std:.2%}), "
              f"macro-F1 {row.macro_f1:.3f}")

    print(f"\nSaved to {OUTPUT_FILE}")
    print("\nNote: configs F and G also use the job TITLE. The title usually contains the "
          "very words used to find the job, so those scores answer an easier question "
          "than 'classify from the description alone'.")


if __name__ == "__main__":
    main()
