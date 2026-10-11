"""
update_all.py - Refresh everything downstream of the collected data.
AI Job Market Intelligence Platform
-----------------------------------------------------------------

  1. Loads data/processed/jobs_master.csv (filtered to ONE country, because
     salaries in different currencies must not be mixed).
  2. Rebuilds the 4 SQL Server tables:
        companies, skills, jobs_normalized, job_skills
  3. Retrains the job-family classifier (TF-IDF + Logistic Regression) and
     saves the model, the vectorizer and data/processed/model_metrics.json.

Usage (from the project root):
    python src/update_all.py                  # everything, UK jobs
    python src/update_all.py --country us     # another country
    python src/update_all.py --skip-sql       # only retrain the model
    python src/update_all.py --skip-model     # only refresh SQL

Local-only dependencies (not needed by the GitHub Action):
    python -m pip install scikit-learn sqlalchemy pyodbc joblib

SQL connection (defaults match your setup):
    SQL_SERVER   default: localhost\\SQLEXPRESS
    DB_URL       optional full SQLAlchemy URL, overrides everything
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import MASTER_FILE, PROCESSED_DIR, SKILLS  # single source of truth

MODEL_FILE = PROCESSED_DIR / "job_classifier_model.pkl"
VECTORIZER_FILE = PROCESSED_DIR / "tfidf_vectorizer.pkl"
METRICS_FILE = PROCESSED_DIR / "model_metrics.json"

MIN_SAMPLES_PER_CLASS = 10


# ============================================================
# Load
# ============================================================

def load_jobs(country: str) -> pd.DataFrame:
    if not MASTER_FILE.exists():
        sys.exit("jobs_master.csv not found. Run `python src/collect.py` first.")

    df = pd.read_csv(MASTER_FILE, dtype={"job_id": str}, encoding="utf-8-sig")

    if "country" not in df.columns:  # very old master file
        df["country"] = "gb"
    df["country"] = df["country"].fillna("gb")

    df = df[df["country"] == country].copy()
    if df.empty:
        sys.exit(f"No jobs found for country '{country}'.")

    df = df.drop_duplicates(subset="job_id")

    if "skills" not in df.columns:
        df["skills"] = ""
    df["skills"] = df["skills"].fillna("")
    df["skill_list"] = df["skills"].apply(
        lambda s: [x.strip() for x in s.split(",") if x.strip()]
    )

    print(f"Loaded {len(df):,} jobs for country '{country}'.")
    return df


# ============================================================
# SQL
# ============================================================

def build_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    companies = pd.DataFrame({"company_name": sorted(df["company"].dropna().unique())})
    companies.insert(0, "company_id", range(1, len(companies) + 1))

    skills = pd.DataFrame({"skill_name": SKILLS})
    skills.insert(0, "skill_id", range(1, len(skills) + 1))

    jobs = df.merge(companies, left_on="company", right_on="company_name", how="left")
    wanted = [
        "job_id", "title", "company_id", "location",
        "salary_min", "salary_max", "currency", "contract_type", "category",
        "description", "search_keyword", "created", "redirect_url",
        "country", "first_seen", "last_seen",
    ]
    jobs = jobs[[c for c in wanted if c in jobs.columns]]

    links = (
        df[["job_id", "skill_list"]]
        .explode("skill_list")
        .dropna(subset=["skill_list"])
        .rename(columns={"skill_list": "skill_name"})
        .merge(skills, on="skill_name")[["job_id", "skill_id"]]
    )

    return companies, skills, jobs, links


def get_engine():
    import sqlalchemy as sa

    url = os.getenv("DB_URL")
    if not url:
        server = os.getenv("SQL_SERVER", r"localhost\SQLEXPRESS")
        url = (
            f"mssql+pyodbc://{server}/JobMarketDB"
            "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        )
    kwargs = {"fast_executemany": True} if url.startswith("mssql") else {}
    return sa.create_engine(url, **kwargs)


def update_sql(df: pd.DataFrame) -> None:
    print("\n--- Updating SQL tables ---")
    companies, skills, jobs, links = build_tables(df)
    engine = get_engine()

    tables = {
        "companies": companies,
        "skills": skills,
        "jobs_normalized": jobs,
        "job_skills": links,
    }
    try:
        for name, table in tables.items():
            table.to_sql(name, con=engine, if_exists="replace", index=False, chunksize=500)
            print(f"  {name:<16} {len(table):>7,} rows")
    except Exception as exc:  # noqa: BLE001 - show a friendly hint, keep the real error
        print("\nSQL update failed. Check that SQL Server is running and the database "
              "'JobMarketDB' exists.")
        raise exc
    print("SQL database updated.")


# ============================================================
# Model
# ============================================================

def train_model(df: pd.DataFrame, country: str) -> None:
    print("\n--- Retraining the classifier ---")
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    data = df.dropna(subset=["description", "search_keyword"])

    # The same ad reposted in many cities would appear in both train and test
    # and inflate the score, so identical postings are counted once.
    before = len(data)
    data = data.drop_duplicates(subset=["company", "title", "description"])
    removed = before - len(data)

    counts = data["search_keyword"].value_counts()
    keep = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    dropped_classes = sorted(set(counts.index) - set(keep))
    data = data[data["search_keyword"].isin(keep)]

    print(f"  Training data: {len(data):,} postings "
          f"({removed:,} repeated postings removed, {data['search_keyword'].nunique()} classes)")
    if dropped_classes:
        print(f"  Skipped classes with <{MIN_SAMPLES_PER_CLASS} samples: {dropped_classes}")

    X, y = data["description"], data["search_keyword"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"\n  Accuracy: {accuracy:.2%}  (on {len(y_test)} held-out postings)\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "country": country,
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(y.nunique()),
        "repeated_postings_removed": int(removed),
        "f1_per_class": {
            label: round(float(report[label]["f1-score"]), 4) for label in sorted(y.unique())
        },
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model, vectorizer and {METRICS_FILE.name} saved to {PROCESSED_DIR}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh SQL tables and retrain the model.")
    parser.add_argument("--country", default="gb", help="country code to use (default: gb)")
    parser.add_argument("--skip-sql", action="store_true", help="do not touch SQL Server")
    parser.add_argument("--skip-model", action="store_true", help="do not retrain the model")
    args = parser.parse_args()

    df = load_jobs(args.country)

    if not args.skip_sql:
        update_sql(df)
    if not args.skip_model:
        train_model(df, args.country)

    print("\nDone.")


if __name__ == "__main__":
    main()
