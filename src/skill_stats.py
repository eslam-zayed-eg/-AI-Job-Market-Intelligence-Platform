"""
skill_stats.py - Print skill demand and salary-by-skill from the master dataset.

Usage (from the project root):
    python src/skill_stats.py                  # UK, repeated ads counted once
    python src/skill_stats.py --country us
    python src/skill_stats.py --keep-repeats   # count every posting, even reposts

Why repeated ads are removed by default: the same advert posted in 30 cities
(e.g. a training-scheme ad with a fixed salary) would otherwise count 30 times
and pull the averages towards itself.

Salary figures use salary_min and are in the currency of the chosen country.
Saves the tables to data/processed/skill_stats_<country>.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import PROCESSED_DIR  # noqa: E402
from update_all import load_jobs  # noqa: E402

MIN_SALARY_SAMPLES = 20


def compute(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Returns (demand, pay, total_postings, postings_with_a_skill).
    `df` must contain a `skill_list` column (as produced by update_all.load_jobs).
    """
    total = len(df)

    exploded = df[["job_id", "skill_list"]].explode("skill_list").dropna(subset=["skill_list"])
    demand = (
        exploded.groupby("skill_list")["job_id"].nunique()
        .sort_values(ascending=False)
        .rename("postings")
        .to_frame()
    )
    demand["share_of_postings_%"] = (demand["postings"] / total * 100).round(1) if total else 0.0
    demand.index.name = "skill"
    with_skill = int(exploded["job_id"].nunique())

    salaries = df[["job_id", "salary_min", "skill_list"]].dropna(subset=["salary_min"])
    salaries = salaries.explode("skill_list").dropna(subset=["skill_list"])
    pay = (
        salaries.groupby("skill_list")["salary_min"]
        .agg(avg_salary_min="mean", median_salary_min="median", postings_with_salary="count")
        .round(0)
    )
    pay = pay[pay["postings_with_salary"] >= MIN_SALARY_SAMPLES].sort_values(
        "avg_salary_min", ascending=False
    )
    pay.index.name = "skill"

    return demand, pay, total, with_skill


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill demand and salary tables.")
    parser.add_argument("--country", default="gb")
    parser.add_argument("--keep-repeats", action="store_true")
    args = parser.parse_args()

    df = load_jobs(args.country)

    if not args.keep_repeats:
        before = len(df)
        df = df.drop_duplicates(subset=["company", "title", "description"])
        print(f"Repeated postings counted once: {before:,} -> {len(df):,}")

    currency = "?"
    if "currency" in df.columns and df["currency"].notna().any():
        currency = df["currency"].dropna().iloc[0]

    demand, pay, total, with_skill = compute(df)
    print(f"\n{total:,} postings | country={args.country} | salary currency={currency}")
    print(f"Postings mentioning at least one tracked skill: {with_skill:,} "
          f"({with_skill / max(total, 1):.0%})")

    print("\n=== Top 15 skills by number of postings ===")
    print(demand.head(15).to_string())

    print(f"\n=== Average salary_min by skill (only skills with >= {MIN_SALARY_SAMPLES} salaried postings) ===")
    print(pay.to_string())

    print("\n=== Postings per job family ===")
    print(df["search_keyword"].value_counts().to_string())

    out = PROCESSED_DIR / f"skill_stats_{args.country}.csv"
    demand.join(pay, how="outer").sort_values("postings", ascending=False).to_csv(out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
