# utils/data_quality.py

import pandas as pd


# ============================================================
# COVERAGE
# ============================================================


def compute_column_coverage(
    df: pd.DataFrame
):

    return pd.DataFrame(
        {
            "column": df.columns,
            "coverage": (
                1 -
                df.isna().mean()
            )
        }
    ).sort_values(
        "coverage"
    )


# ============================================================
# CONFIDENCE
# ============================================================


def compute_confidence(
    used_rows: pd.DataFrame
):

    fields = [

        "plot",

        "actors",

        "director",

        "imdb_rating",

        "imdb_votes"
    ]

    available = [
        c
        for c in fields
        if c in used_rows.columns
    ]

    if not available:
        return 0

    coverage = (
        1 -
        used_rows[available]
        .isna()
        .mean()
        .mean()
    )

    return int(
        coverage * 100
    )


# ============================================================
# INTEGRITY REPORT
# ============================================================


def generate_integrity_report(
    used_rows: pd.DataFrame
):

    important_fields = [

        "plot",

        "director",

        "actors",

        "genre",

        "imdb_rating"
    ]

    issues = []

    for field in important_fields:

        if field not in used_rows.columns:

            issues.append(
                f"Missing column: {field}"
            )

            continue

        missing = (
            used_rows[field]
            .isna()
            .sum()
        )

        if missing:

            issues.append(
                f"{field}: {missing} missing values"
            )

    if not issues:

        return (
            "No major data integrity "
            "issues detected."
        )

    return " | ".join(
        issues
    )


# ============================================================
# DATASET SUMMARY
# ============================================================


def dataset_summary(
    df: pd.DataFrame
):

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(
            df.isna().sum().sum()
        )
    }