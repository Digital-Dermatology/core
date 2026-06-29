"""Shared helpers for ISIC-archive metadata exports.

The modern ISIC archive metadata export (``isic metadata download``) uses
``isic_id`` for the image, a hierarchical ``diagnosis_1..5`` taxonomy, and split
``anatom_site_*`` columns. SkinMap loaders expect a flat ``image`` /
``diagnosis`` / ``anatom_site_general`` schema (cf. the MSKCC and HIBA loaders).
This module normalizes the former into the latter so a loader can read a freshly
downloaded ISIC CSV directly, while leaving already-flat CSVs untouched.
"""

from typing import Sequence

import pandas as pd

ISIC_SITE_COLS = ["anatom_site_special", "anatom_site_1", "anatom_site_2", "anatom_site_3"]


def _first_nonempty(row: pd.Series, cols: Sequence[str]):
    for c in cols:
        if c in row.index:
            v = row[c]
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
    return None


def _combine_nonempty(row: pd.Series, cols: Sequence[str]):
    vals = []
    for c in cols:
        if c in row.index:
            v = row[c]
            if pd.notna(v) and str(v).strip():
                s = str(v).strip()
                if s not in vals:
                    vals.append(s)
    return ", ".join(vals) if vals else None


def normalize_isic_metadata(
    meta: pd.DataFrame,
    image_ext: str = ".jpg",
    diagnosis_max_level: int = 3,
) -> pd.DataFrame:
    """Map a raw ISIC-archive export to SkinMap's flat loader schema.

    Adds, only when missing:

    * ``image`` — ``<isic_id><image_ext>`` (downloaded ISIC files are
      ``ISIC_XXXXXXX.jpg``);
    * ``diagnosis`` — the deepest populated label among ``diagnosis_1`` ..
      ``diagnosis_{diagnosis_max_level}`` (disease-name level by default, which
      keeps the condition string ICD-mappable and consistent with the rest of
      SkinMap rather than the very fine ``diagnosis_4/5`` sub-variants);
    * ``anatom_site_general`` — the populated ``anatom_site_*`` values joined
      into one string so body-region harmonization sees every site signal
      (e.g. ``acral palms or soles, Lower extremity, Foot`` → ``acral``).

    Already-flat columns are preserved, so pre-normalized MSKCC/HIBA-style CSVs
    pass through unchanged.
    """
    meta = meta.copy()

    if "image" not in meta.columns and "isic_id" in meta.columns:
        meta["image"] = meta["isic_id"].astype(str) + image_ext

    if "diagnosis" not in meta.columns:
        diag_cols = [
            f"diagnosis_{i}"
            for i in range(diagnosis_max_level, 0, -1)
            if f"diagnosis_{i}" in meta.columns
        ]
        if diag_cols:
            meta["diagnosis"] = meta.apply(
                lambda r: _first_nonempty(r, diag_cols), axis=1
            )

    if "anatom_site_general" not in meta.columns:
        site_cols = [c for c in ISIC_SITE_COLS if c in meta.columns]
        if site_cols:
            meta["anatom_site_general"] = meta.apply(
                lambda r: _combine_nonempty(r, site_cols), axis=1
            )

    return meta
