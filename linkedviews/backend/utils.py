from __future__ import annotations

import ast
import re
import unicodedata
from typing import Iterable

import pandas as pd


def norm(value: object) -> str:
    text = str(value or "").strip()
    replacements = {
        "İ": "I", "ı": "i", "Ş": "S", "ş": "s", "Ğ": "G", "ğ": "g",
        "Ü": "U", "ü": "u", "Ö": "O", "ö": "o", "Ç": "C", "ç": "c",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def first_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = False,
) -> str | None:
    lookup = {norm(c): c for c in df.columns}
    for candidate in candidates:
        key = norm(candidate)
        if key in lookup:
            return lookup[key]

    if required:
        raise ValueError(f"Gerekli sütun bulunamadı: {candidates}")
    return None


def parse_multi(value: object, split_commas: bool = False) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]", "yok"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            values = list(parsed)
        else:
            values = re.split(r"[;|]+", text)
    except Exception:
        values = re.split(r"[;|]+", text)

    if split_commas and len(values) == 1 and "," in str(values[0]):
        values = [x.strip() for x in str(values[0]).split(",")]

    result: list[str] = []
    seen: set[str] = set()

    for item in values:
        cleaned = str(item).strip().strip("'\"[]")
        key = norm(cleaned)
        if key and key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


def canonical_values(values: Iterable[str]) -> list[str]:
    dedup: dict[str, str] = {}
    for value in values:
        key = norm(value)
        if key and key not in dedup:
            dedup[key] = value.strip().lower()
    return sorted(dedup.values(), key=str.lower)
