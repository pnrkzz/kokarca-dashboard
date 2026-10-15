"""Additive, read-only climate relation endpoint for existing FastAPI app.
In main.py: from climate_relations_router import router as climate_relations_router
            app.include_router(climate_relations_router)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/climate", tags=["climate-relations"])
BASE = Path(os.getenv("KOKARCA_PROCESSED_DIR", r"C:\Users\depoc\kokarca_project\data_processed"))
PATHS = {
    "september": BASE / "STC_IKLIM/02_IKLIM/FINAL_CLIMATE_DATA/FINAL_INTEGRATED_CLIMATE_MATRIX/DATA/H_HALYS_FINAL_INTEGRATED_CLIMATE_EVIDENCE_MATRIX_V1.csv",
    "october": BASE / "FINDIK_IKLIM_ETKI_ANALIZI/FINDIK_IKLIM_GPT337_KAYNAKLI_ANA_TABLO.csv",
    "hazelnut": BASE / "FINDIK_IKLIM_ETKI_ANALIZI/FINDIK_IKLIM_117_BILIMSEL_SINIFLANDIRMA_HAZIRLIK_V48.csv",
}
KEYS = ["global_id", "event_order", "relation_order"]
_cache: Optional[pd.DataFrame] = None

def _load() -> pd.DataFrame:
    global _cache
    if _cache is not None:
        return _cache
    missing = [str(p) for p in PATHS.values() if not p.is_file()]
    if missing:
        raise HTTPException(status_code=503, detail="İklim veri dosyaları bulunamadı: " + "; ".join(missing))
    tables = {name: pd.read_csv(path, low_memory=False) for name, path in PATHS.items()}
    for table in tables.values():
        table["global_id"] = table["global_id"].astype("string").str.strip()
        for key in KEYS[1:]:
            table[key] = pd.to_numeric(table[key], errors="coerce").astype("Int64")
        if table[KEYS].isna().any().any() or table.duplicated(KEYS).any():
            raise HTTPException(status_code=500, detail="İlişki anahtarları eksik veya tekrarlı")

    octo = tables["october"]
    # Exclude records not explicitly eligible for H. halys analysis; retain originals on disk.
    octo = octo[octo["eligible_species_analysis"].eq(True)].copy()
    review_cols = ["source_support_standard", "evidence_nature_43", "effect_path_unified_47", "source_review_note_unified"]
    review = tables["hazelnut"][KEYS + review_cols].copy()
    octo = octo.merge(review, on=KEYS, how="left", validate="one_to_one")
    sept = tables["september"]
    # Keep analytical source namespaces distinct; shared keys are not proof of identical claims.
    combined = sept.merge(octo, on=KEYS, how="outer", suffixes=("_sep", "_oct"), validate="one_to_one", indicator="_origin")
    _cache = combined
    return combined

def _value(row, *cols):
    for col in cols:
        if col in row.index:
            v = row[col]
            if pd.notna(v) and str(v).strip() and str(v).lower() not in ("nan", "none"):
                if isinstance(v, pd.Timestamp):
                    return v.isoformat()
                if hasattr(v, "item"):
                    v = v.item()
                return v
    return None

def _bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")

def _norm(v):
    return str(v or "").strip().casefold().replace("ı", "i")

@router.get("/relations")
def climate_relations(
    province: Optional[str] = None,
    product: Optional[str] = None,
    start_year: Optional[int] = Query(None, alias="start_year"),
    end_year: Optional[int] = Query(None, alias="end_year"),
    limit: int = Query(200, ge=1, le=1000),
):
    df = _load()
    product_name = _norm(product)
    # Only hazelnut has a verified product-specific review. Never present all-species data as fig/cherry data.
    hazelnut_names = {"findik", "fındık", "hazelnut", "corylus avellana"}
    if product_name and product_name not in hazelnut_names:
        return {"total": 0, "items": [], "notice": "Bu ürün için ilişki-temelli ürün eşleştirmesi henüz doğrulanmadı."}
    if product_name:
        df = df[df["source_support_standard"].notna()].copy()
    # Province is assigned only from the existing September location field; no free-text geocoding.
    if province:
        df = df[df["province"].map(_norm) == _norm(province)].copy()
    years = pd.to_numeric(df["year"], errors="coerce")
    if "analysis_year_final" in df.columns:
        years = years.fillna(pd.to_numeric(df["analysis_year_final"], errors="coerce"))
    if start_year is not None:
        df = df[years >= start_year]
        years = years.loc[df.index]
    if end_year is not None:
        df = df[years <= end_year]
    items = []
    for _, row in df.head(limit).iterrows():
        # Preserve both types of statements rather than substituting extraction text for original quotes.
        item = {
            "id": "-".join(str(_value(row, k)) for k in KEYS),
            "global_id": _value(row, "global_id"),
            "origin": _value(row, "_origin"),
            "province": _value(row, "province"),
            "year": int(years.loc[row.name]) if pd.notna(years.loc[row.name]) else None,
            "source_type": _value(row, "source_type", "source_type_oct"),
            "title": _value(row, "title", "title_oct"),
            "url": _value(row, "url", "url_oct"),
            "published_date": _value(row, "published_date", "published_date_oct"),
            "relation_text": _value(row, "relation_text"),
            "evidence_quote": _value(row, "evidence_quote"),
            "driver": _value(row, "driver_code"),
            "effect": _value(row, "effect_code"),
            "climate_change_status": _value(row, "climate_change_status"),
            "source_support": _value(row, "source_support_standard"),
            "effect_path": _value(row, "effect_path_unified_47"),
            "review_note": _value(row, "source_review_note_unified"),
            "era5_status": _value(row, "era5_consistency_locked"),
            "era5_match": _value(row, "era5_final_match_status"),
            "climategpt_answer": _value(row, "climategpt_answer"),
            "climategpt_is_independent_validation": False,
        }
        items.append(item)
    return {"total": int(len(df)), "items": items, "notice": "Kaynakta ifade edilen ilişkiler; bağımsız nedensellik kanıtı değildir. Konumsuz veya tarihsiz kayıtlar ilgili filtreyle listelenmez."}
