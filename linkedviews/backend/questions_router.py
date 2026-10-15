"""Research questions API: source-grounded climate evidence."""
from pathlib import Path
import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/questions", tags=["research-questions"])

BASE = Path(os.getenv(
    "KOKARCA_PROCESSED_DIR",
    r"C:\Users\depoc\kokarca_project\data_processed"
))

EVIDENCE_PATH = BASE / (
    "STC_IKLIM/02_IKLIM/FINAL_CLIMATE_DATA/"
    "FINAL_INTEGRATED_CLIMATE_MATRIX/DATA/"
    "H_HALYS_FINAL_INTEGRATED_CLIMATE_EVIDENCE_MATRIX_V1.csv"
)

CORPUS_PATH = BASE / (
    "official_corpus/master_corpus/"
    "kokarca_master_corpus_v3_publication_dates_final.csv"
)

_cache = None


def load_evidence():
    global _cache

    if _cache is not None:
        return _cache

    for path in (EVIDENCE_PATH, CORPUS_PATH):
        if not path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"Missing source file: {path}"
            )

    evidence = pd.read_csv(EVIDENCE_PATH, low_memory=False)

    metadata_columns = [
        "global_id",
        "source_type",
        "source_domain",
        "title",
        "url",
        "published_date",
        "publication_date_precision",
        "publication_date_verification",
        "date_is_synthetic",
    ]

    corpus = pd.read_csv(
        CORPUS_PATH,
        usecols=metadata_columns,
        low_memory=False
    )

    evidence["global_id"] = (
        evidence["global_id"].astype("string").str.strip()
    )
    corpus["global_id"] = (
        corpus["global_id"].astype("string").str.strip()
    )

    if corpus["global_id"].duplicated().any():
        raise HTTPException(
            status_code=500,
            detail="Duplicate document IDs in master corpus"
        )

    # Keep document metadata distinct from relation fields.
    metadata = corpus.rename(columns={
        col: f"document_{col}"
        for col in metadata_columns
        if col != "global_id"
    })

    merged = evidence.merge(
        metadata,
        on="global_id",
        how="left",
        validate="many_to_one",
        indicator="document_match"
    )

    _cache = merged
    return merged


def clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


@router.get("/evidence")
def get_evidence(
    province: str | None = None,
    source_type: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    df = load_evidence()

    if province:
        df = df[
            df["province"].astype("string").str.casefold()
            == province.casefold()
        ]

    if source_type:
        df = df[
            df["document_source_type"].astype("string")
            == source_type
        ]

    total = len(df)
    page = df.iloc[offset:offset + limit]

    items = []

    for _, row in page.iterrows():
        items.append({
            "global_id": clean(row.get("global_id")),
            "event_order": clean(row.get("event_order")),
            "relation_order": clean(row.get("relation_order")),
            "source_type": clean(row.get("document_source_type")),
            "source_domain": clean(row.get("document_source_domain")),
            "title": clean(row.get("document_title")),
            "url": clean(row.get("document_url")),
            "published_date": clean(row.get("document_published_date")),
            "publication_date_precision": clean(
                row.get("document_publication_date_precision")
            ),
            "publication_date_verification": clean(
                row.get("document_publication_date_verification")
            ),
            "date_is_synthetic": clean(
                row.get("document_date_is_synthetic")
            ),
            "province": clean(row.get("province")),
            "event_year": clean(row.get("year")),
            "evidence_quote": clean(row.get("evidence_quote")),
            "driver": clean(row.get("driver_code")),
            "effect": clean(row.get("effect_code")),
            "climate_change_status": clean(
                row.get("climate_change_status")
            ),
            "geographic_scope": clean(row.get("geographic_scope")),
            "era5_status": clean(row.get("era5_consistency_locked")),
            "final_relation_id": clean(row.get("final_relation_id")),
            "driver_text": clean(row.get("driver_text")),
            "driver_direction": clean(row.get("driver_direction")),
            "effect_text": clean(row.get("effect_text")),
            "effect_direction": clean(row.get("effect_direction")),
            "mechanism": clean(row.get("mechanism")),
            "relation_status": clean(row.get("relation_status")),
            "causal_strength": clean(row.get("causal_strength")),
            "climate_change_relation": clean(row.get("climate_change_relation")),
            "evidence_use_scope": clean(row.get("evidence_use_scope")),
            "country": clean(row.get("country")),
            "district": clean(row.get("district")),
            "time_expression": clean(row.get("time_expression")),
            "confidence": clean(row.get("confidence")),
            "review_required": clean(row.get("review_required")),
            "review_reason": clean(row.get("review_reason")),
            "evidence_verbatim_verified": clean(row.get("evidence_verbatim_verified")),
            "era5_testability": clean(row.get("era5_testability")),
            "era5_evidence_strength": clean(row.get("era5_evidence_strength")),
            "final_evidence_profile": clean(row.get("final_evidence_profile")),
            "final_integrated_evidence_status": clean(row.get("final_integrated_evidence_status")),
            "document_match": clean(row.get("document_match")),
        })

    return {
        "total_relations": total,
        "total_documents": int(df["global_id"].nunique()),
        "items": items,
        "limit": limit,
        "offset": offset,
        "notice": (
            "These are source claims and extracted relations, "
            "not independent proof of climate-change causality."
        )
    }

