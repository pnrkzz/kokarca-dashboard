"""Read-only province-year multilayer STC API (2017–2025)."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/stc", tags=["STC multilayer"])
CSV_NAME = "hhalys_hazelnut_event_climate_intervention_province_year_V22.csv"
START_YEAR, END_YEAR = 2017, 2025


def _csv_path() -> Path:
    override = os.getenv("STC_MULTILAYER_CSV")
    return Path(override) if override else Path(__file__).resolve().parent / "data" / CSV_NAME


def _number(value: str | None) -> int | float | None:
    if value is None or not value.strip():
        return None
    try:
        num = float(value)
    except ValueError:
        return None
    if not math.isfinite(num):
        return None
    return int(num) if num.is_integer() else num


def _rows() -> list[dict[str, Any]]:
    path = _csv_path()
    if not path.is_file():
        raise HTTPException(status_code=503, detail=f"STC ortak veri dosyası bulunamadı: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"province_std", "analysis_year_final", "event_count", "textual_climate_relation_count", "intervention_count"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise HTTPException(status_code=422, detail="STC CSV gerekli sütunları içermiyor")
        result = []
        seen = set()
        for raw in reader:
            try:
                year = int(float(raw["analysis_year_final"]))
            except (ValueError, TypeError):
                continue
            if not START_YEAR <= year <= END_YEAR:
                continue
            province = (raw.get("province_std") or "").strip()
            if not province:
                continue
            key = (province, year)
            if key in seen:
                raise HTTPException(status_code=422, detail=f"Tekrarlanan il-yıl hücresi: {province} {year}")
            seen.add(key)
            item = {k: (_number(v) if k not in ("province_std", "province_key") else v) for k, v in raw.items()}
            item["province"] = province
            item["year"] = year
            result.append(item)
    return sorted(result, key=lambda r: (r["province"], r["year"]))


@router.get("/multilayer")
def multilayer(
    province: str | None = Query(default=None),
    year: int | None = Query(default=None, ge=START_YEAR, le=END_YEAR),
) -> dict[str, Any]:
    all_rows = _rows()
    filtered = [r for r in all_rows if (province is None or r["province"].casefold() == province.casefold()) and (year is None or r["year"] == year)]
    return {
        "period": {"start_year": START_YEAR, "end_year": END_YEAR},
        "scope": "Fındık odaklı il-yıl analiz tablosu; tüm H. halys olaylarının toplamı değildir",
        "unit": "province-year",
        "metrics": {
            "events": "event_count",
            "climate_narratives": "textual_climate_relation_count",
            "interventions": "intervention_count",
            "temperature_anomaly": "annual_temperature_anomaly_c",
            "precipitation_anomaly": "annual_precip_anomaly_mm",
        },
        "count": len(filtered),
        "rows": filtered,
    }
