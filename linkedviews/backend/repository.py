from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils import first_column, norm, parse_multi


class DataRepository:
    """
    Tek doğruluk kaynağı.
    CSV ve GeoJSON yalnızca bir kez okunur.
    Servisler aynı repository nesnesini kullanır.
    """

    def __init__(self, events_path: Path, geojson_path: Path) -> None:
        if not events_path.exists():
            raise FileNotFoundError(f"Olay CSV bulunamadı: {events_path}")
        if not geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON bulunamadı: {geojson_path}")

        raw = pd.read_csv(events_path, low_memory=False)
        self.geojson = json.loads(
            geojson_path.read_text(encoding="utf-8-sig")
        )

        # Base columns
        self.province_col = first_column(
            raw, ["province", "province_name", "il", "event_province"], True
        )
        self.year_col = first_column(
            raw,
            [
                "event_year",
                "validated_spread_observation_year",
                "spread_observation_year",
                "validated_year",
                "spread_year",
                "year",
            ],
            True,
        )
        self.product_col = first_column(
            raw,
            ["crops_raw", "crops", "crop", "products", "product", "urun", "ürün"],
        )
        self.url_col = first_column(
            raw, ["url", "source_url", "document_url", "video_url", "link"]
        )
        self.title_col = first_column(
            raw, ["title", "document_title", "source_title", "headline"]
        )
        self.source_col = first_column(
            raw, ["source_type", "source_category", "document_type", "source"]
        )
        self.doc_col = first_column(
            raw,
            ["global_id", "document_id", "doc_id", "raw_document_id", "event_id", "id"],
        )
        self.published_date_col = first_column(
            raw, ["published_date", "publication_date", "date"]
        )

        # Intervention columns
        self.intervention_year_col = first_column(
            raw, ["intervention_year", "mudahale_yili", "müdahale_yili"]
        )
        self.intervention_type_col = first_column(
            raw,
            ["intervention_type_raw", "intervention_type", "mudahale_turu", "müdahale_turu"],
        )
        self.institution_col = first_column(
            raw, ["institution_raw", "institution", "kurum"]
        )
        self.action_col = first_column(
            raw, ["government_action_raw", "government_action", "action"]
        )
        self.actor_col = first_column(
            raw, ["intervention_actor_raw", "intervention_actor", "actor"]
        )
        self.effect_col = first_column(
            raw, ["intervention_effect_on_spread", "intervention_effect", "effect"]
        )
        self.has_intervention_col = first_column(
            raw, ["has_government_intervention", "has_intervention"]
        )

        # Climate columns
        self.climate_mentioned_col = first_column(
            raw, ["mentions_climate_relation", "climate_relation_mentioned"]
        )
        self.climate_factor_col = first_column(
            raw, ["climate_factor_type_raw", "climate_factor"]
        )
        self.climate_signal_col = first_column(
            raw, ["climate_signal_raw", "climate_signal"]
        )
        self.climate_direction_col = first_column(
            raw, ["climate_relation_direction"]
        )
        self.climate_comment_col = first_column(
            raw, ["climate_comment"]
        )

        # Normalize base dataframe
        raw[self.year_col] = pd.to_numeric(raw[self.year_col], errors="coerce")
        raw[self.province_col] = raw[self.province_col].astype(str).str.strip()

        if self.published_date_col:
            raw["_published_date"] = pd.to_datetime(
                raw[self.published_date_col],
                errors="coerce",
            )
            raw["_published_year"] = raw["_published_date"].dt.year
        else:
            raw["_published_date"] = pd.NaT
            raw["_published_year"] = pd.NA

        if self.intervention_year_col:
            raw["_intervention_year"] = pd.to_numeric(
                raw[self.intervention_year_col], errors="coerce"
            )
        else:
            raw["_intervention_year"] = pd.NA

        if self.product_col:
            raw["_products"] = raw[self.product_col].apply(
                lambda v: parse_multi(v, split_commas=True)
            )
        else:
            raw["_products"] = [[] for _ in range(len(raw))]

        self.df_all = raw.copy()
        self.df_temporal = self.df_all[self.df_all[self.year_col].notna()].copy()
        self.df_temporal[self.year_col] = self.df_temporal[self.year_col].astype(int)

        self.geo_name_key = self._detect_geo_name_key()
        self.geo_lookup = {}
        for feature in self.geojson.get("features", []):
            value = (feature.get("properties") or {}).get(self.geo_name_key)
            if value not in (None, ""):
                self.geo_lookup[norm(value)] = str(value).strip()

        for frame in (self.df_all, self.df_temporal):
            frame["_province"] = frame[self.province_col].map(
                lambda v: self.geo_lookup.get(norm(v), str(v).strip())
            )

        self.base_geojson = self._prepare_base_geojson()

    def _detect_geo_name_key(self) -> str:
        csv_names = {
            norm(v)
            for v in self.df_all[self.province_col].dropna().unique()
        }
        features = self.geojson.get("features", [])
        keys: set[str] = set()

        for feature in features[:100]:
            keys.update((feature.get("properties") or {}).keys())

        scored = []
        for key in keys:
            values = {
                norm((feature.get("properties") or {}).get(key))
                for feature in features
            }
            scored.append((len(csv_names & values), key))

        scored.sort(reverse=True)
        if not scored or scored[0][0] <= 0:
            raise ValueError("GeoJSON il adı alanı otomatik belirlenemedi.")
        return scored[0][1]

    def _prepare_base_geojson(self) -> dict[str, Any]:
        output = copy.deepcopy(self.geojson)
        for feature in output.get("features", []):
            props = feature.setdefault("properties", {})
            province = str(props.get(self.geo_name_key, "")).strip()
            props.clear()
            props["province_name"] = province
        return output
