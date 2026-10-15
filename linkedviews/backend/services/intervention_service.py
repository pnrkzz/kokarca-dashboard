from __future__ import annotations

from typing import Any

import pandas as pd

from repository import DataRepository
from services.base_service import BaseService
from utils import canonical_values, norm, parse_multi


class InterventionService(BaseService):
    def __init__(self, repo: DataRepository) -> None:
        super().__init__(repo)
        self.df = self._prepare()

    def _prepare(self) -> pd.DataFrame:
        r = self.repo
        df = r.df_all.copy()

        df["_intervention_types"] = (
            df[r.intervention_type_col].apply(parse_multi)
            if r.intervention_type_col
            else [[] for _ in range(len(df))]
        )

        df["_institutions"] = (
            df[r.institution_col].apply(parse_multi)
            if r.institution_col
            else [[] for _ in range(len(df))]
        )

        df["_actions"] = (
            df[r.action_col].apply(parse_multi)
            if r.action_col
            else [[] for _ in range(len(df))]
        )

        df["_actors"] = (
            df[r.actor_col].apply(parse_multi)
            if r.actor_col
            else [[] for _ in range(len(df))]
        )

        has_info = (
            df["_intervention_types"].map(bool)
            | df["_institutions"].map(bool)
            | df["_actions"].map(bool)
            | df["_actors"].map(bool)
        )

        if r.has_intervention_col:
            raw = df[r.has_intervention_col].astype(str).str.lower().str.strip()
            has_info = has_info | raw.isin(
                {"true", "1", "yes", "evet", "var", "yes/true"}
            )

        return df[has_info].copy()

    def _filter(
        self,
        start_year: int | None = None,
        end_year: int | None = None,
        intervention_type: str | None = None,
        institution: str | None = None,
        product: str | None = None,
        province: str | None = None,
    ) -> pd.DataFrame:
        df = self._filter_product(self.df, product)

        if start_year is not None:
            df = df[
                df["_intervention_year"].notna()
                & (df["_intervention_year"] >= start_year)
            ]

        if end_year is not None:
            df = df[
                df["_intervention_year"].notna()
                & (df["_intervention_year"] <= end_year)
            ]

        if intervention_type:
            key = norm(intervention_type)
            df = df[
                df["_intervention_types"].apply(
                    lambda vals: any(norm(v) == key for v in vals)
                )
            ]

        if institution:
            key = norm(institution)
            df = df[
                df["_institutions"].apply(
                    lambda vals: any(norm(v) == key for v in vals)
                )
            ]

        if province:
            df = df[df["_province"].map(norm) == norm(province)]

        return df

    def metadata(self) -> dict[str, Any]:
        types, institutions = [], []

        for vals in self.df["_intervention_types"]:
            types.extend(vals)
        for vals in self.df["_institutions"]:
            institutions.extend(vals)

        years = sorted(
            int(v)
            for v in self.df["_intervention_year"].dropna().unique()
        )

        return {
            "record_count": int(len(self.df)),
            "dated_record_count": int(self.df["_intervention_year"].notna().sum()),
            "years": years,
            "intervention_types": canonical_values(types),
            "institutions": canonical_values(institutions),
        }

    def summary(
        self,
        start_year: int | None = None,
        end_year: int | None = None,
        intervention_type: str | None = None,
        institution: str | None = None,
        product: str | None = None,
    ) -> dict[str, Any]:
        df = self._filter(
            start_year=start_year,
            end_year=end_year,
            intervention_type=intervention_type,
            institution=institution,
            product=product,
        )

        province_counts = (
            df.groupby("_province").size().sort_values(ascending=False)
            if not df.empty else pd.Series(dtype=int)
        )

        year_counts = (
            df[df["_intervention_year"].notna()]
            .groupby("_intervention_year")
            .size()
            .sort_index()
            if not df.empty else pd.Series(dtype=int)
        )

        type_counts: dict[str, int] = {}
        institution_counts: dict[str, int] = {}

        for vals in df["_intervention_types"]:
            for value in vals:
                key = value.strip().lower()
                type_counts[key] = type_counts.get(key, 0) + 1

        for vals in df["_institutions"]:
            for value in vals:
                key = value.strip()
                institution_counts[key] = institution_counts.get(key, 0) + 1

        return {
            "record_count": int(len(df)),
            "province_counts": [
                {"province": str(k), "count": int(v)}
                for k, v in province_counts.items()
            ],
            "year_counts": [
                {"year": int(k), "count": int(v)}
                for k, v in year_counts.items()
            ],
            "type_counts": [
                {"name": k, "count": int(v)}
                for k, v in sorted(type_counts.items(), key=lambda x: -x[1])
            ],
            "institution_counts": [
                {"name": k, "count": int(v)}
                for k, v in sorted(institution_counts.items(), key=lambda x: -x[1])
            ],
        }

    def timeline(
        self,
        province: str | None = None,
        intervention_type: str | None = None,
        institution: str | None = None,
        product: str | None = None,
    ) -> list[dict[str, Any]]:
        df = self._filter(
            intervention_type=intervention_type,
            institution=institution,
            product=product,
            province=province,
        )

        df = df[df["_intervention_year"].notna()].copy()
        if df.empty:
            return []

        grouped = (
            df.groupby("_intervention_year")
            .size()
            .rename("event_count")
            .reset_index()
            .sort_values("_intervention_year")
        )

        return [
            {
                "year": int(row["_intervention_year"]),
                "event_count": int(row["event_count"]),
            }
            for _, row in grouped.iterrows()
        ]

    def province_detail(
        self,
        province_name: str,
        start_year: int | None = None,
        end_year: int | None = None,
        intervention_type: str | None = None,
        institution: str | None = None,
        product: str | None = None,
    ) -> dict[str, Any]:
        df = self._filter(
            start_year=start_year,
            end_year=end_year,
            intervention_type=intervention_type,
            institution=institution,
            product=product,
            province=province_name,
        )

        def counts(column: str) -> list[dict[str, Any]]:
            result: dict[str, int] = {}
            for vals in df[column]:
                for value in vals:
                    key = value.strip()
                    result[key] = result.get(key, 0) + 1
            return [
                {"name": k, "count": int(v)}
                for k, v in sorted(result.items(), key=lambda x: -x[1])
            ]

        years = sorted(
            int(v) for v in df["_intervention_year"].dropna().unique()
        )

        return {
            "province": province_name,
            "record_count": int(len(df)),
            "years": years,
            "types": counts("_intervention_types"),
            "institutions": counts("_institutions"),
            "actions": counts("_actions"),
            "actors": counts("_actors"),
            "sources": self._sources(df),
        }
