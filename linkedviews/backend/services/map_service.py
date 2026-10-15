from __future__ import annotations

from typing import Any

from repository import DataRepository
from services.base_service import BaseService
from utils import canonical_values, norm


class MapService(BaseService):
    def metadata(self) -> dict[str, Any]:
        r = self.repo

        products = []
        for vals in r.df_all["_products"]:
            products.extend(vals)

        years = sorted(
            int(v)
            for v in r.df_temporal[r.year_col].dropna().unique()
        )

        published_years = sorted(
            int(v)
            for v in r.df_all["_published_year"].dropna().unique()
        )

        return {
            "event_count": int(len(r.df_all)),
            "temporal_event_count": int(len(r.df_temporal)),
            "document_count": (
                int(r.df_all[r.doc_col].nunique())
                if r.doc_col
                else int(len(r.df_all))
            ),
            "province_count": int(r.df_all["_province"].nunique()),
            "temporal_province_count": int(r.df_temporal["_province"].nunique()),
            "years": years,
            "published_years": published_years,
            "products": canonical_values(products),
            "product_column": r.product_col,
            "geojson_name_key": r.geo_name_key,
        }

    def base_geojson(self) -> dict[str, Any]:
        return self.repo.base_geojson

    def province_year_summary(
        self,
        product: str | None = None,
    ) -> list[dict[str, Any]]:
        r = self.repo
        df = self._filter_product(r.df_temporal, product)

        if df.empty:
            return []

        grouped = (
            df.groupby(["_province", r.year_col])
            .size()
            .rename("event_count")
            .reset_index()
        )

        return [
            {
                "province": str(row["_province"]),
                "year": int(row[r.year_col]),
                "event_count": int(row["event_count"]),
            }
            for _, row in grouped.iterrows()
        ]

    def timeline(
        self,
        product: str | None = None,
        province: str | None = None,
        time_basis: str = "event",
    ) -> list[dict[str, Any]]:
        r = self.repo

        if time_basis == "published":
            df = self._filter_product(r.df_all, product)
            year_col = "_published_year"
            df = df[df[year_col].notna()].copy()
            df[year_col] = df[year_col].astype(int)
        else:
            df = self._filter_product(r.df_temporal, product)
            year_col = r.year_col

        if province:
            df = df[df["_province"].map(norm) == norm(province)]

        if df.empty:
            return []

        grouped = (
            df.groupby(year_col)
            .size()
            .rename("event_count")
            .reset_index()
            .sort_values(year_col)
        )

        return [
            {
                "year": int(row[year_col]),
                "event_count": int(row["event_count"]),
            }
            for _, row in grouped.iterrows()
        ]

    def province_detail(
        self,
        province_name: str,
        product: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict[str, Any]:
        r = self.repo

        has_time_filter = start_year is not None or end_year is not None
        df = r.df_temporal if has_time_filter else r.df_all
        df = self._filter_product(df, product)
        df = df[df["_province"].map(norm) == norm(province_name)]

        if start_year is not None:
            df = df[df[r.year_col] >= start_year]
        if end_year is not None:
            df = df[df[r.year_col] <= end_year]

        if df.empty:
            return {
                "province": province_name,
                "event_count": 0,
                "document_count": 0,
                "first_year": None,
                "last_year": None,
                "first_published_year": None,
                "last_published_year": None,
                "sources": [],
            }

        temporal = df[df[r.year_col].notna()]
        published = df[df["_published_year"].notna()]

        return {
            "province": province_name,
            "event_count": int(len(df)),
            "document_count": (
                int(df[r.doc_col].nunique())
                if r.doc_col else int(len(df))
            ),
            "first_year": (
                int(temporal[r.year_col].min()) if not temporal.empty else None
            ),
            "last_year": (
                int(temporal[r.year_col].max()) if not temporal.empty else None
            ),
            "first_published_year": (
                int(published["_published_year"].min()) if not published.empty else None
            ),
            "last_published_year": (
                int(published["_published_year"].max()) if not published.empty else None
            ),
            "sources": self._sources(df),
        }
