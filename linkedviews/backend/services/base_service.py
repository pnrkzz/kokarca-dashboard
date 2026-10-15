from __future__ import annotations

from typing import Any

import pandas as pd

from repository import DataRepository
from utils import canonical_values, norm


class BaseService:
    def __init__(self, repo: DataRepository) -> None:
        self.repo = repo

    def _filter_product(
        self,
        df: pd.DataFrame,
        product: str | None,
    ) -> pd.DataFrame:
        if not product or norm(product) in {"", "tumurunler", "all"}:
            return df

        key = norm(product)
        return df[
            df["_products"].apply(
                lambda vals: any(norm(v) == key for v in vals)
            )
        ]

    def _sources(
        self,
        df: pd.DataFrame,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        r = self.repo
        sources = []
        seen = set()

        for _, row in df.iterrows():
            url = (
                str(row[r.url_col]).strip()
                if r.url_col and pd.notna(row[r.url_col])
                else None
            )
            if url and not url.startswith(("http://", "https://")):
                url = None

            title = (
                str(row[r.title_col]).strip()
                if r.title_col and pd.notna(row[r.title_col])
                else None
            )
            source_type = (
                str(row[r.source_col]).strip()
                if r.source_col and pd.notna(row[r.source_col])
                else None
            )

            sig = (url, title)
            if sig in seen:
                continue
            seen.add(sig)

            sources.append({
                "title": title,
                "url": url,
                "source_type": source_type,
            })

            if len(sources) >= limit:
                break

        return sources
