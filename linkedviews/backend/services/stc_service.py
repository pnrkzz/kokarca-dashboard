from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from services.base_service import BaseService


class STCService(BaseService):
    """
    Final locked hazelnut province-year STC veri ürününü servis eder.

    STC burada yeniden hesaplanmaz.
    Backend doğrulanmış final STC hücrelerini API'ye aktarır.
    Görselleştirme frontend tarafında gerçekleştirilir.
    """

    def __init__(self, repo):
        super().__init__(repo)

        project_root = Path(__file__).resolve().parents[2]

        self.stc_path = (
            project_root
            / "data_processed"
            / "STC_IKLIM"
            / "01_STC"
            / "final_locked"
            / "visualization_data"
            / "FINAL_hazelnut_province_year_cube.csv"
        )

        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        if not self.stc_path.exists():
            raise FileNotFoundError(
                f"Final locked STC dosyası bulunamadı: {self.stc_path}"
            )

        df = pd.read_csv(self.stc_path)

        required = {
            "province_std",
            "stc_year",
            "event_count",
        }

        missing = required.difference(df.columns)

        if missing:
            raise ValueError(
                "Final STC dosyasında gerekli sütunlar eksik: "
                + ", ".join(sorted(missing))
            )

        df = df[
            [
                "province_std",
                "stc_year",
                "event_count",
            ]
        ].copy()

        df["province_std"] = (
            df["province_std"]
            .astype(str)
            .str.strip()
        )

        df["stc_year"] = pd.to_numeric(
            df["stc_year"],
            errors="coerce",
        )

        df["event_count"] = pd.to_numeric(
            df["event_count"],
            errors="coerce",
        ).fillna(0)

        df = df.dropna(
            subset=["province_std", "stc_year"]
        )

        df["stc_year"] = df["stc_year"].astype(int)
        df["event_count"] = df["event_count"].astype(int)

        df = df.sort_values(
            ["province_std", "stc_year"]
        ).reset_index(drop=True)

        self._df = df
        return self._df

    def cells(
        self,
        product: str | None = None,
    ) -> list[dict[str, Any]]:
        if product:
            normalized = (
                str(product)
                .strip()
                .lower()
                .replace("ı", "i")
            )

            if normalized not in {
                "findik",
                "hazelnut",
            }:
                return []

        df = self._load()

        return [
            {
                "province": str(row.province_std),
                "year": int(row.stc_year),
                "event_count": int(row.event_count),
            }
            for row in df.itertuples(index=False)
        ]

    def metadata(self) -> dict[str, Any]:
        df = self._load()

        years = sorted(
            int(value)
            for value in df["stc_year"].unique()
        )

        provinces = sorted(
            str(value)
            for value in df["province_std"].unique()
        )

        yearly_totals = (
            df.groupby("stc_year")["event_count"]
            .sum()
            .astype(int)
            .to_dict()
        )

        return {
            "dataset": "FINAL_hazelnut_province_year_cube",
            "status": "final_locked",
            "product": "fındık",
            "spatial_unit": "province",
            "temporal_unit": "year",
            "cell_count": int(len(df)),
            "province_count": int(len(provinces)),
            "year_count": int(len(years)),
            "min_year": int(min(years)),
            "max_year": int(max(years)),
            "total_event_count": int(
                df["event_count"].sum()
            ),
            "max_cell_count": int(
                df["event_count"].max()
            ),
            "years": years,
            "provinces": provinces,
            "yearly_totals": {
                str(year): int(count)
                for year, count in yearly_totals.items()
            },
        }
