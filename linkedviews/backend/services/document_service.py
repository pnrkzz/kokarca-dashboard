from __future__ import annotations

from typing import Any

import pandas as pd

from repository import DataRepository
from services.base_service import BaseService
from utils import norm


class DocumentService(BaseService):
    """
    Belgeler Linked View servisidir.

    Harita / zaman / ürün seçimlerinden gelen bağlamı kullanarak
    ilgili belgeleri filtreler ve frontend için zengin belge
    nesneleri döndürür.
    """

    def __init__(self, repo: DataRepository) -> None:
        super().__init__(repo)

        columns = set(repo.df_all.columns)

        self.summary_col = (
            "summary_tr"
            if "summary_tr" in columns
            else None
        )

        self.evidence_col = (
            "evidence_sentence"
            if "evidence_sentence" in columns
            else None
        )

        self.latitude_col = (
            "latitude"
            if "latitude" in columns
            else None
        )

        self.longitude_col = (
            "longitude"
            if "longitude" in columns
            else None
        )

        self.event_id_col = (
            "event_id"
            if "event_id" in columns
            else None
        )

        self.global_id_col = (
            "global_id"
            if "global_id" in columns
            else None
        )

    # ---------------------------------------------------------
    # Yardımcı fonksiyonlar
    # ---------------------------------------------------------

    @staticmethod
    def _clean_text(
        value: object,
    ) -> str | None:

        if value is None:
            return None

        if isinstance(value, float) and pd.isna(value):
            return None

        text = str(value).strip()

        if not text:
            return None

        if text.lower() in {
            "nan",
            "none",
            "null",
        }:
            return None

        return text


    @staticmethod
    def _clean_number(
        value: object,
    ) -> float | None:

        if value is None:
            return None

        try:
            number = float(value)

            if pd.isna(number):
                return None

            return number

        except Exception:
            return None


    # ---------------------------------------------------------
    # Filtre
    # ---------------------------------------------------------

    def _filter_documents(
        self,
        *,
        province: str | None = None,
        product: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        source_type: str | None = None,
    ) -> pd.DataFrame:

        r = self.repo

        df = r.df_all.copy()

        # Ürün
        df = self._filter_product(
            df,
            product,
        )

        # İl
        if province:
            df = df[
                df["_province"]
                .map(norm)
                == norm(province)
            ]

        # Linked View zaman filtresi:
        # event_year / repository year_col kullanılır.
        if start_year is not None:

            year_values = pd.to_numeric(
                df[r.year_col],
                errors="coerce",
            )

            df = df[
                year_values.notna()
                & (
                    year_values
                    >= start_year
                )
            ]

        if end_year is not None:

            year_values = pd.to_numeric(
                df[r.year_col],
                errors="coerce",
            )

            df = df[
                year_values.notna()
                & (
                    year_values
                    <= end_year
                )
            ]

        # Kaynak türü
        if source_type and r.source_col:

            df = df[
                df[r.source_col]
                .map(norm)
                == norm(source_type)
            ]

        return df


    # ---------------------------------------------------------
    # Belge nesnesi
    # ---------------------------------------------------------

    def _row_to_document(
        self,
        row: pd.Series,
    ) -> dict[str, Any]:

        r = self.repo

        title = (
            self._clean_text(
                row[r.title_col]
            )
            if r.title_col
            else None
        )

        url = (
            self._clean_text(
                row[r.url_col]
            )
            if r.url_col
            else None
        )

        if (
            url
            and not url.startswith(
                (
                    "http://",
                    "https://",
                )
            )
        ):
            url = None

        source_type = (
            self._clean_text(
                row[r.source_col]
            )
            if r.source_col
            else None
        )

        published_date = None

        if r.published_date_col:
            value = row[
                r.published_date_col
            ]

            if pd.notna(value):

                try:
                    date_value = pd.to_datetime(
                        value,
                        errors="coerce",
                    )

                    if pd.notna(date_value):
                        published_date = (
                            date_value
                            .strftime(
                                "%Y-%m-%d"
                            )
                        )

                except Exception:
                    published_date = (
                        self._clean_text(
                            value
                        )
                    )

        event_year = None

        if pd.notna(
            row[r.year_col]
        ):
            try:
                event_year = int(
                    float(
                        row[r.year_col]
                    )
                )
            except Exception:
                event_year = None

        province = self._clean_text(
            row.get(
                "_province"
            )
        )

        summary = (
            self._clean_text(
                row[
                    self.summary_col
                ]
            )
            if self.summary_col
            else None
        )

        evidence_sentence = (
            self._clean_text(
                row[
                    self.evidence_col
                ]
            )
            if self.evidence_col
            else None
        )

        latitude = (
            self._clean_number(
                row[
                    self.latitude_col
                ]
            )
            if self.latitude_col
            else None
        )

        longitude = (
            self._clean_number(
                row[
                    self.longitude_col
                ]
            )
            if self.longitude_col
            else None
        )

        event_id = (
            self._clean_text(
                row[
                    self.event_id_col
                ]
            )
            if self.event_id_col
            else None
        )

        global_id = (
            self._clean_text(
                row[
                    self.global_id_col
                ]
            )
            if self.global_id_col
            else None
        )

        return {
            "event_id":
                event_id,

            "global_id":
                global_id,

            "title":
                title,

            "url":
                url,

            "source_type":
                source_type,

            "province":
                province,

            "event_year":
                event_year,

            "published_date":
                published_date,

            "summary":
                summary,

            "evidence_sentence":
                evidence_sentence,

            "latitude":
                latitude,

            "longitude":
                longitude,
        }


    # ---------------------------------------------------------
    # Ana arama
    # ---------------------------------------------------------

    def search(
        self,
        *,
        province: str | None = None,
        product: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        source_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        df = self._filter_documents(
            province=province,
            product=product,
            start_year=start_year,
            end_year=end_year,
            source_type=source_type,
        )

        if df.empty:
            return []

        # Yayım tarihine göre yeniler önce.
        if "_published_date" in df.columns:

            df = df.sort_values(
                "_published_date",
                ascending=False,
                na_position="last",
            )

        documents = []

        seen = set()

        for _, row in df.iterrows():

            document = (
                self._row_to_document(
                    row
                )
            )

            # Aynı belge birden fazla event ile eşleşmiş olabilir.
            # Öncelik global_id -> URL -> title.
            signature = (
                document["global_id"]
                or document["url"]
                or (
                    document["title"],
                    document["province"],
                    document["event_year"],
                )
            )

            if signature in seen:
                continue

            seen.add(
                signature
            )

            documents.append(
                document
            )

            if len(documents) >= limit:
                break

        return documents