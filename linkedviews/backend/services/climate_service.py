from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from repository import DataRepository
from services.base_service import BaseService
from utils import canonical_values, norm, parse_multi


BASE_DIR = Path(
    os.getenv(
        "KOKARCA_UZALCBS_DIR",
        r"C:\Users\depoc\kokarca_project\data_processed\uzalcbs_2026",
    )
)

CEVD_PATH = Path(
    os.getenv(
        "KOKARCA_CEVD_CSV",
        str(
            BASE_DIR
            / "data"
            / "climate_evidence_validation_dataset_cevd_final.csv"
        ),
    )
)

ERA5_CONSISTENCY_PATH = Path(
    os.getenv(
        "KOKARCA_ERA5_CONSISTENCY_CSV",
        str(
            BASE_DIR
            / "data"
            / "uzalcbs_2026_climate_factor_era5_consistency_final.csv"
        ),
    )
)
SEMANTIC_AUDIT_PATH = Path(
    os.getenv(
        "KOKARCA_CLIMATE_SEMANTIC_AUDIT_CSV",
        str(
            BASE_DIR
            / "data"
            / "climate_semantic_audit_gpt_unique_era5eligible_final.csv"
        ),
    )
)

DIRECT_ERA5_LOCKED_PATH = Path(
    os.getenv(
        "KOKARCA_DIRECT_ERA5_LOCKED_CSV",
        str(
            BASE_DIR
            / "data"
            / "climate_narrative_era5_direct_consistency_LOCKED.csv"
        ),
    )
)
RAG_PATH = Path(
    os.getenv(
        "KOKARCA_CLIMATEGPT_RAG_CSV",
        str(
            BASE_DIR
            / "rag"
            / "climategpt_rag_province_year_cards.csv"
        ),
    )
)

EVIDENCE_PACKAGES_PATH = Path(
    os.getenv(
        "KOKARCA_CLIMATEGPT_PACKAGES_CSV",
        str(
            BASE_DIR
            / "rag"
            / "climategpt_evidence_packages.csv"
        ),
    )
)


class ClimateService(BaseService):

    def __init__(
        self,
        repo: DataRepository,
    ) -> None:

        super().__init__(repo)

        # Eski metin tabanlı climate dataframe.
        self.df = self._prepare()

        # Yeni kanıt katmanları.
        self.cevd = self._read_optional_csv(
            CEVD_PATH
        )

        self.era5 = self._read_optional_csv(
            ERA5_CONSISTENCY_PATH
        )

        self.semantic_audit = self._read_optional_csv(
            SEMANTIC_AUDIT_PATH
        )


        self.direct_era5_locked = self._read_optional_csv(
            DIRECT_ERA5_LOCKED_PATH
        )
        self.rag = self._read_optional_csv(
            RAG_PATH
        )

        self.evidence_packages = (
            self._read_optional_csv(
                EVIDENCE_PACKAGES_PATH
            )
        )

        self._prepare_external_tables()
        print("\n[ClimateService] LOCKED CLIMATE DATA")
        print("CEVD:", len(self.cevd))
        print("Semantic audit:", len(self.semantic_audit))
        print("Direct ERA5 locked:", len(self.direct_era5_locked))

        if (
            not self.semantic_audit.empty
            and "era5_comparison_level_locked"
            in self.semantic_audit.columns
        ):
            print("\nERA5 eligibility:")
            print(
                self.semantic_audit[
                    "era5_comparison_level_locked"
                ].value_counts(dropna=False)
        )

        if (
            not self.direct_era5_locked.empty
            and "era5_consistency_locked"
            in self.direct_era5_locked.columns
        ):
            print("\nDirect ERA5 consistency:")
            print(
                self.direct_era5_locked[
                    "era5_consistency_locked"
                ].value_counts(dropna=False)
        )
    # =========================================================
    # Genel yardımcılar
    # =========================================================

    @staticmethod
    def _read_optional_csv(
        path: Path,
    ) -> pd.DataFrame:

        if not path.exists():
            return pd.DataFrame()

        try:
            return pd.read_csv(
                path,
                low_memory=False,
            )

        except Exception as exc:
            print(
                f"[ClimateService] "
                f"CSV okunamadı: {path} -> {exc}"
            )

            return pd.DataFrame()

    @staticmethod
    def _clean_text(
        value: object,
    ) -> str | None:

        if value is None:
            return None

        if (
            isinstance(
                value,
                float,
            )
            and pd.isna(value)
        ):
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
    def _number(
        value: object,
    ) -> float | None:

        try:
            number = float(value)

            if pd.isna(number):
                return None

            return number

        except Exception:
            return None

    @staticmethod
    def _truthy(
        value: object,
    ) -> bool:

        return str(value).strip().lower() in {
            "true",
            "1",
            "yes",
            "evet",
            "var",
        }

    # =========================================================
    # Mevcut ana veri climate hazırlığı
    # =========================================================

    def _prepare(
        self,
    ) -> pd.DataFrame:

        r = self.repo

        df = r.df_all.copy()

        if r.climate_factor_col:

            df["_climate_factors"] = (
                df[
                    r.climate_factor_col
                ]
                .apply(
                    parse_multi
                )
            )

        else:

            df["_climate_factors"] = [
                []
                for _
                in range(
                    len(df)
                )
            ]

        if r.climate_mentioned_col:

            raw = (
                df[
                    r.climate_mentioned_col
                ]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            mentioned = raw.isin(
                {
                    "true",
                    "1",
                    "yes",
                    "evet",
                    "var",
                }
            )

        else:

            mentioned = (
                df[
                    "_climate_factors"
                ]
                .map(bool)
            )

        return df[
            mentioned
            | df[
                "_climate_factors"
            ].map(bool)
        ].copy()

    # =========================================================
    # Harici climate tablolarını standardize et
    # =========================================================

    def _prepare_external_tables(
        self,
    ) -> None:

        for df in [
            self.cevd,
            self.era5,
            self.semantic_audit,
            self.direct_era5_locked,
        ]:

            if df.empty:
                continue

            if "province" in df.columns:

                df["_province_norm"] = (
                    df[
                        "province"
                    ]
                    .map(norm)
                )

            if "climate_year" in df.columns:

                df["_climate_year"] = (
                    pd.to_numeric(
                        df[
                            "climate_year"
                        ],
                        errors="coerce",
                    )
                )

            if (
                "mentions_climate_relation"
                in df.columns
            ):

                df[
                    "_climate_relation"
                ] = (
                    df[
                        "mentions_climate_relation"
                    ]
                    .map(
                        self._truthy
                    )
                )

        if not self.rag.empty:

            if "province" in self.rag.columns:

                self.rag[
                    "_province_norm"
                ] = (
                    self.rag[
                        "province"
                    ]
                    .map(norm)
                )

            if (
                "climate_year"
                in self.rag.columns
            ):

                self.rag[
                    "_climate_year"
                ] = (
                    pd.to_numeric(
                        self.rag[
                            "climate_year"
                        ],
                        errors="coerce",
                    )
                )

        if (
            not self.evidence_packages.empty
            and "province"
            in self.evidence_packages.columns
        ):

            self.evidence_packages[
                "_province_norm"
            ] = (
                self.evidence_packages[
                    "province"
                ]
                .map(norm)
            )

    # =========================================================
    # Linked product filter
    # =========================================================

    def _allowed_event_ids(
        self,
        product: str | None,
    ) -> set[str] | None:

        if not product:
            return None

        product_df = (
            self._filter_product(
                self.repo.df_all,
                product,
            )
        )

        if "event_id" not in product_df.columns:
            return None

        return set(
            product_df[
                "event_id"
            ]
            .dropna()
            .astype(str)
        )

    # =========================================================
    # CEVD / evidence filtreleme
    # =========================================================

    def _filter_evidence(
        self,
        *,
        province: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        factor: str | None = None,
        consistency: str | None = None,
        product: str | None = None,
        narratives_only: bool = False,
    ) -> pd.DataFrame:

        if self.cevd.empty:
            return pd.DataFrame()

        df = self.cevd.copy()

        if province:

            df = df[
                df[
                    "_province_norm"
                ]
                == norm(
                    province
                )
            ]

        if start_year is not None:

            df = df[
                df[
                    "_climate_year"
                ].notna()
                & (
                    df[
                        "_climate_year"
                    ]
                    >= start_year
                )
            ]

        if end_year is not None:

            df = df[
                df[
                    "_climate_year"
                ].notna()
                & (
                    df[
                        "_climate_year"
                    ]
                    <= end_year
                )
            ]

        if factor:

            key = norm(
                factor
            )

            if (
                "climate_factor_group"
                in df.columns
            ):

                df = df[
                    df[
                        "climate_factor_group"
                    ]
                    .map(norm)
                    == key
                ]

        if consistency:

            key = norm(
                consistency
            )

            if (
                "era5_consistency"
                in df.columns
            ):

                df = df[
                    df[
                        "era5_consistency"
                    ]
                    .map(norm)
                    == key
                ]

        allowed_ids = (
            self._allowed_event_ids(
                product
            )
        )

        if (
            allowed_ids is not None
            and "event_id"
            in df.columns
        ):

            df = df[
                df[
                    "event_id"
                ]
                .astype(str)
                .isin(
                    allowed_ids
                )
            ]

        if narratives_only:

            if (
                "_climate_relation"
                in df.columns
            ):

                df = df[
                    df[
                        "_climate_relation"
                    ]
                ]

        return df

    # =========================================================
    # Metadata
    # =========================================================

    def metadata(
        self,
    ) -> dict[str, Any]:

        factors = []

        for values in self.df[
            "_climate_factors"
        ]:

            factors.extend(
                values
            )

        evidence_factors = []

        if (
            not self.cevd.empty
            and "climate_factor_group"
            in self.cevd.columns
        ):

            evidence_factors = (
                self.cevd[
                    "climate_factor_group"
                ]
                .dropna()
                .astype(str)
                .loc[
                    lambda s:
                    s.str.lower()
                    != "unknown"
                ]
                .unique()
                .tolist()
            )

        return {
            "record_count":
                int(
                    len(
                        self.df
                    )
                ),

            "cevd_record_count":
                int(
                    len(
                        self.cevd
                    )
                ),

            "era5_record_count":
                int(
                    len(
                        self.era5
                    )
                ),

            "rag_card_count":
                int(
                    len(
                        self.rag
                    )
                ),

            "evidence_package_count":
                int(
                    len(
                        self.evidence_packages
                    )
                ),

            "factors":
                canonical_values(
                    factors
                ),

            "evidence_factor_groups":
                sorted(
                    evidence_factors
                ),

            "era5_connected":
                bool(
                    not self.era5.empty
                ),

            "climategpt_connected":
                bool(
                    not self.rag.empty
                    or not self.evidence_packages.empty
                ),
        }

    # =========================================================
    # Eski province counts - korunuyor
    # =========================================================

    def province_counts(
        self,
        factor: str | None = None,
    ) -> list[dict[str, Any]]:

        df = self.df

        if factor:

            key = norm(
                factor
            )

            df = df[
                df[
                    "_climate_factors"
                ]
                .apply(
                    lambda values:
                    any(
                        norm(value)
                        == key
                        for value
                        in values
                    )
                )
            ]

        grouped = (
            df.groupby(
                "_province"
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        return [
            {
                "province":
                    str(province),

                "count":
                    int(count),
            }

            for province, count
            in grouped.items()
        ]

    # =========================================================
    # Eski timeline - korunuyor
    # =========================================================

    def timeline(
        self,
        province: str | None = None,
        factor: str | None = None,
        time_basis: str = "published",
    ) -> list[dict[str, Any]]:

        r = self.repo

        df = self.df

        if province:

            df = df[
                df[
                    "_province"
                ]
                .map(norm)
                == norm(
                    province
                )
            ]

        if factor:

            key = norm(
                factor
            )

            df = df[
                df[
                    "_climate_factors"
                ]
                .apply(
                    lambda values:
                    any(
                        norm(value)
                        == key
                        for value
                        in values
                    )
                )
            ]

        year_col = (
            r.year_col
            if time_basis == "event"
            else "_published_year"
        )

        df = df[
            df[
                year_col
            ].notna()
        ].copy()

        if df.empty:
            return []

        df[
            year_col
        ] = (
            df[
                year_col
            ]
            .astype(int)
        )

        grouped = (
            df.groupby(
                year_col
            )
            .size()
            .rename(
                "event_count"
            )
            .reset_index()
            .sort_values(
                year_col
            )
        )

        return [
            {
                "year":
                    int(
                        row[
                            year_col
                        ]
                    ),

                "event_count":
                    int(
                        row[
                            "event_count"
                        ]
                    ),
            }

            for _, row
            in grouped.iterrows()
        ]

    # =========================================================
    # NEW: Linked Climate Summary
    # =========================================================

    def summary(
        self,
        *,
        province: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        product: str | None = None,
    ) -> dict[str, Any]:

        df = self._filter_evidence(
            province=province,
            start_year=start_year,
            end_year=end_year,
            product=product,
        )

        if df.empty:

            return {
                "record_count": 0,
                "climate_narrative_count": 0,
                "era5_testable_count": 0,
                "consistent_count": 0,
                "partially_consistent_count": 0,
                "not_consistent_count": 0,
                "observed_anomaly_count": 0,
            }

        narratives = df[
            df[
                "_climate_relation"
            ]
        ]

        consistency = (
            df[
                "era5_consistency"
            ]
            .fillna(
                "Unknown"
            )
        )

        return {
            "record_count":
                int(
                    len(df)
                ),

            "climate_narrative_count":
                int(
                    len(
                        narratives
                    )
                ),

            "era5_testable_count":
                int(
                    consistency.isin(
                        {
                            "Consistent",
                            "Partially consistent",
                            "Not consistent",
                            "Observed anomaly present",
                        }
                    ).sum()
                ),

            "consistent_count":
                int(
                    (
                        consistency
                        == "Consistent"
                    ).sum()
                ),

            "partially_consistent_count":
                int(
                    (
                        consistency
                        == "Partially consistent"
                    ).sum()
                ),

            "not_consistent_count":
                int(
                    (
                        consistency
                        == "Not consistent"
                    ).sum()
                ),

            "observed_anomaly_count":
                int(
                    (
                        consistency
                        == "Observed anomaly present"
                    ).sum()
                ),
        }

    # =========================================================
    # NEW: Climate Factors
    # =========================================================

    def factors(
        self,
        *,
        province: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        product: str | None = None,
    ) -> list[dict[str, Any]]:

        df = self._filter_evidence(
            province=province,
            start_year=start_year,
            end_year=end_year,
            product=product,
            narratives_only=True,
        )

        if (
            df.empty
            or "climate_factor_group"
            not in df.columns
        ):
            return []

        counts = (
            df[
                "climate_factor_group"
            ]
            .dropna()
            .astype(str)
        )

        counts = counts[
            counts.str.lower()
            != "unknown"
        ]

        counts = (
            counts
            .value_counts()
        )

        return [
            {
                "factor":
                    str(factor),

                "count":
                    int(count),
            }

            for factor, count
            in counts.items()
        ]

    # =========================================================
    # NEW: Evidence Explorer
    # =========================================================

    def evidence(
        self,
        *,
        province: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        factor: str | None = None,
        consistency: str | None = None,
        product: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        df = self._filter_evidence(
            province=province,
            start_year=start_year,
            end_year=end_year,
            factor=factor,
            consistency=consistency,
            product=product,
            narratives_only=True,
        )

        if df.empty:
            return []

        documents = []

        for _, row in (
            df.head(
                limit
            )
            .iterrows()
        ):

            documents.append(
                {
                    "event_id":
                        self._clean_text(
                            row.get(
                                "event_id"
                            )
                        ),

                    "global_id":
                        self._clean_text(
                            row.get(
                                "global_id"
                            )
                        ),

                    "province":
                        self._clean_text(
                            row.get(
                                "province"
                            )
                        ),

                    "climate_year":
                        (
                            int(
                                row[
                                    "_climate_year"
                                ]
                            )
                            if pd.notna(
                                row.get(
                                    "_climate_year"
                                )
                            )
                            else None
                        ),

                    "spread_observation_year":
                        self._number(
                            row.get(
                                "spread_observation_year"
                            )
                        ),

                    "source_type":
                        self._clean_text(
                            row.get(
                                "source_type"
                            )
                        ),

                    "published_date":
                        self._clean_text(
                            row.get(
                                "published_date"
                            )
                        ),

                    "factor":
                        self._clean_text(
                            row.get(
                                "climate_factor_group"
                            )
                        ),

                    "factor_raw":
                        self._clean_text(
                            row.get(
                                "climate_factor_type_raw"
                            )
                        ),

                    "direction":
                        self._clean_text(
                            row.get(
                                "climate_relation_direction"
                            )
                        ),

                    "evidence_sentence":
                        self._clean_text(
                            row.get(
                                "evidence_sentence"
                            )
                        ),

                    "climate_comment":
                        self._clean_text(
                            row.get(
                                "climate_comment"
                            )
                        ),

                    "era5_consistency":
                        self._clean_text(
                            row.get(
                                "era5_consistency"
                            )
                        ),

                    "consistency_reason":
                        self._clean_text(
                            row.get(
                                "consistency_reason"
                            )
                        ),

                    "annual_temperature_anomaly":
                        self._number(
                            row.get(
                                "annual_temperature_c_mean_anomaly"
                            )
                        ),

                    "winter_temperature_anomaly":
                        self._number(
                            row.get(
                                "winter_temperature_c_mean_anomaly"
                            )
                        ),

                    "spring_temperature_anomaly":
                        self._number(
                            row.get(
                                "spring_temperature_c_mean_anomaly"
                            )
                        ),

                    "summer_temperature_anomaly":
                        self._number(
                            row.get(
                                "summer_temperature_c_mean_anomaly"
                            )
                        ),

                    "autumn_temperature_anomaly":
                        self._number(
                            row.get(
                                "autumn_temperature_c_mean_anomaly"
                            )
                        ),

                    "relative_humidity_anomaly":
                        self._number(
                            row.get(
                                "annual_relative_humidity_mean_anomaly"
                            )
                        ),

                    "precipitation_anomaly":
                        self._number(
                            row.get(
                                "annual_precipitation_mm_sum_anomaly"
                            )
                        ),

                    "validation_flag":
                        self._clean_text(
                            row.get(
                                "validation_flag"
                            )
                        ),
                }
            )

        return documents

    # =========================================================
    # NEW: ERA5 climate summary
    # =========================================================

    def era5_summary(
        self,
        *,
        province: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        product: str | None = None,
    ) -> dict[str, Any]:

        df = self._filter_evidence(
            province=province,
            start_year=start_year,
            end_year=end_year,
            product=product,
        )

        if df.empty:

            return {
                "record_count": 0,
                "temperature_anomaly_mean": None,
                "humidity_anomaly_mean": None,
                "precipitation_anomaly_mean": None,
            }

        def mean_col(
            column: str,
        ) -> float | None:

            if column not in df.columns:
                return None

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if not values.notna().any():
                return None

            return float(
                values.mean()
            )

        return {
            "record_count":
                int(
                    len(df)
                ),

            "temperature_anomaly_mean":
                mean_col(
                    "annual_temperature_c_mean_anomaly"
                ),

            "winter_temperature_anomaly_mean":
                mean_col(
                    "winter_temperature_c_mean_anomaly"
                ),

            "humidity_anomaly_mean":
                mean_col(
                    "annual_relative_humidity_mean_anomaly"
                ),

            "precipitation_anomaly_mean":
                mean_col(
                    "annual_precipitation_mm_sum_anomaly"
                ),
        }

    # =========================================================
    # NEW: ClimateGPT / RAG province-year card
    # =========================================================

    def rag_card(
        self,
        *,
        province: str,
        year: int,
    ) -> dict[str, Any] | None:

        if self.rag.empty:
            return None

        df = self.rag[
            (
                self.rag[
                    "_province_norm"
                ]
                == norm(
                    province
                )
            )
            & (
                self.rag[
                    "_climate_year"
                ]
                == year
            )
        ]

        if df.empty:
            return None

        row = df.iloc[0]

        result = {}

        for column in [
            "province",
            "climate_year",
            "event_count",
            "document_count",
            "source_types",
            "climate_narrative_count",
            "climate_factors",
            "climate_comments",
            "spread_stages",
            "population_signals",
            "intervention_count",
            "intervention_types",
            "institutions",
            "government_actions",
            "evidence_examples",
            "summary_examples",
            "temperature_c_mean",
            "temperature_c_min",
            "temperature_c_max",
            "relative_humidity_mean",
            "precipitation_mm_sum",
            "temperature_c_mean_anomaly",
            "temperature_c_min_anomaly",
            "temperature_c_max_anomaly",
            "relative_humidity_mean_anomaly",
            "precipitation_mm_sum_anomaly",
            "rag_id",
            "rag_context",
        ]:

            if column not in row.index:
                continue

            value = row[column]

            if pd.isna(value):
                result[column] = None

            elif isinstance(
                value,
                (
                    int,
                    float,
                ),
            ):
                result[column] = (
                    float(value)
                )

            else:
                result[column] = str(
                    value
                )

        return result