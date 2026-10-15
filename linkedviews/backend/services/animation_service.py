from __future__ import annotations

from collections import Counter
from typing import Any

from mongo_repository import MongoRepository


class AnimationService:

    MIN_YEAR = 2016
    MAX_YEAR = 2025
    VALID_MODES = {"annual", "cumulative"}

    def __init__(self, repository: MongoRepository) -> None:
        self.repo = repository
        self.collection = repository.animation

    # ========================================================
    # META
    # ========================================================

    def metadata(self) -> dict[str, Any]:

        year_pipeline = [
            {
                "$group": {
                    "_id": "$animation_year",
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]

        year_counts = {
            str(row["_id"]): int(row["count"])
            for row in self.collection.aggregate(year_pipeline)
        }

        type_pipeline = [
            {
                "$group": {
                    "_id": "$animation_type",
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"count": -1}},
        ]

        type_counts = {
            str(row["_id"]): int(row["count"])
            for row in self.collection.aggregate(type_pipeline)
        }

        return {
            "min_year": self.MIN_YEAR,
            "max_year": self.MAX_YEAR,
            "record_count": self.collection.count_documents({}),
            "event_count": len(
                self.collection.distinct("event_uid")
            ),
            "year_counts": year_counts,
            "type_counts": type_counts,
            "available_modes": [
                "annual",
                "cumulative",
            ],
            "coordinate_interpretation":
                "administrative_representative_point",
        }

    # ========================================================
    # QUERY VALIDATION
    # ========================================================

    def _build_query(
        self,
        year: int,
        mode: str,
        province: str | None = None,
    ) -> dict[str, Any]:

        if year < self.MIN_YEAR or year > self.MAX_YEAR:
            raise ValueError(
                f"year must be between "
                f"{self.MIN_YEAR} and {self.MAX_YEAR}"
            )

        if mode not in self.VALID_MODES:
            raise ValueError(
                "mode must be 'annual' or 'cumulative'"
            )

        query: dict[str, Any] = {}

        if mode == "annual":
            query["animation_year"] = year
        else:
            query["animation_year"] = {"$lte": year}

        if province:
            query["province"] = province

        return query

    # ========================================================
    # RAW ANIMATION RECORDS
    # ========================================================

    def records(
        self,
        year: int,
        mode: str = "annual",
        province: str | None = None,
    ) -> dict[str, Any]:

        query = self._build_query(
            year=year,
            mode=mode,
            province=province,
        )

        projection = {
            "_id": 0,

            "animation_uid": 1,
            "event_uid": 1,
            "location_uid": 1,

            "animation_year": 1,
            "animation_type": 1,

            "final_event_role": 1,
            "location_role": 1,

            "province": 1,
            "district": 1,
            "locality": 1,
            "display_location": 1,

            "longitude": 1,
            "latitude": 1,
            "geometry": 1,

            "geometry_precision": 1,
            "geometry_match_status": 1,

            "source_type": 1,
            "event_summary_tr": 1,
        }

        docs = list(
            self.collection.find(
                query,
                projection,
            ).sort(
                [
                    ("animation_year", 1),
                    ("province", 1),
                    ("district", 1),
                ]
            )
        )

        return {
            "year": year,
            "mode": mode,
            "province": province,
            "count": len(docs),
            "features": docs,
        }

    # ========================================================
    # AGGREGATED MAP LOCATIONS
    # ========================================================

    def aggregated(
        self,
        year: int,
        mode: str = "annual",
        province: str | None = None,
    ) -> dict[str, Any]:

        query = self._build_query(
            year=year,
            mode=mode,
            province=province,
        )

        pipeline = [
            {"$match": query},

            {
                "$group": {
                    "_id": {
                        "longitude": "$longitude",
                        "latitude": "$latitude",
                        "province": "$province",
                        "district": "$district",
                        "geometry_precision":
                            "$geometry_precision",
                    },

                    "record_count": {"$sum": 1},

                    "event_uids": {
                        "$addToSet": "$event_uid"
                    },

                    "types": {
                        "$push": "$animation_type"
                    },

                    "first_year": {
                        "$min": "$animation_year"
                    },

                    "last_year": {
                        "$max": "$animation_year"
                    },
                }
            },

            {"$sort": {"record_count": -1}},
        ]

        locations = []

        for row in self.collection.aggregate(pipeline):

            key = row["_id"]

            type_counts = dict(
                Counter(row.get("types", []))
            )

            locations.append(
                {
                    "longitude": key["longitude"],
                    "latitude": key["latitude"],
                    "province": key["province"],
                    "district": key["district"],
                    "geometry_precision":
                        key["geometry_precision"],

                    "record_count":
                        int(row["record_count"]),

                    "event_count":
                        len(row["event_uids"]),

                    "first_year":
                        int(row["first_year"]),

                    "last_year":
                        int(row["last_year"]),

                    "type_counts":
                        type_counts,
                }
            )

        return {
            "year": year,
            "mode": mode,
            "province": province,
            "location_count": len(locations),
            "locations": locations,
        }