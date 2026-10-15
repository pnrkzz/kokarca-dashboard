from __future__ import annotations

import os
from typing import Any
from pymongo import MongoClient


class MongoRepository:
    """
    MongoDB Atlas V2 repository.
    Existing CSV DataRepository remains unchanged.
    """

    def __init__(self) -> None:

        # Notebook'ta BAŞARILI çalışan URI'nin aynısını buraya yaz.
        MONGODB_URI = os.environ["MONGODB_URI"]

        self.client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=10000,
        )

        # Connection test
        self.client.admin.command("ping")

        # Database
        self.db = self.client["kokarca_knowledge_v2"]

        # Core collections
        self.events = self.db["events_v2"]
        self.locations = self.db["event_location_v2"]
        self.animation = self.db["animation_event_v2"]

        # Fact collections
        self.crop_host = self.db["event_crop_host_v2"]
        self.damage = self.db["event_damage_v2"]
        self.economic_impact = self.db["event_economic_impact_v2"]
        self.quantity = self.db["event_quantity_v2"]
        self.intervention = self.db["event_intervention_v2"]
        self.actor = self.db["event_actor_v2"]
        self.environment_climate = self.db[
            "event_environment_climate_v2"
        ]

    def health(self) -> dict[str, Any]:

        self.client.admin.command("ping")

        return {
            "status": "ok",
            "database": self.db.name,
            "events": self.events.count_documents({}),
            "locations": self.locations.count_documents({}),
            "animation": self.animation.count_documents({}),
        }