from pathlib import Path
import os

BASE_DIR = Path(r"C:\Users\depoc\kokarca_project")

EVENTS_PATH = Path(
    os.getenv(
        "KOKARCA_EVENTS_CSV",
        BASE_DIR / "data_processed" / "uzalcbs_2026" / "data" / "uzalcbs_2026_event_dataset_final_validated_v2.csv",
    )
)

GEOJSON_PATH = Path(
    os.getenv(
        "KOKARCA_PROVINCE_GEOJSON",
        BASE_DIR / "spatial" / "lvl1-TR.geojson",
    )
)
