from __future__ import annotations
from stc_multilayer_router import router as stc_multilayer_router

from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from config import EVENTS_PATH, GEOJSON_PATH
from repository import DataRepository
from mongo_repository import MongoRepository
from services.map_service import MapService
from services.intervention_service import InterventionService
from services.document_service import DocumentService
from services.climate_service import ClimateService
from services.stc_service import STCService
from services.animation_service import AnimationService
from climate_relations_router import router as climate_relations_router
from questions_router import router as questions_router

app = FastAPI(
    title="Kahverengi Kokarca Geovisual Analytics API",
    version="1.1.0",
)

app.include_router(climate_relations_router)
app.include_router(questions_router)
app.include_router(stc_multilayer_router)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repo = DataRepository(EVENTS_PATH, GEOJSON_PATH)
map_service = MapService(repo)
intervention_service = InterventionService(repo)
document_service = DocumentService(repo)
climate_service = ClimateService(repo)
stc_service = STCService(repo)
def get_animation_service():
    """MongoDB bağlantısını yalnızca animasyon çağrıldığında aç."""
    from fastapi import HTTPException
    from pymongo.errors import PyMongoError

    try:
        return AnimationService(MongoRepository())
    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="MongoDB animasyon servisi şu anda kullanılamıyor."
        )



@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "version": "1.1.0",
        "docs": "/docs",
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "events_csv": str(EVENTS_PATH),
        "geojson": str(GEOJSON_PATH),
    }


# ---------------------------
# Map / time / product
# ---------------------------

@app.get("/api/metadata")
def metadata() -> dict[str, Any]:
    return map_service.metadata()


@app.get("/api/base-geojson")
def base_geojson() -> dict[str, Any]:
    return map_service.base_geojson()


@app.get("/api/province-year-summary")
def province_year_summary(
    product: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    return map_service.province_year_summary(product)


@app.get("/api/timeline")
def timeline(
    product: str | None = Query(default=None),
    province: str | None = Query(default=None),
    time_basis: str = Query(default="event"),
) -> list[dict[str, Any]]:
    return map_service.timeline(
        product=product,
        province=province,
        time_basis=time_basis,
    )


@app.get("/api/province/{province_name}")
def province_detail(
    province_name: str,
    product: str | None = Query(default=None),
    start_year: int | None = Query(default=None),
    end_year: int | None = Query(default=None),
) -> dict[str, Any]:
    return map_service.province_detail(
        province_name,
        product=product,
        start_year=start_year,
        end_year=end_year,
    )


# ---------------------------
# Interventions
# ---------------------------

@app.get("/api/interventions/metadata")
def intervention_metadata() -> dict[str, Any]:
    return intervention_service.metadata()


@app.get("/api/interventions/summary")
def intervention_summary(
    start_year: int | None = Query(default=None),
    end_year: int | None = Query(default=None),
    intervention_type: str | None = Query(default=None),
    institution: str | None = Query(default=None),
    product: str | None = Query(default=None),
) -> dict[str, Any]:
    return intervention_service.summary(
        start_year=start_year,
        end_year=end_year,
        intervention_type=intervention_type,
        institution=institution,
        product=product,
    )


@app.get("/api/interventions/timeline")
def intervention_timeline(
    province: str | None = Query(default=None),
    intervention_type: str | None = Query(default=None),
    institution: str | None = Query(default=None),
    product: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    return intervention_service.timeline(
        province=province,
        intervention_type=intervention_type,
        institution=institution,
        product=product,
    )


@app.get("/api/interventions/province/{province_name}")
def intervention_province_detail(
    province_name: str,
    start_year: int | None = Query(default=None),
    end_year: int | None = Query(default=None),
    intervention_type: str | None = Query(default=None),
    institution: str | None = Query(default=None),
    product: str | None = Query(default=None),
) -> dict[str, Any]:
    return intervention_service.province_detail(
        province_name,
        start_year=start_year,
        end_year=end_year,
        intervention_type=intervention_type,
        institution=institution,
        product=product,
    )


# ---------------------------
# Documents
# ---------------------------

@app.get("/api/documents")
def documents(
    province: str | None = Query(default=None),
    product: str | None = Query(default=None),

    start_year: int | None = Query(
        default=None
    ),

    end_year: int | None = Query(
        default=None
    ),

    source_type: str | None = Query(
        default=None
    ),

    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),

) -> list[dict[str, Any]]:

    return document_service.search(
        province=province,
        product=product,
        start_year=start_year,
        end_year=end_year,
        source_type=source_type,
        limit=limit,
    )
# ---------------------------
# Climate
# ---------------------------

@app.get("/api/climate/metadata")
def climate_metadata() -> dict[str, Any]:
    return climate_service.metadata()


@app.get("/api/climate/province-counts")
def climate_province_counts(
    factor: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    return climate_service.province_counts(factor=factor)


@app.get("/api/climate/timeline")
def climate_timeline(
    province: str | None = Query(default=None),
    factor: str | None = Query(default=None),
    time_basis: str = Query(default="published"),
) -> list[dict[str, Any]]:
    return climate_service.timeline(
        province=province,
        factor=factor,
        time_basis=time_basis,
    )
@app.get("/api/climate/summary")
def climate_summary(
    province: str | None = Query(default=None),
    start_year: int | None = Query(default=None),
    end_year: int | None = Query(default=None),
    product: str | None = Query(default=None),
) -> dict[str, Any]:

    return climate_service.summary(
        province=province,
        start_year=start_year,
        end_year=end_year,
        product=product,
    )


@app.get("/api/climate/factors")
def climate_factors(
    province: str | None = Query(default=None),
    start_year: int | None = Query(default=None),
    end_year: int | None = Query(default=None),
    product: str | None = Query(default=None),
) -> list[dict[str, Any]]:

    return climate_service.factors(
        province=province,
        start_year=start_year,
        end_year=end_year,
        product=product,
    )


@app.get("/api/climate/evidence")
def climate_evidence(
    province: str | None = Query(default=None),
    start_year: int | None = Query(default=None),
    end_year: int | None = Query(default=None),
    factor: str | None = Query(default=None),
    consistency: str | None = Query(default=None),
    product: str | None = Query(default=None),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
) -> list[dict[str, Any]]:

    return climate_service.evidence(
        province=province,
        start_year=start_year,
        end_year=end_year,
        factor=factor,
        consistency=consistency,
        product=product,
        limit=limit,
    )


@app.get("/api/climate/era5")
def climate_era5(
    province: str | None = Query(default=None),
    start_year: int | None = Query(default=None),
    end_year: int | None = Query(default=None),
    product: str | None = Query(default=None),
) -> dict[str, Any]:

    return climate_service.era5_summary(
        province=province,
        start_year=start_year,
        end_year=end_year,
        product=product,
    )


@app.get("/api/climate/rag-card")
def climate_rag_card(
    province: str = Query(...),
    year: int = Query(...),
) -> dict[str, Any] | None:

    return climate_service.rag_card(
        province=province,
        year=year,
    )

# ---------------------------
# STC
# ---------------------------

@app.get("/api/stc/cells")
def stc_cells(
    product: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    return stc_service.cells(product=product)


@app.get("/api/stc/meta")
def stc_metadata() -> dict[str, Any]:
    return stc_service.metadata()
# ---------------------------
# Temporal H. halys Animation
# MongoDB V2
# ---------------------------

@app.get("/api/animation/meta")
def animation_metadata() -> dict[str, Any]:
    return get_animation_service().metadata()


@app.get("/api/animation")
def animation_records(
    year: int = Query(..., ge=2016, le=2025),
    mode: str = Query(
        default="annual",
        pattern="^(annual|cumulative)$",
    ),
    province: str | None = Query(default=None),
) -> dict[str, Any]:

    return get_animation_service().records(
        year=year,
        mode=mode,
        province=province,
    )


@app.get("/api/animation/aggregated")
def animation_aggregated(
    year: int = Query(..., ge=2016, le=2025),
    mode: str = Query(
        default="annual",
        pattern="^(annual|cumulative)$",
    ),
    province: str | None = Query(default=None),
) -> dict[str, Any]:

    return get_animation_service().aggregated(
        year=year,
        mode=mode,
        province=province,
    )




from v22_interventions_router import router as v22_interventions_router
app.include_router(v22_interventions_router)
