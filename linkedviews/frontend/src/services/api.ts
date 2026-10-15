import axios from "axios";
import type {
  InterventionProvinceDetail,
  Metadata,
  ProvinceDetail,
  ProvinceYearSummary,
  TimelinePoint,
} from "../types";
import type { DocumentItem } from "../types/documents";
import type {
  ClimateSummary,
  ClimateFactorCount,
  ClimateEvidenceItem,
  Era5Summary,
  ClimateRagCard,
} from "../types/climate";

const api = axios.create({
  baseURL: "/api",
  timeout: 20000,
});
export async function getMetadata() {
  return (await api.get<Metadata>("/metadata")).data;
}

export async function getBaseGeoJSON() {
  return (await api.get("/base-geojson")).data;
}

export async function getProvinceYearSummary(product?: string) {
  return (
    await api.get<ProvinceYearSummary[]>("/province-year-summary", {
      params: { product: product === "hazelnut" ? "findik" : (product || undefined) },
    })
  ).data;
}

export async function getTimeline(params: {
  product?: string;
  province?: string;
  timeBasis?: "event" | "published";
}) {
  return (
    await api.get<TimelinePoint[]>("/timeline", {
      params: {
        product: params.product || undefined,
        province: params.province || undefined,
        time_basis: params.timeBasis || "event",
      },
    })
  ).data;
}

export async function getInterventionTimeline(params: {
  province?: string;
  product?: string;
}) {
  return (
    await api.get<TimelinePoint[]>("/interventions/timeline", {
      params: {
        province: params.province || undefined,
        product: params.product || undefined,
      },
    })
  ).data;
}

export async function getClimateTimeline(params: {
  province?: string;
  timeBasis?: "event" | "published";
}) {
  return (
    await api.get<TimelinePoint[]>("/climate/timeline", {
      params: {
        province: params.province || undefined,
        time_basis: params.timeBasis || "published",
      },
    })
  ).data;
}

export async function getProvinceDetail(
  province: string,
  params: {
    product?: string;
    startYear?: number;
    endYear?: number;
  },
) {
  return (
    await api.get<ProvinceDetail>(`/province/${encodeURIComponent(province)}`, {
      params: {
        product: params.product || undefined,
        start_year: params.startYear,
        end_year: params.endYear,
      },
    })
  ).data;
}

export async function getInterventionProvinceDetail(
  province: string,
  params: {
    startYear?: number;
    endYear?: number;
    product?: string;
  },
) {
  return (
    await api.get<InterventionProvinceDetail>(
      `/interventions/province/${encodeURIComponent(province)}`,
      {
        params: {
          start_year: params.startYear,
          end_year: params.endYear,
          product: params.product || undefined,
        },
      },
    )
  ).data;
}

export async function getDocuments(params: {
  province?: string;
  product?: string;
  startYear?: number;
  endYear?: number;
  sourceType?: string;
  limit?: number;
}) {
  return (
    await api.get<DocumentItem[]>("/documents", {
      params: {
        province: params.province || undefined,
        product: params.product || undefined,
        start_year: params.startYear,
        end_year: params.endYear,
        source_type: params.sourceType || undefined,
        limit: params.limit ?? 100,
      },
    })
  ).data;
}
export async function getClimateSummary(params: {
  province?: string;
  product?: string;
  startYear?: number;
  endYear?: number;
}) {
  return (
    await api.get<ClimateSummary>("/climate/summary", {
      params: {
        province: params.province || undefined,
        product: params.product || undefined,
        start_year: params.startYear,
        end_year: params.endYear,
      },
    })
  ).data;
}

export async function getClimateFactors(params: {
  province?: string;
  product?: string;
  startYear?: number;
  endYear?: number;
}) {
  return (
    await api.get<ClimateFactorCount[]>("/climate/factors", {
      params: {
        province: params.province || undefined,
        product: params.product || undefined,
        start_year: params.startYear,
        end_year: params.endYear,
      },
    })
  ).data;
}

export async function getClimateEvidence(params: {
  province?: string;
  product?: string;
  startYear?: number;
  endYear?: number;
  factor?: string;
  consistency?: string;
  limit?: number;
}) {
  return (
    await api.get<ClimateEvidenceItem[]>("/climate/evidence", {
      params: {
        province: params.province || undefined,
        product: params.product || undefined,
        start_year: params.startYear,
        end_year: params.endYear,
        factor: params.factor || undefined,
        consistency: params.consistency || undefined,
        limit: params.limit ?? 100,
      },
    })
  ).data;
}

export async function getEra5Summary(params: {
  province?: string;
  product?: string;
  startYear?: number;
  endYear?: number;
}) {
  return (
    await api.get<Era5Summary>("/climate/era5", {
      params: {
        province: params.province || undefined,
        product: params.product || undefined,
        start_year: params.startYear,
        end_year: params.endYear,
      },
    })
  ).data;
}

export async function getClimateRagCard(
  province: string,
  year: number,
) {
  return (
    await api.get<ClimateRagCard | null>("/climate/rag-card", {
      params: {
        province,
        year,
      },
    })
  ).data;
}
/* =========================================================
   H. halys Temporal Animation
   MongoDB: animation_event_v2
   ========================================================= */

import type {
  AnimationMeta,
  AnimationMode,
  AnimationResponse,
} from "../types/animation";

export async function getAnimationMeta() {
  return (
    await api.get<AnimationMeta>("/animation/meta")
  ).data;
}

export async function getAnimation(params: {
  year: number;
  mode?: AnimationMode;
  province?: string;
}) {
  return (
    await api.get<AnimationResponse>("/animation", {
      params: {
        year: params.year,
        mode: params.mode ?? "annual",
        province: params.province || undefined,
      },
    })
  ).data;
}

/* =========================================================
   Final Locked Hazelnut Space-Time Cube
   ========================================================= */

import type {
  STCCell,
  STCMeta,
} from "../types/stc";

export async function getSTCMeta() {
  return (
    await api.get<STCMeta>("/stc/meta")
  ).data;
}

export async function getSTCCells(product?: string) {
  return (
    await api.get<STCCell[]>("/stc/cells", {
      params: {
        product: product === "hazelnut" ? "findik" : (product || undefined),
      },
    })
  ).data;
}

