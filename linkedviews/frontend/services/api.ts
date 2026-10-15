import axios from "axios";
import type {
  Metadata,
  ProvinceDetail,
  ProvinceYearSummary,
  TimelinePoint,
  InterventionMetadata,
  InterventionSummary,
  InterventionProvinceDetail,
} from "../types";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000/api",
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
      params: { product: product || undefined },
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
  interventionType?: string;
  institution?: string;
  product?: string;
}) {
  return (
    await api.get<TimelinePoint[]>("/interventions/timeline", {
      params: {
        province: params.province || undefined,
        intervention_type: params.interventionType || undefined,
        institution: params.institution || undefined,
        product: params.product || undefined,
      },
    })
  ).data;
}

export async function getClimateTimeline(params: {
  province?: string;
  factor?: string;
  timeBasis?: "event" | "published";
}) {
  return (
    await api.get<TimelinePoint[]>("/climate/timeline", {
      params: {
        province: params.province || undefined,
        factor: params.factor || undefined,
        time_basis: params.timeBasis || "published",
      },
    })
  ).data;
}

export async function getProvinceDetail(
  province: string,
  params: { product?: string; startYear?: number; endYear?: number },
) {
  return (
    await api.get<ProvinceDetail>(
      `/province/${encodeURIComponent(province)}`,
      {
        params: {
          product: params.product || undefined,
          start_year: params.startYear,
          end_year: params.endYear,
        },
      },
    )
  ).data;
}

export async function getInterventionMetadata() {
  return (
    await api.get<InterventionMetadata>("/interventions/metadata")
  ).data;
}

export async function getInterventionSummary(params: {
  startYear?: number;
  endYear?: number;
  interventionType?: string;
  institution?: string;
  product?: string;
}) {
  return (
    await api.get<InterventionSummary>("/interventions/summary", {
      params: {
        start_year: params.startYear,
        end_year: params.endYear,
        intervention_type: params.interventionType || undefined,
        institution: params.institution || undefined,
        product: params.product || undefined,
      },
    })
  ).data;
}

export async function getInterventionProvinceDetail(
  province: string,
  params: {
    startYear?: number;
    endYear?: number;
    interventionType?: string;
    institution?: string;
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
          intervention_type: params.interventionType || undefined,
          institution: params.institution || undefined,
          product: params.product || undefined,
        },
      },
    )
  ).data;
}
export type ClimateTimelineItem = {
  year: number;
  event_count: number;
};

export async function getClimateTimeline(params: {
  province?: string;
  factor?: string;
  timeBasis?: "event" | "published";
}) {
  const response = await api.get<ClimateTimelineItem[]>(
    "/api/climate/timeline",
    {
      params: {
        province: params.province,
        factor: params.factor,
        time_basis: params.timeBasis ?? "event",
      },
    }
  );

  return response.data;
}