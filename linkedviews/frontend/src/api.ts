import axios from "axios";
import type { Metadata, ProvinceDetail, ProvinceYearSummary, TimelinePoint } from "./types";

const api = axios.create({ baseURL: "/api" });

export async function getMetadata() {
  const { data } = await api.get<Metadata>("/metadata");
  return data;
}

export async function getBaseGeoJSON() {
  const { data } = await api.get("/base-geojson");
  return data;
}

export async function getProvinceYearSummary(product?: string) {
  const { data } = await api.get<ProvinceYearSummary[]>("/province-year-summary", { params: { product: product === "hazelnut" ? "findik" : (product || undefined) } });
  return data;
}

export async function getProvinceDetail(province: string, params: { product?: string; startYear?: number; endYear?: number }) {
  const { data } = await api.get<ProvinceDetail>(`/province/${encodeURIComponent(province)}`, {
    params: { product: params.product || undefined, start_year: params.startYear, end_year: params.endYear },
  });
  return data;
}

export async function getTimeline(params: { product?: string; province?: string }) {
  const { data } = await api.get<TimelinePoint[]>("/timeline", {
    params: { product: params.product || undefined, province: params.province || undefined },
  });
  return data;
}
import type {
  AnimationMeta,
  AnimationMode,
  AnimationResponse,
} from "./types/animation";


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



