export type AnimationMode = "annual" | "cumulative";

export type AnimationType =
  | "first_detection"
  | "direct_observation"
  | "spread_destination"
  | "research_observation"
  | "spread_observation";

export type GeometryPrecision =
  | "PROVINCE"
  | "DISTRICT"
  | "PROVINCE_FALLBACK";

export interface AnimationGeometry {
  type: "Point";
  coordinates: [number, number];
}

export interface AnimationFeature {
  province: string | null;
  district: string | null;
  locality: string | null;

  location_role: string;

  event_uid: string;
  location_uid: string;

  geometry_match_status: string;
  geometry_precision: GeometryPrecision;

  latitude: number;
  longitude: number;

  source_type: string;
  final_event_role: string;

  animation_year: number;
  animation_type: AnimationType;
  animation_uid: string;

  display_location: string;

  geometry: AnimationGeometry;
}

export interface AnimationResponse {
  year: number;
  mode: AnimationMode;
  province: string | null;
  count: number;
  features: AnimationFeature[];
}

export interface AnimationMeta {
  min_year: number;
  max_year: number;
  record_count: number;
  event_count: number;

  year_counts: Record<string, number>;
  type_counts: Record<string, number>;

  available_modes: AnimationMode[];

  coordinate_interpretation: string;
}