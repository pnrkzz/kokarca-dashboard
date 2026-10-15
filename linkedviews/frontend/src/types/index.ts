export type Metadata = {
  event_count: number;
  temporal_event_count: number;
  document_count: number;
  province_count: number;
  temporal_province_count: number;
  years: number[];
  published_years?: number[];
  products: string[];
  product_column: string | null;
  geojson_name_key: string;
};

export type ProvinceYearSummary = {
  province: string;
  year: number;
  event_count: number;
};

export type TimelinePoint = {
  year: number;
  event_count: number;
};

export type SourceItem = {
  title: string | null;
  url: string | null;
  source_type: string | null;
};

export type ProvinceDetail = {
  province: string;
  event_count: number;
  document_count: number;
  first_year: number | null;
  last_year: number | null;
  first_published_year: number | null;
  last_published_year: number | null;
  sources: SourceItem[];
};

export type NamedCount = {
  name: string;
  count: number;
};

export type InterventionMetadata = {
  record_count: number;
  dated_record_count: number;
  years: number[];
  intervention_types: string[];
  institutions: string[];
};

export type InterventionSummary = {
  record_count: number;
  province_counts: { province: string; count: number }[];
  year_counts: { year: number; count: number }[];
  type_counts: NamedCount[];
  institution_counts: NamedCount[];
};

export type InterventionProvinceDetail = {
  province: string;
  record_count: number;
  years: number[];
  types: NamedCount[];
  institutions: NamedCount[];
  actions: NamedCount[];
  actors: NamedCount[];
  sources: SourceItem[];
};


export type TimeLayer =
  | "event"
  | "published"
  | "intervention"
  | "climate"
  | "compare";
