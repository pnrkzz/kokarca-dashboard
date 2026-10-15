export type STCCell = {
  province: string;
  year: number;
  event_count: number;
};

export type STCMeta = {
  dataset: string;
  status: string;
  product: string;
  spatial_unit: string;
  temporal_unit: string;
  cell_count: number;
  province_count: number;
  year_count: number;
  min_year: number;
  max_year: number;
  total_event_count: number;
  max_cell_count: number;
  years: number[];
  provinces: string[];
  yearly_totals: Record<string, number>;
};
