export type ClimateSummary = {
  record_count: number;
  climate_narrative_count: number;
  era5_testable_count: number;
  consistent_count: number;
  partially_consistent_count: number;
  not_consistent_count: number;
  observed_anomaly_count: number;
};

export type ClimateFactorCount = {
  factor: string;
  count: number;
};

export type ClimateEvidenceItem = {
  event_id: string | null;
  global_id: string | null;
  province: string | null;
  climate_year: number | null;
  spread_observation_year: number | null;
  source_type: string | null;
  published_date: string | null;
  factor: string | null;
  factor_raw: string | null;
  direction: string | null;
  evidence_sentence: string | null;
  climate_comment: string | null;
  era5_consistency: string | null;
  consistency_reason: string | null;
  annual_temperature_anomaly: number | null;
  winter_temperature_anomaly: number | null;
  spring_temperature_anomaly: number | null;
  summer_temperature_anomaly: number | null;
  autumn_temperature_anomaly: number | null;
  relative_humidity_anomaly: number | null;
  precipitation_anomaly: number | null;
  validation_flag: string | null;
};

export type Era5Summary = {
  record_count: number;
  temperature_anomaly_mean: number | null;
  winter_temperature_anomaly_mean: number | null;
  humidity_anomaly_mean: number | null;
  precipitation_anomaly_mean: number | null;
};

export type ClimateRagCard = Record<string, string | number | null>;
