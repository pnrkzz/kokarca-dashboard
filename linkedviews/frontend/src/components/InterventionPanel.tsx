import type {
  InterventionMetadata,
  InterventionProvinceDetail,
  InterventionSummary,
} from "../types";

type Props = {
  meta: InterventionMetadata | null;
  summary: InterventionSummary | null;
  detail: InterventionProvinceDetail | null;
  startYear?: number;
  endYear?: number;
  interventionType: string;
  institution: string;
  onStartYear: (value?: number) => void;
  onEndYear: (value?: number) => void;
  onType: (value: string) => void;
  onInstitution: (value: string) => void;
};

export default function InterventionPanel(props: Props) {
  const {
    meta,
    summary,
    startYear,
    endYear,
    interventionType,
    institution,
    onStartYear,
    onEndYear,
    onType,
    onInstitution,
  } = props;

  return (
    <>
      <div className="intervention-filterbar">
        <label>
          BaÅŸlangÄ±Ã§ yÄ±lÄ±
          <select
            value={startYear ?? ""}
            onChange={(e) =>
              onStartYear(e.target.value ? Number(e.target.value) : undefined)
            }
          >
            <option value="">TÃ¼mÃ¼</option>
            {meta?.years.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </label>

        <label>
          BitiÅŸ yÄ±lÄ±
          <select
            value={endYear ?? ""}
            onChange={(e) =>
              onEndYear(e.target.value ? Number(e.target.value) : undefined)
            }
          >
            <option value="">TÃ¼mÃ¼</option>
            {meta?.years.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </label>

        <label>
          MÃ¼dahale tÃ¼rÃ¼
          <select
            value={interventionType}
            onChange={(e) => onType(e.target.value)}
          >
            <option value="">TÃ¼m tÃ¼rler</option>
            {meta?.intervention_types.map((x) => (
              <option key={x} value={x}>{x}</option>
            ))}
          </select>
        </label>

        <label>
          Kurum
          <select
            value={institution}
            onChange={(e) => onInstitution(e.target.value)}
          >
            <option value="">TÃ¼m kurumlar</option>
            {meta?.institutions.map((x) => (
              <option key={x} value={x}>{x}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="intervention-dashboard">
        <div className="analytics-card">
          <h3>YÄ±llara GÃ¶re MÃ¼dahale</h3>
          {summary?.year_counts.map((x) => (
            <div className="rank-row" key={x.year}>
              <span>{x.year}</span>
              <strong>{x.count}</strong>
            </div>
          ))}
        </div>

        <div className="analytics-card">
          <h3>En SÄ±k MÃ¼dahale TÃ¼rleri</h3>
          {summary?.type_counts.slice(0, 8).map((x) => (
            <div className="rank-row" key={x.name}>
              <span>{x.name}</span>
              <strong>{x.count}</strong>
            </div>
          ))}
        </div>

        <div className="analytics-card">
          <h3>En Aktif Kurumlar</h3>
          {summary?.institution_counts.slice(0, 8).map((x) => (
            <div className="rank-row" key={x.name}>
              <span>{x.name}</span>
              <strong>{x.count}</strong>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

