import type { TimelinePoint, TimeLayer } from "../types";

type Props = {
  years: number[];
  eventData: TimelinePoint[];
  publishedData: TimelinePoint[];
  interventionData: TimelinePoint[];
  climateData: TimelinePoint[];
  activeStart: number;
  activeEnd: number;
  title: string;
  layer: TimeLayer;
  onLayerChange: (layer: TimeLayer) => void;
};

function complete(years: number[], data: TimelinePoint[]) {
  const lookup = new Map(data.map((x) => [x.year, x.event_count]));
  return years.map((year) => ({ year, event_count: lookup.get(year) || 0 }));
}

export default function Timeline(props: Props) {
  const {
    years, eventData, publishedData, interventionData, climateData,
    activeStart, activeEnd, title, layer, onLayerChange,
  } = props;

  const eventFull = complete(years, eventData);
  const publishedFull = complete(years, publishedData);
  const interventionFull = complete(years, interventionData);
  const climateFull = complete(years, climateData);

  const current =
    layer === "published" ? publishedFull :
    layer === "intervention" ? interventionFull :
    layer === "climate" ? climateFull : eventFull;

  const maxSingle = Math.max(...current.map((x) => x.event_count), 1);
  const maxCompare = Math.max(
    ...eventFull.map((x) => x.event_count),
    ...publishedFull.map((x) => x.event_count),
    1,
  );

  return (
    <footer className="timeline timeline-expanded">
      <div className="timeline-head">
        <div className="timeline-title">
          <strong>Zaman Görünümü</strong>
          <span>{title}</span>
        </div>

        <div className="timeline-tabs">
          <button className={layer === "event" ? "selected" : ""} onClick={() => onLayerChange("event")}>Olay Yılı</button>
          <button className={layer === "published" ? "selected" : ""} onClick={() => onLayerChange("published")}>Yayım Yılı</button>
          <button className={layer === "intervention" ? "selected" : ""} onClick={() => onLayerChange("intervention")}>Müdahale Yılı</button>
          <button className={layer === "climate" ? "selected" : ""} onClick={() => onLayerChange("climate")}>İklim Anlatısı</button>
          <button className={layer === "compare" ? "selected" : ""} onClick={() => onLayerChange("compare")}>Olay + Yayım</button>
        </div>
      </div>

      <div className="timeline-bars timeline-bars-wide">
        {years.map((year, idx) => {
          const active = year >= activeStart && year <= activeEnd;

          if (layer === "compare") {
            const eventValue = eventFull[idx]?.event_count || 0;
            const publishedValue = publishedFull[idx]?.event_count || 0;
            return (
              <div className={`timeline-item ${active ? "active-year" : ""}`} key={year} title={`${year} | Olay: ${eventValue} | Yayım: ${publishedValue}`}>
                <div className="compare-bars">
                  <div className="bar bar-event" style={{height: `${Math.max(3, (eventValue / maxCompare) * 54)}px`}} />
                  <div className="bar bar-published" style={{height: `${Math.max(3, (publishedValue / maxCompare) * 54)}px`}} />
                </div>
                <span>{year}</span>
              </div>
            );
          }

          const value = current[idx]?.event_count || 0;
          return (
            <div className={`timeline-item ${active ? "active-year" : ""}`} key={year} title={`${year}: ${value}`}>
              <div className="bar" style={{height: `${Math.max(4, (value / maxSingle) * 54)}px`}} />
              <span>{year}</span>
            </div>
          );
        })}
      </div>

      {layer === "compare" && (
        <div className="timeline-legend">
          <span><i className="legend-box legend-event" /> Olay yılı</span>
          <span><i className="legend-box legend-published" /> Yayım yılı</span>
        </div>
      )}
    </footer>
  );
}
