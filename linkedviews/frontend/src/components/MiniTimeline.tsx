import type { TimelinePoint } from "../types";

export default function MiniTimeline({ data }: { data: TimelinePoint[] }) {
  if (!data.length) return <p className="muted-copy">Zamansal kayıt yok.</p>;
  const max = Math.max(...data.map(x=>x.event_count),1);
  return <div className="mini-timeline">{data.map(item=><div className="mini-year" key={item.year} title={`${item.year}: ${item.event_count}`}><div className="mini-bar" style={{height:`${Math.max(5,(item.event_count/max)*52)}px`}}/><span>{item.year}</span></div>)}</div>;
}
