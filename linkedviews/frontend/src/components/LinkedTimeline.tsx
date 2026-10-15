import type { Metadata, TimelinePoint, TimeLayer } from "../types";
import { useLinkedStore } from "../store/useLinkedStore";

type Props={metadata:Metadata|null;eventData:TimelinePoint[];publishedData:TimelinePoint[];interventionData:TimelinePoint[];climateData:TimelinePoint[]};
function completeSeries(years:number[],data:TimelinePoint[]){const m=new Map(data.map(r=>[r.year,r.event_count]));return years.map(year=>({year,event_count:m.get(year)||0}));}
const labels:Record<TimeLayer,string>={event:"Olay",published:"Yayım",intervention:"Müdahale",climate:"İklim",compare:"Birleşik"};
export default function LinkedTimeline({metadata,eventData,publishedData,interventionData,climateData}:Props){
  const layer=useLinkedStore(s=>s.timeLayer); const setLayer=useLinkedStore(s=>s.setTimeLayer); const province=useLinkedStore(s=>s.selectedProvince); const focusYear=useLinkedStore(s=>s.focusYear); const selectedYear=useLinkedStore(s=>s.selectedYear); const mode=useLinkedStore(s=>s.timeMode);
  const years=metadata?.years??[]; const ev=completeSeries(years,eventData), pub=completeSeries(years,publishedData), intv=completeSeries(years,interventionData), cli=completeSeries(years,climateData);
  const cur=layer==="published"?pub:layer==="intervention"?intv:layer==="climate"?cli:ev; const max=Math.max(...cur.map(x=>x.event_count),1); const maxC=Math.max(...ev.map(x=>x.event_count),...pub.map(x=>x.event_count),1); const total=(d:TimelinePoint[])=>d.reduce((a,b)=>a+b.event_count,0);
  return <section className="timeline-panel">
    <div className="timeline-head"><div><span className="eyebrow">BAĞLI ZAMAN GÖRÜNÜMÜ</span><h3>Zaman</h3><p>{province||"Türkiye geneli"} · Bir yıla tıklayınca harita o yıla odaklanır.</p></div><div className="timeline-tabs">{(Object.keys(labels) as TimeLayer[]).map(x=><button key={x} className={layer===x?"selected":""} onClick={()=>setLayer(x)}>{labels[x]}</button>)}</div></div>
    <div className="timeline-plot">{years.map((year,i)=>{const sel=mode==="single"&&selectedYear===year;if(layer==="compare"){const a=ev[i]?.event_count||0,b=pub[i]?.event_count||0;return <button className={`timeline-year ${sel?"selected-year":""}`} key={year} onClick={()=>focusYear(year)} title={`${year} · Olay ${a} · Yayım ${b}`}><div className="compare-column"><div className="time-bar event-bar" style={{height:`${Math.max(4,(a/maxC)*88)}px`}}/><div className="time-bar published-bar" style={{height:`${Math.max(4,(b/maxC)*88)}px`}}/></div><span>{year}</span></button>} const v=cur[i]?.event_count||0; const cls=layer==="published"?"published-bar":layer==="intervention"?"intervention-bar":layer==="climate"?"climate-bar":"event-bar"; return <button className={`timeline-year ${sel?"selected-year":""}`} key={year} onClick={()=>focusYear(year)} title={`${year} · ${v}`}><div className={`time-bar wide ${cls}`} style={{height:`${Math.max(4,(v/max)*88)}px`}}/><span>{year}</span></button>})}</div>
    <div className="timeline-footer"><span>Olay: {total(eventData)}</span><span>Yayım: {total(publishedData)}</span><span>Müdahale: {total(interventionData)}</span><span>İklim: {total(climateData)}</span></div>
  </section>;
}
