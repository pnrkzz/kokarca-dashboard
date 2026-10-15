import type { InterventionProvinceDetail, ProvinceDetail, TimelinePoint } from "../types";
import { useLinkedStore } from "../store/useLinkedStore";
import MiniTimeline from "./MiniTimeline";

type Props = { provinceDetail: ProvinceDetail | null; interventionDetail: InterventionProvinceDetail | null; eventTimeline: TimelinePoint[]; publishedTimeline: TimelinePoint[]; climateTimeline: TimelinePoint[]; loading: boolean };

export default function ContextPanel({ provinceDetail, interventionDetail, eventTimeline, publishedTimeline, climateTimeline, loading }: Props) {
  const province = useLinkedStore((s) => s.selectedProvince);
  const setProvince = useLinkedStore((s) => s.setSelectedProvince);
  if (!province) return <aside className="context-panel empty-context"><div className="empty-orbit">⌖</div><h3>Bir il seçin</h3><p>Haritada bir ile tıkladığınızda olay, yayın, müdahale ve zaman görünümleri aynı bağlamda güncellenir.</p></aside>;
  const sum=(d:TimelinePoint[])=>d.reduce((a,b)=>a+b.event_count,0);
  return <aside className="context-panel">
    <div className="context-panel-head"><div><span className="eyebrow">SEÇİLİ MEKÂN</span><h2>{province}</h2></div><button className="clear-button" onClick={()=>setProvince(null)}>×</button></div>
    {loading ? <div className="panel-loader"><span/>Bağlı görünümler güncelleniyor…</div> : <>
      <div className="context-metrics">
        <div><span>Olay</span><strong>{provinceDetail?.event_count ?? 0}</strong></div><div><span>Belge</span><strong>{provinceDetail?.document_count ?? 0}</strong></div>
        <div><span>İlk olay</span><strong>{provinceDetail?.first_year ?? "—"}</strong></div><div><span>Son olay</span><strong>{provinceDetail?.last_year ?? "—"}</strong></div>
        <div><span>İlk yayım</span><strong>{provinceDetail?.first_published_year ?? "—"}</strong></div><div><span>Son yayım</span><strong>{provinceDetail?.last_published_year ?? "—"}</strong></div>
      </div>
      <div className="linked-section"><div className="linked-section-head"><h4>Olay zaman çizgisi</h4><span>{sum(eventTimeline)}</span></div><MiniTimeline data={eventTimeline}/></div>
      <div className="linked-section two-columns">
        <div><div className="linked-section-head"><h4>Müdahaleler</h4><span>{interventionDetail?.record_count ?? 0}</span></div><div className="mini-rank-list">{(interventionDetail?.types ?? []).slice(0,5).map(i=><div key={i.name}><span>{i.name}</span><strong>{i.count}</strong></div>)}</div></div>
        <div><div className="linked-section-head"><h4>Kurumlar</h4><span>{interventionDetail?.institutions?.length ?? 0}</span></div><div className="mini-rank-list">{(interventionDetail?.institutions ?? []).slice(0,5).map(i=><div key={i.name}><span>{i.name}</span><strong>{i.count}</strong></div>)}</div></div>
      </div>
      <div className="linked-section"><div className="linked-section-head"><h4>Bağlam özeti</h4></div><div className="summary-chips"><span>Yayım {sum(publishedTimeline)}</span><span>İklim {sum(climateTimeline)}</span><span>Müdahale {interventionDetail?.record_count ?? 0}</span></div></div>
      <div className="linked-section"><div className="linked-section-head"><h4>Kaynak önizlemesi</h4><span>{provinceDetail?.sources?.length ?? 0}</span></div><div className="source-preview-list">{(provinceDetail?.sources ?? []).filter(s=>s.url).slice(0,5).map((s,i)=><a key={`${s.url}-${i}`} href={s.url!} target="_blank" rel="noreferrer"><span>{s.source_type || "Kaynak"}</span><strong>{s.title || "Belgeyi aç"}</strong></a>)}</div></div>
    </>}
  </aside>;
}
