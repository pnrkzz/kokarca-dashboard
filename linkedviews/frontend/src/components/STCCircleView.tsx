import { useMemo, useState } from "react";
import { useLinkedStore } from "../store/useLinkedStore";

type Cell = { province: string; year: number; event_count: number };
type Props = { geojson: any; cells: Cell[]; metric: string; metricLabel: string; unit: string; selectedYear: number | null; selectedProvince: string | null };
const tr = (v: unknown) => String(v ?? "").toLocaleLowerCase("tr-TR").replace(/ı/g,"i").normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9]/g, "");
function pointsOf(g: any): number[][] {
  if (!g) return [];
  if (g.type === "Polygon") return g.coordinates.flat();
  if (g.type === "MultiPolygon") return g.coordinates.flat(2);
  return [];
}
function center(g: any): [number, number] | null {
  const p = pointsOf(g).filter((v:any)=>Array.isArray(v) && Number.isFinite(v[0]) && Number.isFinite(v[1]));
  if (!p.length) return null;
  const minX=Math.min(...p.map(v=>v[0])),maxX=Math.max(...p.map(v=>v[0]));
  const minY=Math.min(...p.map(v=>v[1])),maxY=Math.max(...p.map(v=>v[1]));
  return [(minX+maxX)/2,(minY+maxY)/2];
}
const fmt=(n:number)=>new Intl.NumberFormat("tr-TR",{maximumFractionDigits:2}).format(n);
function color(v:number, metric:string, max:number) {
  if (metric.includes("anomaly")) {
    const t=Math.min(1,Math.abs(v)/Math.max(max,0.00001));
    return v < 0 ? (t>.55?"#2563eb":"#93c5fd") : v > 0 ? (t>.55?"#dc2626":"#fca5a5") : "#d1d5db";
  }
  const t=Math.max(0,Math.min(1,v/Math.max(max,1)));
  return t===0?"#94a3b8":t<.25?"#818cf8":t<.5?"#a78bfa":t<.75?"#f59e75":"#ef765e";
}
export default function STCCircleView({geojson,cells,metric,metricLabel,unit,selectedYear,selectedProvince}:Props) {
  const setSelectedProvince=useLinkedStore(s=>s.setSelectedProvince);
  const focusYear=useLinkedStore(s=>s.focusYear);
  const [hover,setHover]=useState<Cell|null>(null);
  const [showLabels,setShowLabels]=useState(true);
  const locations=useMemo(()=>{
    const m=new Map<string,[number,number]>();
    for(const f of geojson?.features??[]){
      const name=f.properties?.province_name??f.properties?.name??f.properties?.NAME_1??f.properties?.shapeName;
      const c=center(f.geometry);
      if(name&&c)m.set(tr(name),c);
    }
    return m;
  },[geojson]);
  const values=useMemo(()=>cells.map(c=>c.event_count).filter(Number.isFinite),[cells]);
  const max=Math.max(1,...values.map(Math.abs));
  const provinces=useMemo(()=>Array.from(new Set(cells.map(c=>c.province))).sort((a,b)=>a.localeCompare(b,"tr")),[cells]);
  const plotted=useMemo(()=>cells.flatMap(c=>{
    const ll=locations.get(tr(c.province));
    if(!ll||!Number.isFinite(c.event_count))return [];
    const [lon,lat]=ll;
    // Oblique projection of longitude/latitude and vertically stacked years.
    const x=88+(lon-25)*23.4+(42-lat)*9;
    const groundY=355-(lat-35)*12.5+(lon-25)*2.7;
    const y=groundY-(c.year-2017)*25;
    return [{...c,x,y,groundY,r:3.5+Math.sqrt(Math.abs(c.event_count)/max)*8}];
  }),[cells,locations,max]);
  const missing=provinces.filter(p=>!locations.has(tr(p)));
  const isAnomaly=metric.includes("anomaly");
  return <section className="stc-circle-panel" style={{minWidth:0}}>
    <div style={{display:"flex",justifyContent:"space-between",gap:12,alignItems:"center",flexWrap:"wrap",margin:"10px 0"}}>
      <strong>Yuvarlak sembollü 3B STC · {metricLabel}</strong>
      <label style={{display:"flex",gap:6,alignItems:"center"}}><input type="checkbox" checked={showLabels} onChange={e=>setShowLabels(e.target.checked)}/> İl etiketlerini göster</label>
    </div>
    <div style={{overflowX:"auto",background:"#101827",borderRadius:12,border:"1px solid #344155"}}>
      <svg viewBox="0 0 940 460" role="img" aria-label="İllerin coğrafi konumlarında 2017 ile 2025 yılları arasındaki yuvarlak sembollü mekânsal zamansal küp" style={{width:"100%",minWidth:650,display:"block"}}>
        <rect width="940" height="460" fill="#101827"/>
        {Array.from({length:9},(_,i)=>{
          const y=355-i*25;
          return <g key={i}><line x1="54" y1={y} x2="908" y2={y+22} stroke="#7e91ad" strokeOpacity={i===0?.5:.14} strokeDasharray="4 7"/><text x="14" y={y+4} fill="#cbd5e1" fontSize="13">{2017+i}</text></g>;
        })}
        <text x="14" y="95" fill="#e2e8f0" fontSize="13" transform="rotate(-90 14 95)">Zaman</text>
        {plotted.filter(c=>c.year===2017).map((c,i)=><line key={i} x1={c.x} y1={c.groundY} x2={c.x} y2={c.groundY-200} stroke="#cbd5e1" strokeOpacity=".11" strokeDasharray="2 5"/>)}
        {plotted.map((c,i)=>{
          const active=(!selectedProvince||tr(selectedProvince)===tr(c.province))&&(!selectedYear||selectedYear===c.year);
          return <circle key={`${c.province}-${c.year}-${i}`} cx={c.x} cy={c.y} r={c.r} fill={color(c.event_count,metric,max)} fillOpacity={active?.94:.2} stroke={hover?.province===c.province&&hover.year===c.year?"#fff":"#0f172a"} strokeWidth={hover?.province===c.province&&hover.year===c.year?2.5:1} style={{cursor:"pointer"}} onMouseEnter={()=>setHover(c)} onMouseLeave={()=>setHover(null)} onClick={()=>{setSelectedProvince(c.province);focusYear(c.year);}}><title>{`${c.province} · ${c.year}: ${fmt(c.event_count)} ${unit}`}</title></circle>;
        })}
        {showLabels&&provinces.map(p=>{
          const ll=locations.get(tr(p));if(!ll)return null;
          const [lon,lat]=ll;
          const x=88+(lon-25)*23.4+(42-lat)*9, y=355-(lat-35)*12.5+(lon-25)*2.7;
          return <text key={p} x={x} y={y+20} fill="#e2e8f0" fontSize="11" textAnchor="middle" paintOrder="stroke" stroke="#101827" strokeWidth="3">{p}</text>;
        })}
        <text x="710" y="438" fill="#cbd5e1" fontSize="12">İllerin yaklaşık coğrafi konumları</text>
      </svg>
    </div>
    <div className="stc-circle-insights" aria-label="Grafik değerlendirmesi">
      <div className="stc-circle-stat"><span aria-hidden="true">📍</span><div><strong>{provinces.length} il</strong><small>Coğrafi kapsam</small></div></div>
      <div className="stc-circle-stat"><span aria-hidden="true">🗓️</span><div><strong>2017–2025</strong><small>9 yıllık dönem</small></div></div>
      <div className="stc-circle-stat"><span aria-hidden="true">🔵</span><div><strong>{cells.filter(c=>Number.isFinite(c.event_count)&&c.event_count!==0).length} hücre</strong><small>Sıfırdan farklı değer</small></div></div>
      <div className="stc-circle-stat"><span aria-hidden="true">ⓘ</span><div><strong>{isAnomaly?"Mavi − / Kırmızı +":"Mor düşük / Turuncu yüksek"}</strong><small>Renk ölçeği</small></div></div>
    </div>
    <div className="stc-circle-interpretation">
      <strong>Grafik sonucu ve yorumu</strong>
      <p>{(()=>{
        const valid=cells.filter(c=>Number.isFinite(c.event_count));
        if(!valid.length)return "Bu gösterge için yorumlanabilecek sayısal kayıt bulunamadı.";
        const nonzero=valid.filter(c=>c.event_count!==0);
        const peak=[...valid].sort((a,b)=>Math.abs(b.event_count)-Math.abs(a.event_count))[0];
        const peakText=`En büyük mutlak değer ${peak.province}, ${peak.year}: ${fmt(peak.event_count)} ${unit}.`;
        if(isAnomaly){
          const negative=valid.filter(c=>c.event_count<0).length, positive=valid.filter(c=>c.event_count>0).length;
          return `${valid.length} il–yıl hücresinin ${negative} tanesinde negatif, ${positive} tanesinde pozitif anomali vardır. ${peakText} Renk anomalinin yönünü, çember büyüklüğü mutlak değerini gösterir. Bu karşılaştırma tek başına iklim değişikliğine atıf değildir.`;
        }
        const total=valid.reduce((s,c)=>s+c.event_count,0);
        const note=metric==="event_count"?"Bu değerler metinlerden çıkarılmış olay kayıtlarıdır; böcek popülasyonu veya bağımsız varlık gözlemi değildir.":metric==="textual_climate_relation_count"?"İklim anlatıları kaynaklarda bildirilen ilişkileri gösterir; nedensellik kanıtı değildir.":metric==="intervention_count"?"Müdahale kayıtları, uygulamanın etkinliğini tek başına ölçmez.":"Değerler seçili göstergenin il–yıl dağılımını ifade eder.";
        return `${valid.length} il–yıl hücresinin ${nonzero.length} tanesinde sıfırdan büyük değer bulunuyor; gösterge toplamı ${fmt(total)} ${unit}. ${peakText} ${note}`;
      })()}</p>
      <p className="stc-circle-method">Coğrafi konumlar yaklaşık merkezlerle şematik olarak gösterilmiştir; düşey eksen 2017–2025 yıllarını temsil eder. Kaydı olmayan hücre ile sıfır değeri aynı kabul edilmez.</p>
    </div>
    <div style={{display:"flex",flexWrap:"wrap",alignItems:"center",gap:14,marginTop:10,fontSize:13}}>
      {isAnomaly?<><span style={{color:"#93c5fd"}}>● Negatif anomali</span><span style={{color:"#d1d5db"}}>● Sıfıra yakın</span><span style={{color:"#fca5a5"}}>● Pozitif anomali</span></>:<><span style={{color:"#818cf8"}}>● Düşük</span><span style={{color:"#a78bfa"}}>● Orta</span><span style={{color:"#ef765e"}}>● Yüksek</span></>}
      <span>Çember büyüklüğü: {isAnomaly?"anomali mutlak büyüklüğü":"gösterge değeri"}</span>
    </div>
    <p style={{margin:"10px 0"}}>{hover?<><strong>{hover.province} · {hover.year}</strong> — {metricLabel}: <strong>{fmt(hover.event_count)} {unit}</strong>. Tıklayarak il ve yılı seçebilirsiniz.</>:"Ayrıntı için sembolün üzerine gelin; il ve yılı diğer görünümlerde filtrelemek için tıklayın."}</p>
    {missing.length>0?<p role="alert">Konumu eşleşmeyen iller: {missing.join(", ")}</p>:null}
  </section>;
}
