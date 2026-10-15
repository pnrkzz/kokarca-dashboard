import { useEffect, useMemo, useState } from "react";
import { getBaseGeoJSON } from "../services/api";
import { useLinkedStore } from "../store/useLinkedStore";
import STC3DView from "./STC3DView";
import STCCircleView from "./STCCircleView";
import type { STCCell, STCMeta } from "../types/stc";

type Row = Record<string, string | number | null> & { province: string; year: number };
type Response = { period: { start_year: number; end_year: number }; scope: string; rows: Row[] };
type Metric = "event_count" | "textual_climate_relation_count" | "intervention_count" | "annual_temperature_anomaly_c" | "annual_precip_anomaly_mm";
const metrics: { key: Metric; label: string; unit: string; explanation: string }[] = [
  {key:"event_count",label:"Olay yoğunluğu",unit:"olay kaydı",explanation:"Metinlerden çıkarılan il–yıl olay kayıtlarının sayısıdır. Doğrulanmış birey sayısı, böcek popülasyonu veya bağımsız occurrence sayısı değildir. Fazla kayıt, kaynakların haber üretme yoğunluğundan da etkilenebilir."},
  {key:"textual_climate_relation_count",label:"İklim anlatıları",unit:"anlatı kaydı",explanation:"Kaynak metinlerde H. halys ile iklim veya hava koşullarının birlikte ele alındığı kayıtların sayısıdır. Tek başına nedensellik veya iklim değişikliği etkisinin kanıtı değildir."},
  {key:"intervention_count",label:"Devlet müdahaleleri",unit:"müdahale kaydı",explanation:"Resmî kurumların mücadele, izleme, eğitim veya benzeri faaliyetlerine ilişkin metin tabanlı kayıtları gösterir. Müdahale sayısı, etkinlik veya başarı düzeyini doğrudan ölçmez."},
  {key:"annual_temperature_anomaly_c",label:"ERA5 sıcaklık anomalisi",unit:"°C",explanation:"İl–yıl ortalama sıcaklığının kullanılan referans dönemi ortalamasından sapmasıdır. Negatif değer daha serin, pozitif değer daha sıcak koşulları gösterir. Tek bir yılın anomalisi uzun dönemli iklim değişikliği eğilimi değildir."},
  {key:"annual_precip_anomaly_mm",label:"ERA5 yağış anomalisi",unit:"mm",explanation:"İl–yıl toplam yağışının kullanılan referans dönemi ortalamasından sapmasıdır. Negatif değer daha az, pozitif değer daha fazla yağış anlamına gelir. Anomalinin işareti korunur."}
];
const format=(n:number)=>new Intl.NumberFormat("tr-TR",{maximumFractionDigits:2}).format(n);
export default function STCView() {
  const selectedProvince=useLinkedStore(s=>s.selectedProvince);
  const selectedYear=useLinkedStore(s=>s.selectedYear);
  const setSelectedProvince=useLinkedStore(s=>s.setSelectedProvince);
  const focusYear=useLinkedStore(s=>s.focusYear);
  const [rows,setRows]=useState<Row[]>([]);
  const [geojson,setGeojson]=useState<any>(null);
  const [error,setError]=useState("");
  const [geoError,setGeoError]=useState("");
  const [loadingRows,setLoadingRows]=useState(true);
  const [loadingGeo,setLoadingGeo]=useState(true);
  const [reloadKey,setReloadKey]=useState(0);
  const [metric,setMetric]=useState<Metric>("event_count");
  const [mode,setMode]=useState<"circles"|"3d"|"2d">("circles");
  useEffect(()=>{
    let active=true;
    setError("");setGeoError("");setLoadingRows(true);setLoadingGeo(true);
    // Load independently: a failed/slow GeoJSON request must not discard STC rows.
    fetch("/api/stc/multilayer")
      .then(async r=>{if(!r.ok)throw new Error(`STC API: ${r.status}`);return r.json() as Promise<Response>;})
      .then(data=>{if(active)setRows(data.rows.filter(r=>r.year>=2017&&r.year<=2025));})
      .catch(e=>{if(active)setError(`STC verisi alınamadı: ${String(e)}`);})
      .finally(()=>{if(active)setLoadingRows(false);});
    getBaseGeoJSON()
      .then(geo=>{if(active)setGeojson(geo);})
      .catch(e=>{if(active)setGeoError(`İl sınırları yüklenemedi: ${String(e)}`);})
      .finally(()=>{if(active)setLoadingGeo(false);});
    return ()=>{active=false;};
  },[reloadKey]);
  const years=Array.from({length:9},(_,i)=>2017+i);
  const provinces=useMemo(()=>Array.from(new Set(rows.map(r=>r.province))).sort((a,b)=>a.localeCompare(b,"tr")),[rows]);
  const cells=useMemo<STCCell[]>(()=>rows.filter(r=>r[metric]!==null&&r[metric]!==undefined&&r[metric]!==""&&Number.isFinite(Number(r[metric]))).map(r=>({province:r.province,year:r.year,event_count:Number(r[metric])})),[rows,metric]);
  const maxCount=Math.max(1,...cells.map(c=>Math.abs(c.event_count)));
  const meta=useMemo<STCMeta>(()=>({dataset:"Fındık odaklı il–yıl ortak veri tablosu",status:"ok",product:"fındık",spatial_unit:"province",temporal_unit:"year",cell_count:cells.length,province_count:provinces.length,year_count:9,min_year:2017,max_year:2025,total_event_count:cells.reduce((s,c)=>s+Math.abs(c.event_count),0),max_cell_count:maxCount,years,provinces,yearly_totals:{}}),[cells,provinces,maxCount]);
  const lookup=useMemo(()=>new Map(rows.map(r=>[`${r.province}::${r.year}`,r])),[rows]);
  const selected=metrics.find(m=>m.key===metric)!;
  const anomaly=metric.includes("anomaly");
  const fmt=(r:Row|undefined)=>{if(!r)return "—";const value=r[metric];return value===null||value===undefined||value===""?"Veri yok":format(Number(value));};
  return <div className="stc-view" style={{color:"#e2e8f0"}}>
    <header className="stc-header"><div><div className="stc-eyebrow">SPACE–TIME CUBE · 2017–2025</div><h2 style={{color:"#f8fafc"}}>Olay · İklim · Devlet müdahaleleri</h2><p>Fındık odaklı il–yıl veri tablosu. 2026 kapsam dışıdır. Verisi olmayan hücreler sıfır kabul edilmez.</p></div></header>
    {error?<p role="alert">{error} <button type="button" onClick={()=>setReloadKey(k=>k+1)}>Yeniden dene</button></p>:null}
    {loadingRows?<p>STC verileri yükleniyor...</p>:null}
    {!loadingRows&&!rows.length&&!error?<p>Seçilen dönemde STC kaydı bulunamadı.</p>:null}
    {geoError?<p role="alert">{geoError} 2B matris kullanılabilir. <button type="button" onClick={()=>setReloadKey(k=>k+1)}>Yeniden dene</button></p>:null}
    <section className="stc-panel" style={{padding:16,minWidth:0}}>
      <div style={{display:"flex",gap:12,flexWrap:"wrap",alignItems:"center",marginBottom:14}}>
        <label>Gösterge: <select value={metric} onChange={e=>setMetric(e.target.value as Metric)}>{metrics.map(m=><option key={m.key} value={m.key}>{m.label}</option>)}</select></label>
        <button type="button" className="stc-mode-button" onClick={()=>setMode("circles")} aria-pressed={mode==="circles"}>● Yuvarlak 3B</button>
        <button type="button" className="stc-mode-button" onClick={()=>setMode("3d")} aria-pressed={mode==="3d"}>▥ Haritalı 3B</button>
        <button type="button" className="stc-mode-button" onClick={()=>setMode("2d")} aria-pressed={mode==="2d"}>▦ 2B matris</button>
        <span>{provinces.length} il · 2017–2025</span>
      </div>
      <details className="stc-explanation" style={{marginBottom:10,fontSize:13}}>
        <summary style={{cursor:"pointer",width:"fit-content"}} title="Gösterge ve yöntem açıklaması">ⓘ Gösterge hakkında</summary>
        <p>{selected.explanation}</p>
        <p>{anomaly?"Mavi negatif, kırmızı pozitif anomalidir; renk yoğunluğu mutlak sapmayı gösterir.":"Renk yoğunluğu seçili göstergenin büyüklüğünü gösterir."} Düşey eksen 2017–2025 yıllarını temsil eder.</p>
      </details>
      {(mode==="circles"||mode==="3d")&&!geojson&&rows.length>0?<p role="status">{loadingGeo?"İl sınırları yükleniyor; beklerken 2B matrisi kullanabilirsiniz.":"İl sınırları hazır değil; 2B matrisi kullanabilir veya yeniden deneyebilirsiniz."}</p>:null}
      {mode==="circles"&&geojson&&rows.length?<STCCircleView geojson={geojson} cells={cells} metric={metric} metricLabel={selected.label} unit={selected.unit} selectedProvince={selectedProvince} selectedYear={selectedYear}/>:null}
      {mode==="3d"&&geojson&&rows.length?<STC3DView geojson={geojson} cells={cells} meta={meta} metric={metric} metricLabel={selected.label} unit={selected.unit}/>:null}
      {mode==="2d"?<div className="stc-table-wrap"><table className="stc-table"><thead><tr><th>İl</th>{years.map(y=><th key={y}><button onClick={()=>focusYear(y)}>{y}</button></th>)}</tr></thead><tbody>{provinces.map(p=><tr key={p}><th><button onClick={()=>setSelectedProvince(selectedProvince===p?null:p)}>{p}</button></th>{years.map(y=>{const r=lookup.get(`${p}::${y}`);const value=r?.[metric];const valid=value!==null&&value!==undefined&&value!=="";return <td key={y}><button className="stc-cell" title={`${p} ${y} · ${selected.label}`} style={{background:valid?(anomaly?(Number(value)<0?"#2563eb":Number(value)>0?"#b45353":"#64748b"):`rgba(139,92,246,${Math.max(.12,Math.abs(Number(value))/maxCount)})`):"transparent",color:"#fff"}} onClick={()=>{setSelectedProvince(p);focusYear(y);}}>{fmt(r)}</button></td>;})}</tr>)}</tbody></table></div>:null}
      <p style={{marginTop:12}}>Seçim: {selectedProvince||"Tüm iller"} · {selectedYear||"Tüm yıllar"} · Ölçü birimi: {selected.unit}. Sembol veya tablo hücresine tıklayarak koordineli görünümleri filtreleyebilirsiniz.</p>
      <p style={{fontSize:12,opacity:.85}}>Kaynak türleri: akademik yayınlar, resmî belgeler, ulusal/yerel haberler ve YouTube. Gösterilen toplulaştırılmış sayılar tek tek kaynak kanıtlarının yerini tutmaz; belge düzeyinde doğrulama için kaynak bağlantıları korunmalıdır.</p>
    </section>
  </div>;
}
