import { useEffect, useMemo, useState } from "react";
import MapView from "./MapView";
import { getBaseGeoJSON } from "../services/api";
import { useLinkedStore } from "../store/useLinkedStore";

type Relation = { id: string; province: string | null };
type RelationsResponse = { total: number; items: Relation[]; notice: string };

export default function ClimateMap({ startYear, endYear }: { startYear: number; endYear: number }) {
  const province = useLinkedStore(s => s.selectedProvince);
  const setProvince = useLinkedStore(s => s.setSelectedProvince);
  const product = useLinkedStore(s => s.selectedProduct);
  const [geo, setGeo] = useState<any>(null);
  const [relations, setRelations] = useState<RelationsResponse | null>(null);
  const [error, setError] = useState("");
  useEffect(() => { getBaseGeoJSON().then(setGeo).catch(e => setError(String(e))); }, []);
  useEffect(() => {
    const controller = new AbortController();
    const q = new URLSearchParams({start_year:String(startYear), end_year:String(endYear),limit:"1000"});
    if (product) q.set("product", product);
    setRelations(null); setError("");
    fetch(`/api/climate/relations?${q}`, {signal:controller.signal})
      .then(async r => { if (!r.ok) throw new Error(`İklim API: ${r.status}`); return r.json() as Promise<RelationsResponse>; })
      .then(setRelations)
      .catch(e => { if (!controller.signal.aborted) setError(String(e)); });
    return () => controller.abort();
  }, [startYear,endYear,product]);
  const counts = useMemo(() => {
    const m = new Map<string,number>();
    for (const r of relations?.items || []) {
      if (!r.province) continue;
      m.set(r.province, (m.get(r.province) || 0) + 1);
    }
    return m;
  }, [relations]);
  const displayGeo = useMemo(() => geo ? ({...geo,features:geo.features.map((f:any) => ({
    ...f, properties:{...f.properties,event_count:counts.get(String(f.properties?.province_name || "")) || 0}
  }))}) : null, [geo,counts]);
  const maxCount = Math.max(1,...counts.values());
  return <section className="climate-panel" style={{marginBottom:16,padding:12}}>
    <div className="climate-panel-head"><div><h3>Türkiye — Kaynaklı iklim ilişkileri</h3>
      <p>İl seçimi, iklim ilişkileri ve belgeleri birlikte filtreler. Renklendirme yalnızca konumu belirtilmiş ilişkilerin sayısıdır; iklim etkisi veya nedensellik ölçüsü değildir.</p>
    </div></div>
    {error && <p role="alert">{error}</p>}
    {!relations && !error && <p>İklim haritası yükleniyor…</p>}
    {relations && <p className="climate-muted">Toplam {relations.total} ilişki; il konumu bulunan {Array.from(counts.values()).reduce((a,b)=>a+b,0)} ilişki haritada sayıldı. Konumsuz kayıtlar haritada gösterilmez.</p>}
    <div style={{height:420,minHeight:420,position:"relative"}}>
      <MapView geojson={displayGeo} selectedProvince={province} maxCount={maxCount}
        onProvinceClick={setProvince} metricLabel="iklim ilişkisi" />
    </div>
  </section>;
}
