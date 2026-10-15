import { useEffect, useState } from "react";
import { useLinkedStore } from "../store/useLinkedStore";

type Relation = {
  id: string; global_id: string | null; origin: string | null;
  province: string | null; year: number | null; source_type: string | null;
  title: string | null; url: string | null; published_date: string | null;
  relation_text: string | null; evidence_quote: string | null;
  driver: string | null; effect: string | null; climate_change_status: string | null;
  source_support: string | null; effect_path: string | null; review_note: string | null;
  era5_status: string | null; era5_match: string | null; climategpt_answer: string | null;
};
type Response = { total: number; items: Relation[]; notice: string };

export default function ClimateRelationsPanel({ startYear, endYear }: { startYear: number; endYear: number }) {
  const province = useLinkedStore(s => s.selectedProvince);
  const product = useLinkedStore(s => s.selectedProduct);
  const [data, setData] = useState<Response | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams({ start_year: String(startYear), end_year: String(endYear), limit: "200" });
    if (province) params.set("province", province);
    if (product) params.set("product", product);
    setLoading(true); setError(""); setData(null);
    fetch(`/api/climate/relations?${params}`, { signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error(`İklim ilişkileri API hatası: ${response.status}`);
        return response.json() as Promise<Response>;
      })
      .then(setData)
      .catch(err => { if (!controller.signal.aborted) setError(String(err)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [province, product, startYear, endYear]);

  return <section className="climate-panel" style={{ marginTop: 16 }}>
    <div className="climate-panel-head"><div>
      <h3>Kaynaklı iklim ilişkileri</h3>
      <p>Seçili ürün, il ve yıl için metinlerde belirtilen ilişkiler. ERA5 ve ClimateGPT ayrı değerlendirme katmanlarıdır.</p>
    </div><span>{data?.total ?? "—"} ilişki</span></div>
    {loading && <p>İlişkiler yükleniyor…</p>}
    {error && <p role="alert">{error}</p>}
    {data && <><p className="climate-muted">{data.notice}</p>
      {data.items.length === 0 && <p>Seçili filtreler için ilişki bulunamadı.</p>}
      <div className="climate-evidence-list">{data.items.map(item => <article key={item.id} className="climate-evidence-card">
        <div className="climate-card-top"><div className="climate-card-badges">
          <span className="source-mini">{item.source_type || "Kaynak türü belirtilmemiş"}</span>
          <span className="factor-badge">{item.driver || "İklim ilişkisi"}</span>
        </div><span>{item.province || "Konum belirtilmemiş"} · {item.year ?? "Yıl belirtilmemiş"}</span></div>
        {item.evidence_quote && <><small>Özgün kanıt ifadesi</small><blockquote>{item.evidence_quote}</blockquote></>}
        {item.relation_text && <div className="climate-interpretation"><small>Metinden çıkarılmış ilişki</small><p>{item.relation_text}</p></div>}
        {item.effect_path && <p><strong>Etki yolu:</strong> {item.effect_path}</p>}
        {item.source_support && <p><strong>Kaynak incelemesi:</strong> {item.source_support}</p>}
        {item.era5_status && <p><strong>ERA5:</strong> {item.era5_status}</p>}
        {item.climategpt_answer && <details><summary>ClimateGPT literatür bağlamı (bağımsız doğrulama değil)</summary><p>{item.climategpt_answer}</p></details>}
        <p><small>Belge: {item.global_id} · {item.title || "Başlık belirtilmemiş"}</small></p>
        {item.url && /^https?:\/\//i.test(item.url) && <a href={item.url} target="_blank" rel="noreferrer">Kaynağı aç ↗</a>}
      </article>)}</div>
      {data.total > data.items.length && <p>İlk {data.items.length} ilişki gösteriliyor; toplam {data.total}.</p>}
    </>}
  </section>;
}
