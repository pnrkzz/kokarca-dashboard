import ClimateMap from "./ClimateMap";
import ClimateRelationsPanel from "./ClimateRelationsPanel";
import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import {
  getClimateEvidence,
  getClimateFactors,
  getClimateRagCard,
  getClimateSummary,
  getEra5Summary,
} from "../services/api";
import { useLinkedStore } from "../store/useLinkedStore";
import type {
  ClimateEvidenceItem,
  ClimateFactorCount,
  ClimateRagCard,
  ClimateSummary,
  Era5Summary,
} from "../types/climate";

type Props = { startYear: number; endYear: number };

const FACTOR_LABELS: Record<string, string> = {
  climate_adaptation_general: "İklim uyumu",
  cold_weather: "Soğuk hava",
  temperature_general: "Sıcaklık",
  frost: "Don",
  mild_winter: "Ilıman kış",
  temperature_event: "Sıcaklık olayı",
  climate_change_general: "İklim değişikliği",
  precipitation_event: "Yağış",
  climate_suitability_general: "İklim uygunluğu",
  drought_event: "Kuraklık",
};

const CONSISTENCY_LABELS: Record<string, string> = {
  "Consistent": "Tutarlı",
  "Partially consistent": "Kısmen tutarlı",
  "Not consistent": "Tutarlı değil",
  "Observed anomaly present": "Anomali gözlendi",
  "Not directly testable": "Doğrudan test edilemez",
};

function factorLabel(v: string | null) {
  return v ? (FACTOR_LABELS[v] || v.replaceAll("_", " ")) : "Belirsiz";
}
function consistencyLabel(v: string | null) {
  return v ? (CONSISTENCY_LABELS[v] || v) : "Belirsiz";
}
function fmt(v: number | null | undefined, unit = "") {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v > 0 ? "+" : ""}${v.toFixed(2)}${unit}`;
}

export default function ClimateView({ startYear, endYear }: Props) {
  const selectedProvince = useLinkedStore((s) => s.selectedProvince);
  const selectedProduct = useLinkedStore((s) => s.selectedProduct);

  const [summary, setSummary] = useState<ClimateSummary | null>(null);
  const [factors, setFactors] = useState<ClimateFactorCount[]>([]);
  const [evidence, setEvidence] = useState<ClimateEvidenceItem[]>([]);
  const [era5, setEra5] = useState<Era5Summary | null>(null);
  const [rag, setRag] = useState<ClimateRagCard | null>(null);
  const [factor, setFactor] = useState("");
  const [consistency, setConsistency] = useState("");
  const [loading, setLoading] = useState(false);

  const ragYear = startYear === endYear ? startYear : endYear;

  useEffect(() => {
    setLoading(true);
    Promise.all([
      getClimateSummary({ province: selectedProvince || undefined, product: selectedProduct || undefined, startYear, endYear }),
      getClimateFactors({ province: selectedProvince || undefined, product: selectedProduct || undefined, startYear, endYear }),
      getClimateEvidence({ province: selectedProvince || undefined, product: selectedProduct || undefined, startYear, endYear, factor: factor || undefined, consistency: consistency || undefined, limit: 100 }),
      getEra5Summary({ province: selectedProvince || undefined, product: selectedProduct || undefined, startYear, endYear }),
    ])
      .then(([s, f, e, a]) => {
        setSummary(s); setFactors(f); setEvidence(e); setEra5(a);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedProvince, selectedProduct, startYear, endYear, factor, consistency]);

  useEffect(() => {
    if (!selectedProvince) { setRag(null); return; }
    getClimateRagCard(selectedProvince, ragYear).then(setRag).catch(() => setRag(null));
  }, [selectedProvince, ragYear]);

  const factorChartOption = useMemo(() => {
    const data = [...factors].sort((a, b) => b.count - a.count).slice(0, 10);

    return {
      backgroundColor: "transparent",
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      grid: { left: 126, right: 34, top: 18, bottom: 26 },
      xAxis: {
        type: "value",
        axisLabel: { color: "#9fa1b5", fontSize: 10 },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: "rgba(255,255,255,.05)" } },
      },
      yAxis: {
        type: "category",
        inverse: true,
        data: data.map((x) => factorLabel(x.factor)),
        axisLabel: { color: "#cfd1dc", fontSize: 11 },
        axisTick: { show: false },
        axisLine: { show: false },
      },
      series: [
        {
          type: "bar",
          data: data.map((x) => x.count),
          barWidth: 13,
          itemStyle: { borderRadius: [0, 5, 5, 0] },
          label: { show: true, position: "right", color: "#d7d8e3", fontSize: 10 },
        },
      ],
    };
  }, [factors]);

  const consistencyChartOption = useMemo(() => {
    const data = [
      { name: "Tutarlı", value: summary?.consistent_count ?? 0 },
      { name: "Kısmen tutarlı", value: summary?.partially_consistent_count ?? 0 },
      { name: "Tutarlı değil", value: summary?.not_consistent_count ?? 0 },
    ].filter((x) => x.value > 0);

    return {
      backgroundColor: "transparent",
      tooltip: { trigger: "item", formatter: "{b}: {c} ({d}%)" },
      legend: { bottom: 0, textStyle: { color: "#aeb0bf", fontSize: 10 } },
      series: [
        {
          type: "pie",
          radius: ["52%", "72%"],
          center: ["50%", "43%"],
          avoidLabelOverlap: true,
          label: { show: true, formatter: "{d}%", color: "#d7d8e3", fontSize: 10 },
          labelLine: { length: 8, length2: 5 },
          data,
        },
      ],
    };
  }, [summary]);

  const contextText = [
    selectedProvince || "Türkiye",
    startYear === endYear ? String(startYear) : `${startYear}–${endYear}`,
    selectedProduct || "Tüm ürünler",
  ].join(" · ");

  return (
    <section className="climate-view">
      <div className="climate-header">
        <div>
          <span className="eyebrow">LINKED CLIMATE ANALYSIS</span>
          <h2>İklim ve İklim Değişikliği Analizi</h2>
          <p className="climate-header-description">
            Çok kaynaklı metinlerden çıkarılan iklim anlatılarının, gözlemsel ERA5 verileri ve
            LLM tabanlı iklim değerlendirmeleriyle bağlantılı analizi
          </p>
          <p className="climate-header-context">{contextText}</p>
        </div>

        <div className="climate-filters">
          <label>
            <span>Faktör</span>
            <select value={factor} onChange={(e) => setFactor(e.target.value)}>
              <option value="">Tümü</option>
              {factors.map((x) => (
                <option key={x.factor} value={x.factor}>
                  {factorLabel(x.factor)}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Tutarlılık</span>
            <select value={consistency} onChange={(e) => setConsistency(e.target.value)}>
              <option value="">Tümü</option>
              <option value="Consistent">Tutarlı</option>
              <option value="Partially consistent">Kısmen tutarlı</option>
              <option value="Not consistent">Tutarlı değil</option>
              <option value="Observed anomaly present">Anomali gözlendi</option>
              <option value="Not directly testable">Doğrudan test edilemez</option>
            </select>
          </label>
        </div>
      </div>

      <div className="climate-context-bar">
        <span>⌖ {selectedProvince || "Türkiye geneli"}</span>
        <span>◷ {startYear === endYear ? startYear : `${startYear}–${endYear}`}</span>
        <span>◈ {selectedProduct || "Tüm ürünler"}</span>
      </div>

      <ClimateMap startYear={startYear} endYear={endYear} />

      <div className="climate-metrics">
        <div>
          <span>Metinsel iklim anlatısı</span>
          <strong>{summary?.climate_narrative_count ?? "—"}</strong>
          <small>LLM ile belirlenen iklim ilişkili anlatılar</small>
        </div>
        <div>
          <span>ERA5 ile sınanabilir</span>
          <strong>{summary?.era5_testable_count ?? "—"}</strong>
          <small>Gözlemsel veriyle karşılaştırılabilen anlatılar</small>
        </div>
        <div className="metric-good">
          <span>Tutarlı</span>
          <strong>{summary?.consistent_count ?? "—"}</strong>
          <small>ERA5 gözlemiyle desteklenen</small>
        </div>
        <div>
          <span>Kısmen tutarlı</span>
          <strong>{summary?.partially_consistent_count ?? "—"}</strong>
          <small>Kısmi gözlemsel destek</small>
        </div>
        <div className="metric-bad">
          <span>Tutarlı değil</span>
          <strong>{summary?.not_consistent_count ?? "—"}</strong>
          <small>ERA5 ile desteklenmeyen</small>
        </div>
      </div>

      <div className="climate-analysis-grid">
        <section className="climate-panel climate-chart-panel">
          <div className="climate-panel-head">
            <div>
              <h3>Metinlerden Çıkarılan İklim Faktörleri</h3>
              <p>LLM tarafından iklim anlatılarında belirlenen başlıca faktörler</p>
            </div>
            <span>{factors.reduce((a, b) => a + b.count, 0)}</span>
          </div>

          {factors.length === 0 ? (
            <div className="climate-empty">Bu bağlamda iklim faktörü bulunamadı.</div>
          ) : (
            <ReactECharts
              option={factorChartOption}
              style={{ height: "300px", width: "100%" }}
            />
          )}
        </section>

        <section className="climate-panel climate-chart-panel">
          <div className="climate-panel-head">
            <div>
              <h3>ERA5 Tutarlılık Dağılımı</h3>
              <p>Metinsel iklim anlatılarının gözlemsel iklim verileriyle karşılaştırılması</p>
            </div>
          </div>

          {(summary?.consistent_count ?? 0) +
            (summary?.partially_consistent_count ?? 0) +
            (summary?.not_consistent_count ?? 0) === 0 ? (
            <div className="climate-empty">Bu bağlamda tutarlılık sonucu bulunamadı.</div>
          ) : (
            <ReactECharts
              option={consistencyChartOption}
              style={{ height: "300px", width: "100%" }}
            />
          )}
        </section>
      </div>

      <div className="climate-secondary-grid">
        <section className="climate-panel">
          <div className="climate-panel-head">
            <div>
              <h3>Gözlemsel İklim Göstergeleri — ERA5</h3>
              <p>Seçili bağlam için ortalama anomali değerleri</p>
            </div>
            <span>{era5?.record_count ?? 0} kayıt</span>
          </div>

          <div className="era5-grid">
            <div><span>Yıllık sıcaklık</span><strong>{fmt(era5?.temperature_anomaly_mean, " °C")}</strong></div>
            <div><span>Kış sıcaklığı</span><strong>{fmt(era5?.winter_temperature_anomaly_mean, " °C")}</strong></div>
            <div><span>Bağıl nem</span><strong>{fmt(era5?.humidity_anomaly_mean)}</strong></div>
            <div><span>Yağış</span><strong>{fmt(era5?.precipitation_anomaly_mean, " mm")}</strong></div>
          </div>
        </section>

        <section className="climate-panel">
          <div className="climate-panel-head">
            <div>
              <h3>ClimateGPT / RAG Bağlamı</h3>
              <p>İl-yıl düzeyinde olay, belge ve iklim anlatısı bağlamı</p>
            </div>
            <span>{selectedProvince ? ragYear : "—"}</span>
          </div>

          {!selectedProvince ? (
            <p className="climate-muted">ClimateGPT / RAG bağlamı için bir il seçin.</p>
          ) : !rag ? (
            <p className="climate-muted">Bu il-yıl için RAG kartı bulunamadı.</p>
          ) : (
            <div className="rag-summary">
              <div><span>Olay</span><strong>{rag.event_count ?? "—"}</strong></div>
              <div><span>Belge</span><strong>{rag.document_count ?? "—"}</strong></div>
              <div><span>İklim anlatısı</span><strong>{rag.climate_narrative_count ?? "—"}</strong></div>
              <div>
                <span>Ort. sıcaklık anomalisi</span>
                <strong>
                  {typeof rag.temperature_c_mean_anomaly === "number"
                    ? fmt(rag.temperature_c_mean_anomaly, " °C")
                    : "—"}
                </strong>
              </div>
            </div>
          )}

          <div className="climategpt-actions">
            <a
              href="https://climategpt.ai/"
              target="_blank"
              rel="noreferrer"
              className="climategpt-external-link"
            >
              <span>🌐</span>
              ClimateGPT'yi Aç
              <b>↗</b>
            </a>

            <details className="climategpt-query-details">
              <summary>
                <span>🧠</span>
                Bu çalışmada kullanılan örnek sorgular
              </summary>
              <div className="climategpt-query-list">
                <p>ClimateGPT değerlendirmesinde kullanılan örnek soru çerçeveleri:</p>
                <ul>
                  <li>Halyomorpha halys ile sıcaklık arasında bilimsel olarak bilinen ilişki nedir?</li>
                  <li>Ilıman kış koşulları Halyomorpha halys'in hayatta kalmasını veya yayılımını etkiler mi?</li>
                  <li>İklim değişikliğinin Halyomorpha halys'in potansiyel dağılımı üzerindeki etkisine ilişkin bilimsel kanıtlar nelerdir?</li>
                  <li>Sıcaklık artışları türün gelişim süresi, nesil sayısı veya popülasyon dinamiklerini nasıl etkileyebilir?</li>
                </ul>
                <small>
                  ClimateGPT çıktıları tek başına gözlemsel doğrulama olarak kullanılmamış;
                  ERA5 karşılaştırmasından ayrı bir bilimsel bağlam katmanı olarak değerlendirilmiştir.
                </small>
              </div>
            </details>
          </div>
        </section>
      </div>

      <ClimateRelationsPanel startYear={startYear} endYear={endYear} />

      <main className="climate-evidence-panel">
        <div className="climate-panel-head evidence-head">
          <div>
            <h3>İklim Kanıtları</h3>
            <p>Metinsel iklim bilgileri, ERA5 gözlemleri ve tutarlılık değerlendirmesi</p>
          </div>
          <span>{evidence.length} kanıt</span>
        </div>

        {loading ? (
          <div className="climate-loading">İklim kanıtları yükleniyor…</div>
        ) : evidence.length === 0 ? (
          <div className="climate-empty">Bu bağlamda iklim kanıtı bulunamadı.</div>
        ) : (
          <div className="climate-evidence-list">
            {evidence.map((item, index) => (
              <article
                key={item.event_id || `${item.global_id}-${index}`}
                className="climate-evidence-card"
              >
                <div className="climate-card-top">
                  <div className="climate-card-badges">
                    <span className="factor-badge">{factorLabel(item.factor)}</span>
                    <span className="source-mini">{item.source_type || "Kaynak"}</span>
                  </div>
                  <div className="climate-card-year">
                    <span>İklim yılı</span>
                    <strong>{item.climate_year ?? "—"}</strong>
                  </div>
                </div>

                {item.evidence_sentence && <blockquote>{item.evidence_sentence}</blockquote>}

                {item.climate_comment && (
                  <div className="climate-interpretation">
                    <span>Metinden çıkarılan iklim ilişkisi</span>
                    <p>{item.climate_comment}</p>
                  </div>
                )}

                <div className="anomaly-strip">
                  <div><span>Yıllık sıcaklık</span><strong>{fmt(item.annual_temperature_anomaly, " °C")}</strong></div>
                  <div><span>Kış</span><strong>{fmt(item.winter_temperature_anomaly, " °C")}</strong></div>
                  <div><span>Nem</span><strong>{fmt(item.relative_humidity_anomaly)}</strong></div>
                  <div><span>Yağış</span><strong>{fmt(item.precipitation_anomaly, " mm")}</strong></div>
                </div>

                <div className="consistency-box">
                  <div>
                    <span>ERA5 tutarlılığı</span>
                    <strong>{consistencyLabel(item.era5_consistency)}</strong>
                  </div>
                  {item.consistency_reason && <p>{item.consistency_reason}</p>}
                </div>
              </article>
            ))}
          </div>
        )}
      </main>
    </section>
  );
}
