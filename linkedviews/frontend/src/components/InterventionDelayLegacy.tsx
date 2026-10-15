import { useMemo } from "react";
import type { InterventionProvinceDetail, TimelinePoint } from "../types";
import { useLinkedStore } from "../store/useLinkedStore";

type Props = {
  detail: InterventionProvinceDetail | null;
  timeline: TimelinePoint[];
  startYear: number;
  endYear: number;
  loading: boolean;
};

type DelayRow = { province: string; firstReport: number; firstIntervention: number; delay: number; status: "valid" | "excluded" };
// Bildiri veri kaynağı: data_processed/uzalcbs_2026/figures/FIG_INTERVENTION_RESPONSE_DELAY_DATA.csv
// Sunum için sabitlenmiş veri; API'deki farklı müdahale tanımlarıyla birleştirilmez.
const DELAYS: DelayRow[] = [
  { province: "Yalova", firstReport: 2025, firstIntervention: 2018, delay: -7, status: "excluded" },
  { province: "Amasya", firstReport: 2025, firstIntervention: 2024, delay: -1, status: "excluded" },
  { province: "Giresun", firstReport: 2019, firstIntervention: 2018, delay: -1, status: "excluded" },
  { province: "Ordu", firstReport: 2019, firstIntervention: 2018, delay: -1, status: "excluded" },
  { province: "Bartın", firstReport: 2023, firstIntervention: 2023, delay: 0, status: "valid" },
  { province: "Diyarbakır", firstReport: 2025, firstIntervention: 2025, delay: 0, status: "valid" },
  { province: "Sinop", firstReport: 2023, firstIntervention: 2023, delay: 0, status: "valid" },
  { province: "Bilecik", firstReport: 2024, firstIntervention: 2025, delay: 1, status: "valid" },
  { province: "Samsun", firstReport: 2017, firstIntervention: 2018, delay: 1, status: "valid" },
  { province: "Artvin", firstReport: 2016, firstIntervention: 2018, delay: 2, status: "valid" },
  { province: "Düzce", firstReport: 2021, firstIntervention: 2023, delay: 2, status: "valid" },
  { province: "Trabzon", firstReport: 2017, firstIntervention: 2019, delay: 2, status: "valid" },
  { province: "Çanakkale", firstReport: 2023, firstIntervention: 2025, delay: 2, status: "valid" },
  { province: "Rize", firstReport: 2016, firstIntervention: 2019, delay: 3, status: "valid" },
  { province: "İstanbul", firstReport: 2017, firstIntervention: 2020, delay: 3, status: "valid" },
  { province: "Sakarya", firstReport: 2018, firstIntervention: 2024, delay: 6, status: "valid" },
];
const validRows = DELAYS.filter(x => x.status === "valid");
const excludedRows = DELAYS.filter(x => x.status === "excluded");
const mean = validRows.reduce((s, x) => s + x.delay, 0) / validRows.length;
const sorted = [...validRows].sort((a, b) => a.delay - b.delay);
const median = (sorted[5].delay + sorted[6].delay) / 2;
const fmt = (n: number, digits = 2) => new Intl.NumberFormat("tr-TR", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(n);

export default function InterventionView({ startYear, endYear }: Props) {
  const province = useLinkedStore(s => s.selectedProvince);
  const product = useLinkedStore(s => s.selectedProduct);
  const focusProvince = useLinkedStore(s => s.setSelectedProvince);
  const current = useMemo(() => DELAYS.find(x => x.province.toLocaleLowerCase("tr-TR") === (province || "").toLocaleLowerCase("tr-TR")), [province]);
  const displayRows = useMemo(() => [...validRows].sort((a, b) => b.delay - a.delay || a.province.localeCompare(b.province, "tr")), []);
  const withinTwoYears = validRows.filter(x => x.delay <= 2).length;
  const metricStyle = { padding: "16px 18px", borderRadius: 12, background: "#1b2940", border: "1px solid #334155" };
  return <section className="intervention-linked-view">
    <div className="climate-header"><div>
      <span className="eyebrow">LINKED INTERVENTION VIEW · BİLDİRİ SÜRÜMÜ</span>
      <h2>İlk Kayıt – İlk Kamu Müdahalesi</h2>
      <p className="climate-header-description">İl düzeyinde ilk belgelenmiş varlık kaydı ile ilk müdahale kaydı arasındaki süre.</p>
      <p className="climate-header-context">Sabit bildiri veri kümesi · 16 il · 12 karşılaştırılabilir il</p>
    </div></div>
    <div className="climate-context-bar">
      <span>⌖ Seçili il: {province || "Türkiye"}</span>
      <span>◷ Uygulama filtresi: {startYear === endYear ? startYear : `${startYear}–${endYear}`}</span>
      <span>◈ Ürün: {product || "Tüm ürünler"}</span>
    </div>
    <p className="muted" style={{ margin: "12px 0", fontSize: 12 }}>
      <strong>Yöntem notu:</strong> Aşağıdaki bildiri sonuçları sabittir; üstteki ürün ve yıl filtreleri bu 12 illik karşılaştırmayı yeniden hesaplamaz. İl seçimi yalnızca ilgili satırı vurgular.
    </p>
    <div className="climate-metrics" style={{ marginTop: 16 }}>
      <div><span>Ortalama gecikme</span><strong>{fmt(mean)} yıl</strong><small>12 karşılaştırılabilir il</small></div>
      <div><span>Medyan gecikme</span><strong>{fmt(median, 0)} yıl</strong><small>İl bazında</small></div>
      <div><span>İki yıl içinde</span><strong>{withinTwoYears}/{validRows.length} · %{fmt(withinTwoYears / validRows.length * 100, 0)}</strong><small>Aynı yıl müdahale dahil</small></div>
    </div>
    <section className="context-card" style={{ marginTop: 16 }}>
      <span className="eyebrow">BİLDİRİDEKİ İL BAZINDA GECİKME</span>
      <h3>İlk kayıt ile ilk müdahale arasındaki süre</h3>
      <p className="muted" style={{ fontSize: 12 }}>Çubuğa tıklayarak ilgili ili Linked Views seçiminde vurgulayabilirsiniz. Değerler yıl cinsindendir.</p>
      <div style={{ display: "grid", gap: 10, marginTop: 16 }}>
        {displayRows.map(row => {
          const selected = current?.province === row.province;
          return <button type="button" key={row.province} onClick={() => focusProvince(selected ? null : row.province)}
            title={`${row.province}: ${row.firstReport} → ${row.firstIntervention}, ${row.delay} yıl`}
            style={{ textAlign: "left", border: selected ? "1px solid #38bdf8" : "1px solid transparent", borderRadius: 8, background: selected ? "#1e3a55" : "transparent", color: "inherit", padding: "6px 8px", cursor: "pointer" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 10, fontSize: 13, marginBottom: 5 }}>
              <span><strong>{row.province}</strong> <span style={{ opacity: .72, fontSize: 11 }}>({row.firstReport} → {row.firstIntervention})</span></span>
              <strong>{row.delay} yıl</strong>
            </div>
            <div style={{ height: 12, borderRadius: 6, background: "#26344c", overflow: "hidden" }}>
              <div style={{ width: `${row.delay / 6 * 100}%`, height: "100%", borderRadius: 6, background: selected ? "#38bdf8" : "linear-gradient(90deg,#818cf8,#38bdf8)" }}/>
            </div>
          </button>;
        })}
      </div>
      <p className="muted" style={{ fontSize: 12, marginTop: 16 }}>En uzun gecikme: <strong>Sakarya, 6 yıl</strong>. Aynı yıl müdahale kaydı: <strong>3 il</strong> (Bartın, Diyarbakır, Sinop).</p>
    </section>
    {current && <section className="context-card" style={{ marginTop: 16, borderColor: "#38bdf8" }}>
      <span className="eyebrow">SEÇİLİ İL · BİLDİRİ VERİSİ</span>
      <h3>{current.province}</h3>
      <p>İlk belgelenmiş kayıt: <strong>{current.firstReport}</strong> · İlk müdahale kaydı: <strong>{current.firstIntervention}</strong></p>
      <p>{current.status === "valid" ? `Gecikme: ${current.delay} yıl` : "Kronoloji nedeniyle 12 illik gecikme hesabının dışında tutulmuştur."}</p>
    </section>}
    <section className="context-card" style={{ marginTop: 16 }}>
      <span className="eyebrow">KAPSAM VE DIŞLAMALAR</span>
      <h3>Karşılaştırma ölçütü</h3>
      <p>İlk rapor ve ilk müdahale yılı belirlenen 16 ilin 12'sinde müdahale yılı ilk kayıt yılına eşit veya daha sonradır. Diğer dört il kronolojik karşılaştırmaya dahil edilmemiştir.</p>
      <div style={{ ...metricStyle, marginTop: 12, fontSize: 13 }}>
        {excludedRows.map(x => <div key={x.province} style={{ margin: "5px 0" }}><strong>{x.province}</strong>: ilk kayıt {x.firstReport}, ilk müdahale {x.firstIntervention} (müdahale daha erken)</div>)}
      </div>
      <p className="muted" style={{ fontSize: 12, marginTop: 12 }}>Kaynak: UZAL-CBS 2026 bildiri figür verisi, <code>FIG_INTERVENTION_RESPONSE_DELAY_DATA.csv</code>. Buradaki “ilk” ifadeleri kaynaklarda belgelenmiş en erken tarihleri belirtir; gerçek biyolojik ilk görülme veya sahadaki ilk uygulama olduğu iddia edilmez.</p>
    </section>
  </section>;
}
