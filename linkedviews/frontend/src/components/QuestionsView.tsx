import "./QuestionsView.css";
import { useEffect, useMemo, useState } from "react";

type Evidence = {
  final_relation_id: string;
  global_id: string;
  source_type?: string;
  title?: string;
  url?: string;
  published_date?: string;
  publication_date_precision?: string;
  date_is_synthetic?: boolean | string;
  province?: string;
  event_year?: string | number;
  evidence_quote?: string;
  driver?: string;
  effect?: string;
  effect_direction?: string;
  climate_change_status?: string;
  evidence_use_scope?: string;
  geographic_scope?: string;
  final_integrated_evidence_status?: string;
};

type ResponseData = {
  items: Evidence[];
  total_relations?: number;
  total_documents?: number;
};

const questions = [
  {
    id: "spread",
    title: "İklim değişikliği H. halys yayılımını artırdı mı?",
    pattern: /spread|distribution|range_expansion|population_expansion|establishment/i,
  },
  {
    id: "population",
    title: "İklim koşulları H. halys popülasyonunu nasıl etkiliyor?",
    pattern: /population|abundance|survival|generation|voltinism/i,
  },
  {
    id: "winter",
    title: "Sıcaklık ve kış koşulları H. halys kışlama davranışını etkiliyor mu?",
    pattern: /overwinter|diapause|winter|emergence/i,
  },
  {
    id: "damage",
    title: "İklim koşulları H. halys kaynaklı ürün zararlarıyla ilişkili mi?",
    pattern: /crop|damage|yield|fruit|hazelnut/i,
  },
];

const sourceNames: Record<string, string> = {
  academic: "Bilimsel yayın",
  official: "Resmî kaynak",
  national_news: "Ulusal haber",
  local_news: "Yerel haber",
  youtube: "YouTube",
};

function directionName(value?: string) {
  const names: Record<string, string> = {
    INCREASE: "Artış",
    EXPANSION: "Genişleme",
    DECREASE: "Azalma",
    CONTRACTION: "Daralma",
    SHIFT: "Yer değiştirme",
    EARLIER: "Erkenleşme",
    LATER: "Gecikme",
    NONE: "Belirtilmemiş",
  };
  return names[value || ""] || value || "Belirsiz";
}

function scopeName(value?: string) {
  const names: Record<string, string> = {
    TURKEY_EMPIRICAL: "Türkiye - ampirik",
    TURKEY_SUPPORTING: "Türkiye - destekleyici",
    GENERAL_SCIENTIFIC: "Genel bilimsel bilgi",
    FOREIGN_SUPPORTING: "Yurt dışı bulgu",
    FUTURE_PROJECTION: "Gelecek projeksiyonu",
  };
  return names[value || ""] || value || "Belirtilmemiş";
}

function eraName(value?: string) {
  const names: Record<string, string> = {
    CLIMATEGPT_CONTEXT_ONLY: "Yalnızca ClimateGPT bağlamı",
    NO_CLEAR_ERA5_ANOMALY: "ERA5: belirgin anomali yok",
    ERA5_NOT_DIRECTLY_TESTABLE: "ERA5: doğrudan test edilemez",
    ERA5_SPATIAL_SCALE_LIMITATION: "ERA5: mekânsal ölçek sınırlaması",
    ERA5_DIRECTIONAL_CONTRAST: "ERA5: yönsel uyumsuzluk",
    ERA5_DIRECTIONAL_SUPPORT: "ERA5: yönsel destek",
    SOURCE_TEXT_ONLY: "Yalnızca kaynak metni",
  };
  return names[value || ""] || value || "Değerlendirme yok";
}

function relationSummary(item: Evidence): string {
  const effect = (item.effect || "").toLowerCase();
  const driver = (item.driver || "").toLowerCase();
  const direction = item.effect_direction || "";
  const subject = /overwinter|diapause|winter|emergence/.test(effect)
    ? "kışlama veya mevsimsel davranış"
    : /crop|damage|yield|fruit|hazelnut/.test(effect)
      ? "tarımsal zarar"
      : /population|abundance|survival|generation|voltinism/.test(effect)
        ? "popülasyon ve yaşam döngüsü"
        : /spread|distribution|range_expansion|establishment/.test(effect)
          ? "coğrafi dağılım veya yayılım"
          : "türün iklimle ilişkisi";
  const climate = /temperature|warming|heat|cold|winter/.test(driver)
    ? "sıcaklık koşulları"
    : /rain|precipitation/.test(driver)
      ? "yağış koşulları"
      : /humidity|moisture/.test(driver)
        ? "nem koşulları"
        : /climate_change/.test(driver)
          ? "iklim değişikliği"
          : "iklimsel veya çevresel koşullar";
  const movement = direction === "INCREASE" || direction === "EXPANSION"
    ? "artış veya genişleme yönünde"
    : direction === "DECREASE" || direction === "CONTRACTION"
      ? "azalış veya daralma yönünde"
      : direction === "SHIFT" ? "yer değiştirme yönünde" : "";
  const kind = item.climate_change_status === "FUTURE_PROJECTION" ||
    item.evidence_use_scope === "FUTURE_PROJECTION"
    ? "bir gelecek projeksiyonuna"
    : item.climate_change_status === "EXPLICIT_CAUSAL"
      ? "kaynakta ileri sürülen nedensel bir ilişkiye"
      : "bir ilişki ifadesine";
  return `Bu belgede ${climate} ile ${subject} arasında ${movement ? movement + " " : ""}${kind} yer veriliyor.`;
}

function publicationYear(item: Evidence): string {
  const raw = (item.published_date || "").trim();
  const year = raw.match(/^(\d{4})/);
  const synthetic = item.date_is_synthetic === true ||
    String(item.date_is_synthetic).toLowerCase() === "true";
  return year && !synthetic ? year[1] : "Yılı belirsiz";
}

export default function QuestionsView() {
  const [items, setItems] = useState<Evidence[]>([]);
  const [questionId, setQuestionId] = useState("spread");
  const [source, setSource] = useState("all");
  const [scope, setScope] = useState("all");
  const [direction, setDirection] = useState("all");
  const [spreadView, setSpreadView] = useState<"geographic" | "population">("geographic");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetch("/api/questions/evidence?limit=500", {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`API hatası: ${response.status}`);
        }
        return response.json() as Promise<ResponseData>;
      })
      .then((data) => {
        setItems(data.items || []);
        setError("");
      })
      .catch((err: Error) => {
        if (err.name !== "AbortError") setError(err.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  const question = questions.find((q) => q.id === questionId)!;

  const relevant = useMemo(
    () => items.filter((item) => question.pattern.test(item.effect || "")),
    [items, question]
  );

  const filtered = useMemo(
    () =>
      relevant.filter((item) =>
        (source === "all" || item.source_type === source) &&
        (scope === "all" || item.evidence_use_scope === scope) &&
        (direction === "all" || item.effect_direction === direction)
      ),
    [relevant, source, scope, direction]
  );

  const scopeOrder: Record<string, number> = {
    TURKEY_EMPIRICAL: 0,
    TURKEY_SUPPORTING: 1,
    GENERAL_SCIENTIFIC: 2,
    FOREIGN_SUPPORTING: 3,
    FUTURE_PROJECTION: 4,
  };

  const sorted = [...filtered].sort(
    (a, b) =>
      (scopeOrder[a.evidence_use_scope || ""] ?? 99) -
      (scopeOrder[b.evidence_use_scope || ""] ?? 99)
  );

  const displayed = questionId === "spread"
    ? sorted.filter((item) =>
        spreadView === "population"
          ? spreadCategory(item).startsWith("Popülasyon")
          : spreadCategory(item).startsWith("Coğrafi")
      )
    : sorted;

  const scopeCounts = Object.keys(scopeOrder).map((key) => ({
    key,
    label: scopeName(key),
    count: displayed.filter((item) => item.evidence_use_scope === key).length,
  }));

  function evidenceReviewNote(item: Evidence): string | null {
    if (item.final_relation_id === "LOC_000304_E01_CR01") {
      return "İnceleme gerekli: Yüksek rakımda tespit, tek başına yükseltiye doğru dağılım kaymasını veya iklim değişikliğinin etkisini kanıtlamaz.";
    }

    if (item.final_relation_id === "NAT_000453_E01_CR01") {
      return "Nedensellik yönü sorunu: Kaynak cümlesi kahverengi kokarcayı iklim değişikliğinin sebebi olarak ifade ediyor. Yayılımın iklim değişikliğinden kaynaklandığına kanıt olarak kullanılamaz.";
    }

    if (item.final_relation_id === "YTB_000032_E02_CR04") {
      return "İklim değişikliği atfı yok: Kaynak, nemli ve ılıman iklim tercihini ve coğrafi yayılımı anlatıyor; iklim değişikliğinin yayılımı artırdığını göstermiyor.";
    }

    return null;
  }
  function spreadCategory(item: Evidence): string {
    const effect = (item.effect || "").toLowerCase();

    if (effect.includes("population_expansion")) {
      return "Popülasyon genişlemesi - coğrafi yayılım değil";
    }

    if (/spread|distribution|range_expansion|establishment/i.test(effect)) {
      return "Coğrafi yayılım veya dağılım";
    }

    return "Diğer";
  }
  function claimGroup(item: Evidence): string {
    const status = item.climate_change_status || "NONE";

    if (status === "FUTURE_PROJECTION" ||
        item.evidence_use_scope === "FUTURE_PROJECTION") {
      return "Gelecek projeksiyonu";
    }

    if (status === "EXPLICIT_CAUSAL" ||
        status === "EXPLICIT_ASSOCIATION") {
      return "Açık iklim değişikliği iddiası";
    }

    if (status === "NONE") {
      return "Genel iklim ilişkisi";
    }

    return "Diğer bağlamsal ifadeler";
  }

  const claimGroups = [
    "Açık iklim değişikliği iddiası",
    "Gelecek projeksiyonu",
    "Genel iklim ilişkisi",
    "Diğer bağlamsal ifadeler",
  ].map((label) => ({
    label,
    count: displayed.filter((item) => claimGroup(item) === label).length,
  }));
  const geographicCount = filtered.filter(
    (item) => spreadCategory(item).startsWith("Coğrafi")
  ).length;

  const populationCount = filtered.filter(
    (item) => spreadCategory(item).startsWith("Popülasyon")
  ).length;

  const documents = new Set(displayed.map((item) => item.global_id)).size;

  return (
    <section className="questions-view" style={{ padding: 24, maxWidth: 1250, margin: "0 auto" }}>
      <h2>Araştırma Soruları ve Kanıtlar</h2>
      <p>
        Çok kaynaklı metinlerden çıkarılmış iklim ilişkilerinin,
        kaynakları ve bilimsel sınırlılıklarıyla birlikte incelenmesi.
      </p>

      <div style={{ display: "grid", gap: 12, marginBottom: 20 }}>
        <label>
          <strong>Araştırma sorusu</strong>
          <select
            style={{ display: "block", width: "100%", padding: 10, marginTop: 6 }}
            value={questionId}
            onChange={(e) => {
              setQuestionId(e.target.value);
              setSource("all");
              setScope("all");
              setDirection("all");
            }}
          >
            {questions.map((q) => (
              <option key={q.id} value={q.id}>{q.title}</option>
            ))}
          </select>
        </label>
      </div>

      <div style={{
        display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 18,
      }}>
        <label>
          Kaynak türü{" "}
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="all">Tümü</option>
            {Object.entries(sourceNames).map(([key, label]) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </select>
        </label>

        <label>
          Kanıt kapsamı{" "}
          <select value={scope} onChange={(e) => setScope(e.target.value)}>
            <option value="all">Tümü</option>
            {[
              "TURKEY_EMPIRICAL",
              "TURKEY_SUPPORTING",
              "GENERAL_SCIENTIFIC",
              "FOREIGN_SUPPORTING",
              "FUTURE_PROJECTION",
            ].map((value) => (
              <option key={value} value={value}>{scopeName(value)}</option>
            ))}
          </select>
        </label>

        <label>
          Etki yönü{" "}
          <select value={direction} onChange={(e) => setDirection(e.target.value)}>
            <option value="all">Tümü</option>
            {[
              "INCREASE", "EXPANSION", "SHIFT",
              "DECREASE", "CONTRACTION", "EARLIER", "LATER",
            ].map((value) => (
              <option key={value} value={value}>{directionName(value)}</option>
            ))}
          </select>
        </label>
      </div>

      {loading && <p>Kanıtlar yükleniyor...</p>}
      {error && <p role="alert">Veri alınamadı: {error}</p>}

      {!loading && !error && (
        <>
          {questionId === "spread" && (
            <div style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 10,
              marginBottom: 16,
            }}>
              <button
                type="button"
                onClick={() => setSpreadView("geographic")}
                aria-pressed={spreadView === "geographic"}
                style={{
                  padding: "10px 14px",
                  borderRadius: 8,
                  background: spreadView === "geographic" ? "#285b85" : "#243044",
                  color: "#fff",
                  border: "1px solid #64748b",
                  cursor: "pointer",
                }}
              >
                Coğrafi yayılım ({geographicCount})
              </button>
              <button
                type="button"
                onClick={() => setSpreadView("population")}
                aria-pressed={spreadView === "population"}
                style={{
                  padding: "10px 14px",
                  borderRadius: 8,
                  background: spreadView === "population" ? "#285b85" : "#243044",
                  color: "#fff",
                  border: "1px solid #64748b",
                  cursor: "pointer",
                }}
              >
                Popülasyon genişlemesi ({populationCount})
              </button>
            </div>
          )}

          <p>
            <strong>{displayed.length}</strong> ilişki ·{" "}
            <strong>{documents}</strong> farklı belge
            {" "}({relevant.length} soru adayı)
          </p>

          <p style={{ fontSize: 13, opacity: 0.85 }}>
            Bu kayıtlar konuya göre seçilmiş aday kanıtlardır.
            Kaynakta yer alan nedensellik iddiası doğrulanmış nedensellik
            anlamına gelmez. Gelecek projeksiyonları geçmiş gözlemlerden
            ayrılmalıdır. ClimateGPT bağımsız doğrulama değildir.
          </p>

          {/* YAYIN YILINA DAYALI BELGE REHBERI */}
          <section style={{ margin: "24px 0", padding: 20, background: "#152235", border: "1px solid #46617b", borderRadius: 12 }}>
            <h3 style={{ marginTop: 0 }}>Literatür ve Kaynak Bulgularının Değerlendirilmesi</h3>
            <p style={{ fontSize: 13, opacity: 0.85 }}>
              Belgeler yayın yılına göre gruplandırılmıştır. Her açıklama, kaynak metninden
              çıkarılmış ilişkinin Türkçe konu özetidir; özgün cümlenin çevirisi veya
              bağımsız bilimsel doğrulama değildir. Yıl, olay yılı değil yayın yılıdır.
            </p>
            {(() => {
              const byDocument = new Map<string, Evidence[]>();
              displayed.forEach((item) => {
                if (!item.global_id) return;
                const previous = byDocument.get(item.global_id) || [];
                previous.push(item);
                byDocument.set(item.global_id, previous);
              });
              const byYear = new Map<string, Array<[string, Evidence[]]>>();
              Array.from(byDocument.entries()).forEach((entry) => {
                const year = publicationYear(entry[1][0]);
                const docs = byYear.get(year) || [];
                docs.push(entry);
                byYear.set(year, docs);
              });
              const years = Array.from(byYear.keys()).sort((a, b) => {
                if (a === "Yılı belirsiz") return 1;
                if (b === "Yılı belirsiz") return -1;
                return Number(a) - Number(b);
              });
              const count = byDocument.size;
              const directions = new Set(displayed.map((item) => item.effect_direction));
              const hasFuture = displayed.some((item) => item.climate_change_status === "FUTURE_PROJECTION" || item.evidence_use_scope === "FUTURE_PROJECTION");
              return (
                <>
                  <p><strong>{count}</strong> belge · <strong>{years.length}</strong> yayın yılı grubu.</p>
                  <p style={{ lineHeight: 1.7 }}>
                    <strong>Seçili soruya ilişkin genel görünüm: </strong>
                    {count === 0
                      ? "Bu filtrelerle ilişkilendirilmiş belge bulunamadı."
                      : `Seçili belgelerde ${question.title.toLocaleLowerCase("tr-TR")} sorusuyla ilişkili ifadeler yer alıyor. `}
                    {count > 0 && directions.has("INCREASE") && (directions.has("DECREASE") || directions.has("CONTRACTION"))
                      ? "Kaynak etiketlerinde hem artış hem azalış yönleri bulunmaktadır; tek yönlü bir sonuç çıkarılamaz. "
                      : null}
                    {count > 0 && hasFuture
                      ? "Gelecek projeksiyonları, gerçekleşmiş gözlemlerden ayrı değerlendirilmelidir. "
                      : null}
                    {count > 0 ? "Aşağıdaki belge bağlantıları üzerinden özgün ifadeler incelenebilir." : null}
                  </p>
                  {years.map((year) => (
                    <section key={year} style={{ marginTop: 20 }}>
                      <h4 style={{ borderBottom: "1px solid #64748b55", paddingBottom: 8, marginBottom: 10 }}>
                        {year === "Yılı belirsiz" ? year : `${year} yılında yayımlanan belgeler`}
                        {" · "}{byYear.get(year)?.length || 0} belge
                      </h4>
                      {(byYear.get(year) || []).map(([id, entries]) => {
                        const first = entries[0];
                        const url = entries.find((e) => e.url && /^https?:\/\//i.test(e.url))?.url;
                        const distinct = Array.from(new Map(entries.map((e) => [relationSummary(e), e])).values());
                        return (
                          <article key={id} style={{ padding: "12px 0", borderBottom: "1px solid #64748b33" }}>
                            <p style={{ margin: "0 0 6px", fontSize: 13, opacity: 0.85 }}>
                              {sourceNames[first.source_type || ""] || "Kaynak"} · {id}
                              {first.province ? ` · ${first.province}` : ""}
                            </p>
                            <strong>{first.title || "Başlık bilgisi bulunamadı"}</strong>
                            <p style={{ margin: "8px 0", lineHeight: 1.7 }}>
                              {distinct.slice(0, 3).map(relationSummary).join(" ")}
                              {distinct.length > 3 ? ` Ayrıca ${distinct.length - 3} farklı ilişki türü daha kaydedilmiştir.` : ""}
                            </p>
                            {url ? (
                              <a href={url} target="_blank" rel="noopener noreferrer">Belgeyi incele ↗</a>
                            ) : (
                              <span style={{ fontSize: 13, opacity: 0.75 }}>Dış kaynak bağlantısı bulunamadı · Belge kodu: {id}</span>
                            )}
                            <details style={{ marginTop: 8 }}>
                              <summary style={{ cursor: "pointer", fontSize: 13 }}>Özgün kaynak ifadeleri ({entries.length} ilişki)</summary>
                              {Array.from(new Set(entries.map((e) => (e.evidence_quote || "").trim()).filter(Boolean))).map((quote, index) => (
                                <blockquote key={`${id}-${index}`} style={{ margin: "10px 0", padding: "8px 12px", borderLeft: "3px solid #60a5fa", whiteSpace: "pre-wrap" }}>
                                  {quote}
                                </blockquote>
                              ))}
                            </details>
                          </article>
                        );
                      })}
                    </section>
                  ))}
                </>
              );
            })()}
          </section>
          <h3>Kanıtların değerlendirilmesi</h3>
          <div style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 10,
            margin: "16px 0 20px",
          }}>
            {scopeCounts.map((group) => (
              <div key={group.key} style={{
                padding: "10px 14px",
                background: "#1b2535",
                border: "1px solid #394355",
                borderRadius: 8,
              }}>
                <strong>{group.count}</strong>
                {" · "}
                {group.label}
              </div>
            ))}
          </div>

          <h3 style={{ marginTop: 20 }}>İklim değişikliği iddiasının türü</h3>

          <div style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 10,
            marginBottom: 20,
          }}>
            {claimGroups.map((group) => (
              <div key={group.label} style={{
                padding: "10px 14px",
                background: "#18283a",
                border: "1px solid #46617b",
                borderRadius: 8,
              }}>
                <strong>{group.count}</strong>
                {" · "}
                {group.label}
              </div>
            ))}
          </div>

          <details style={{ marginTop: 20 }}>
            <summary style={{ cursor: "pointer", fontWeight: 700 }}>Tüm kaynak kayıtlarını ayrıntılı incele</summary>
          <h3>Kaynaklarda geçen ifadeler</h3>
          <p style={{ fontSize: 13, opacity: 0.8 }}>
            Aşağıdaki ifadeler kaynak kayıtlarından alınmıştır.
            Otomatik araştırma yanıtı değildir.
          </p>

          {Object.entries(sourceNames).map(([sourceKey, sourceLabel]) => {
            const sourceItems = displayed.filter(
              (item) => item.source_type === sourceKey
            );

            if (sourceItems.length === 0) return null;

            return (
              <section key={sourceKey} style={{ marginBottom: 26 }}>
                <h3 style={{ marginBottom: 12 }}>
                  {sourceLabel} ({sourceItems.length})
                </h3>

                <div style={{ display: "grid", gap: 14 }}>
                  {sourceItems.map((item) => (
                    <article
                      key={item.final_relation_id}
                      style={{
                        border: "1px solid #64748b55",
                        borderRadius: 10,
                        padding: 18,
                      }}
                    >
                      <div style={{ fontSize: 13, opacity: 0.8 }}>
                        {item.global_id}
                        {item.published_date
                          ? ` · ${item.published_date.slice(0, 10)}`
                          : ""}
                        {item.province ? ` · ${item.province}` : ""}
                      </div>

                      <h4 style={{ margin: "10px 0" }}>
                        {item.title || "Başlık bilgisi bulunamadı"}
                      </h4>

                      <div style={{
                        fontSize: 12,
                        fontWeight: 700,
                        marginTop: 14,
                        marginBottom: 8,
                      }}>
                        KAYNAKTA GEÇEN İFADE
                      </div>

                      <blockquote style={{
                        margin: "0 0 16px",
                        padding: "14px 18px",
                        borderLeft: "4px solid #60a5fa",
                        background: "#172538",
                        borderRadius: 6,
                        lineHeight: 1.8,
                        fontSize: 16,
                        whiteSpace: "pre-wrap",
                      }}>
                        {item.evidence_quote || "Kaynak alıntısı bulunamadı"}
                      </blockquote>

                      {item.url && /^https?:\/\//i.test(item.url) && (
                        <p>
                          <a
                            href={item.url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            Özgün kaynağı aç ↗
                          </a>
                        </p>
                      )}

                      <details style={{ marginTop: 12, fontSize: 13 }}>
                        <summary style={{ cursor: "pointer" }}>
                          İlişki ve bilimsel inceleme ayrıntıları
                        </summary>

                        <div style={{ marginTop: 12, lineHeight: 1.9 }}>
                          <div><strong>İlişki kodu:</strong> {item.final_relation_id}</div>
                          <div><strong>İlişki:</strong> {item.driver || "—"} → {item.effect || "—"}</div>
                          <div><strong>Etki yönü:</strong> {directionName(item.effect_direction)}</div>
                          <div><strong>Kanıt kapsamı:</strong> {scopeName(item.evidence_use_scope)}</div>
                          <div><strong>İklim değişikliği etiketi:</strong> {item.climate_change_status || "—"}</div>
                          <div><strong>Değerlendirme:</strong> {eraName(item.final_integrated_evidence_status)}</div>
                          {evidenceReviewNote(item) && (
                            <p><strong>İnceleme notu:</strong> {evidenceReviewNote(item)}</p>
                          )}
                        </div>
                      </details>
                    </article>
                  ))}
                </div>
              </section>
            );
          })}
          </details>
          {displayed.length === 0 && (
            <p>Bu filtrelerle eşleşen kayıt bulunamadı.</p>
          )}
        </>
      )}
    </section>
  );
}











