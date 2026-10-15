import { useEffect, useMemo, useState } from "react";
import { getDocuments } from "../services/api";
import { useLinkedStore } from "../store/useLinkedStore";
import type { DocumentItem } from "../types/documents";

type Props = {
  startYear: number;
  endYear: number;
};

function sourceLabel(value: string | null) {
  switch (value) {
    case "official": return "Resmî";
    case "local_news": return "Yerel Haber";
    case "national_news": return "Ulusal Haber";
    case "academic": return "Akademik";
    case "youtube": return "YouTube";
    default: return value || "Kaynak";
  }
}

function displayTitle(doc: DocumentItem) {
  if (doc.title?.trim()) return doc.title.trim();
  try {
    if (doc.url) {
      const host = new URL(doc.url).hostname.replace(/^www\./, "");
      return `Başlık mevcut değil · ${host}`;
    }
  } catch {}
  return "Başlık mevcut değil";
}

export default function DocumentsView({ startYear, endYear }: Props) {
  const selectedProvince = useLinkedStore((s) => s.selectedProvince);
  const selectedProduct = useLinkedStore((s) => s.selectedProduct);

  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [sourceType, setSourceType] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    getDocuments({
      province: selectedProvince || undefined,
      product: selectedProduct || undefined,
      startYear,
      endYear,
      sourceType: sourceType || undefined,
      limit: 200,
    })
      .then(setDocuments)
      .catch((error) => {
        console.error(error);
        setDocuments([]);
      })
      .finally(() => setLoading(false));
  }, [selectedProvince, selectedProduct, startYear, endYear, sourceType]);

  const sourceTypes = useMemo(() => {
    return Array.from(
      new Set(
        documents
          .map((doc) => doc.source_type)
          .filter((value): value is string => Boolean(value)),
      ),
    ).sort();
  }, [documents]);

  const contextText = [
    selectedProvince || "Türkiye",
    startYear === endYear ? String(startYear) : `${startYear}–${endYear}`,
    selectedProduct || "Tüm ürünler",
  ].join(" · ");

  return (
    <section className="documents-view">
      <div className="documents-header">
        <div>
          <span className="eyebrow">LINKED DOCUMENT VIEW</span>
          <h2>Belgeler</h2>
          <p>{contextText}</p>
        </div>

        <div className="documents-tools">
          <div className="document-count">
            <span>Belge</span>
            <strong>{documents.length}</strong>
          </div>

          <label>
            <span>Kaynak türü</span>
            <select
              value={sourceType}
              onChange={(event) => setSourceType(event.target.value)}
            >
              <option value="">Tümü</option>
              {sourceTypes.map((type) => (
                <option key={type} value={type}>
                  {sourceLabel(type)}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="documents-context-bar">
        <span>⌖ {selectedProvince || "Türkiye geneli"}</span>
        <span>◷ {startYear === endYear ? startYear : `${startYear}–${endYear}`}</span>
        <span>◈ {selectedProduct || "Tüm ürünler"}</span>
        <span>Linked state korunuyor</span>
      </div>

      {loading ? (
        <div className="documents-loading">Belgeler yükleniyor…</div>
      ) : documents.length === 0 ? (
        <div className="documents-empty">
          <strong>Bu bağlamda belge bulunamadı.</strong>
          <p>İl, yıl veya ürün filtresini değiştirerek yeniden deneyin.</p>
        </div>
      ) : (
        <div className="documents-grid">
          {documents.map((doc, index) => {
            const id = doc.event_id || doc.global_id || `${doc.url}-${index}`;
            const expanded = expandedId === id;

            return (
              <article
                className={`document-card ${expanded ? "expanded" : ""}`}
                key={id}
              >
                <div className="document-card-top">
                  <span className={`source-badge source-${doc.source_type || "other"}`}>
                    {sourceLabel(doc.source_type)}
                  </span>

                  <div className="document-dates">
                    <span>Olay yılı<strong>{doc.event_year ?? "—"}</strong></span>
                    <span>Yayım<strong>{doc.published_date ?? "—"}</strong></span>
                  </div>
                </div>

                <h3>{displayTitle(doc)}</h3>

                <div className="document-meta">
                  <span>⌖ {doc.province || "—"}</span>
                  {doc.global_id && <span>ID {doc.global_id}</span>}
                </div>

                {doc.evidence_sentence && (
                  <div className="evidence-box">
                    <span>Kanıt cümlesi</span>
                    <p>{doc.evidence_sentence}</p>
                  </div>
                )}

                {expanded && doc.summary && (
                  <div className="summary-box">
                    <span>Özet</span>
                    <p>{doc.summary}</p>
                  </div>
                )}

                <div className="document-actions">
                  {doc.summary && (
                    <button onClick={() => setExpandedId(expanded ? null : id)}>
                      {expanded ? "Özeti Kapat" : "Özeti Göster"}
                    </button>
                  )}

                  {doc.url && (
                    <a href={doc.url} target="_blank" rel="noreferrer">
                      Kaynağı Aç ↗
                    </a>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
