import { useEffect, useMemo, useState } from 'react';
import type { InterventionProvinceDetail, TimelinePoint } from '../types';
import { useLinkedStore } from '../store/useLinkedStore';
import InterventionDelayLegacy from './InterventionDelayLegacy';

type Props = { detail: InterventionProvinceDetail | null; timeline: TimelinePoint[]; startYear: number; endYear: number; loading: boolean };
type Count = { name: string; count: number };
type Summary = { data_version: string; total_fact_rows: number; master_matched_rows: number; supported_matched_rows: number; unique_events: number; type_counts: Count[]; status_counts: Count[]; source_counts: Count[]; year_counts: { year: number; count: number }[]; caution: string };
type RecordRow = { global_id: string; event_order: string; fact_order: string; source_type: string; intervention_type: string; status: string; fact_text: string; intervention_method: string; year: number | null; actors: string[] };
type RecordResult = { total: number; records: RecordRow[] };
const API = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001').replace(/\/$/, '');
const tr: Record<string, string> = { completed: 'Tamamlandı (kaynak sınıflandırması)', ongoing: 'Devam ediyor', planned: 'Planlandı', reported_past: 'Geçmişte bildirildi', unclear: 'Belirsiz', chemical_control: 'Kimyasal', biological_control: 'Biyolojik', biotechnical_control: 'Biyoteknik', biocidal_winter_control: 'Kışlak biyosidal', mechanical_control: 'Mekanik', monitoring: 'İzleme', training_information: 'Eğitim / bilgilendirme', action_plan_coordination: 'Plan / koordinasyon', other: 'Diğer' };
const label = (s: string) => tr[s] || s;
export default function InterventionView(props: Props) {
  const province = useLinkedStore(s => s.selectedProvince);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [records, setRecords] = useState<RecordResult | null>(null);
  const [type, setType] = useState('');
  const [status, setStatus] = useState('');
  const [tab, setTab] = useState<'v22' | 'delay'>('v22');
  const [error, setError] = useState('');
  useEffect(() => { let active = true; fetch(`${API}/api/v22/interventions/summary`).then(r => { if (!r.ok) throw new Error(`API ${r.status}`); return r.json(); }).then(v => { if (active) setSummary(v); }).catch(e => { if (active) setError(String(e)); }); return () => { active = false; }; }, []);
  useEffect(() => { let active = true; const params = new URLSearchParams({ limit: '60' }); if (type) params.set('intervention_type', type); if (status) params.set('status', status); fetch(`${API}/api/v22/interventions/records?${params}`).then(r => { if (!r.ok) throw new Error(`API ${r.status}`); return r.json(); }).then(v => { if (active) setRecords(v); }).catch(e => { if (active) setError(String(e)); }); return () => { active = false; }; }, [type, status]);
  const max = useMemo(() => Math.max(1, ...(summary?.type_counts || []).map(x => x.count)), [summary]);
  return <section className="intervention-linked-view" style={{ display: 'grid', gap: 16 }}>
    <header className="climate-header"><div><span className="eyebrow">V22 · MÜDAHALE KANIT ANALİZİ</span><h2>Müdahale ve mücadele kayıtları</h2><p className="muted">Kaynak belgelerden çıkarılmış müdahale ifadeleri. Sayılar saha uygulaması sayısı değildir.</p></div></header>
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}><button type="button" onClick={() => setTab('v22')}>V22 müdahale analizi</button><button type="button" onClick={() => setTab('delay')}>Bildiri gecikme analizi (sabit)</button></div>
    {tab === 'delay' ? <InterventionDelayLegacy {...props} /> : <>
      {error && <p role="alert" style={{ color: '#fca5a5' }}>V22 API erişim hatası: {error}</p>}
      {!summary && !error && <p>V22 verileri yükleniyor…</p>}
      {summary && <>
        <div className="climate-metrics"><div><span>V22 müdahale bilgi satırı</span><strong>{summary.total_fact_rows.toLocaleString('tr-TR')}</strong></div><div><span>Final master ile eşleşen</span><strong>{summary.master_matched_rows.toLocaleString('tr-TR')}</strong></div><div><span>Desteklenen ve eşleşen</span><strong>{summary.supported_matched_rows.toLocaleString('tr-TR')}</strong></div><div><span>Benzersiz olay</span><strong>{summary.unique_events.toLocaleString('tr-TR')}</strong></div></div>
        <p className="muted" style={{ fontSize: 12 }}>İl filtresi: {province || 'Türkiye'}. Bu sürümde müdahale–il bağlantısı doğrulanmadığından il filtresi ulusal grafikleri değiştirmez. Yıl filtresi de yalnızca eski bildiri bölümünde görünür; V22 grafiklerine uygulanmaz. Olay yılı müdahale uygulama tarihi olmayabilir.</p>
        <section className="context-card"><h3>Müdahale türleri</h3><div style={{ display: 'grid', gap: 9 }}>{summary.type_counts.map(x => <div key={x.name}><div style={{ display: 'flex', justifyContent: 'space-between' }}><span>{label(x.name)}</span><strong>{x.count}</strong></div><div style={{ background: '#26344c', height: 10, borderRadius: 6 }}><div style={{ background: '#38bdf8', width: `${x.count / max * 100}%`, height: '100%', borderRadius: 6 }} /></div></div>)}</div></section>
        <section className="context-card"><h3>Kaynakta bildirilen durumlar</h3><div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>{summary.status_counts.map(x => <span key={x.name}>{label(x.name)}: <strong>{x.count}</strong></span>)}</div><p className="muted" style={{ fontSize: 12 }}>“Tamamlandı” ve “devam ediyor” değerleri metin çıkarımındaki etiketlerdir; sahada gerçekleşme ayrıca doğrulanmış değildir.</p></section>
        <section className="context-card"><h3>Olay yılına göre müdahale ifadeleri</h3><div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>{summary.year_counts.map(x => <span key={x.year}>{x.year}: <strong>{x.count}</strong></span>)}</div><p className="muted" style={{ fontSize: 12 }}>Bu yıllar olay düzeyindedir; müdahalenin yapıldığı yıl olduğu garanti edilmez.</p></section>
        <section className="context-card"><h3>Kaynak kanıtları</h3><div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 12 }}><select value={type} onChange={e => setType(e.target.value)}><option value="">Tüm türler</option>{summary.type_counts.map(x => <option key={x.name} value={x.name}>{label(x.name)}</option>)}</select><select value={status} onChange={e => setStatus(e.target.value)}><option value="">Tüm durumlar</option>{summary.status_counts.map(x => <option key={x.name} value={x.name}>{label(x.name)}</option>)}</select></div><p className="muted">{records ? `${records.total.toLocaleString('tr-TR')} desteklenen/eşleşen bilgi satırı · ilk ${records.records.length} kayıt` : 'Yükleniyor…'}</p><div style={{ display: 'grid', gap: 10 }}>{records?.records.map((x, i) => <article key={`${x.global_id}-${x.event_order}-${x.fact_order}-${i}`} style={{ padding: 12, border: '1px solid #334155', borderRadius: 8 }}><strong>{label(x.intervention_type)} · {label(x.status)}</strong><p style={{ margin: '8px 0' }}>{x.fact_text}</p><small>{x.global_id} / olay {x.event_order} / kanıt {x.fact_order} · {x.source_type} · {x.year ?? 'Yıl belirsiz'}</small>{x.actors?.length > 0 && <p className="muted" style={{ fontSize: 12 }}>Aktörler (olay düzeyi): {x.actors.join(', ')}</p>}</article>)}</div></section>
      </>}
    </>}
  </section>;
}
