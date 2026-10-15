import type { Metadata } from "../types";
import { useLinkedStore } from "../store/useLinkedStore";

type Props = { metadata: Metadata | null };

export default function TopBar({ metadata }: Props) {
  const province = useLinkedStore((s) => s.selectedProvince);
  const product = useLinkedStore((s) => s.selectedProduct);

  return (
    <header className="topbar">
      <div className="title-block">
        <h1>Kahverengi Kokarca</h1>
        <p>Mekânsal-Zamansal Geovisual Analytics Platformu</p>
        <span>Linked Views • Büyük Dil Modelleri • İstilacı Tür Analizi</span>
      </div>
      <div className="context-strip">
        <div className="context-chip"><span>BAĞLAM</span><strong>{province || "Türkiye"}</strong></div>
        <div className="context-chip"><span>ÜRÜN</span><strong>{product || "Tümü"}</strong></div>
        <div className="context-chip numeric"><span>OLAY</span><strong>{metadata?.event_count ?? "—"}</strong></div>
        <div className="context-chip numeric"><span>ZAMANSAL</span><strong>{metadata?.temporal_event_count ?? "—"}</strong></div>
      </div>
    </header>
  );
}
