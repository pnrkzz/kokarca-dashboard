import type { Metadata } from "../types";
import { useLinkedStore } from "../store/useLinkedStore";

type Props = { metadata: Metadata | null };
const modes = [["all","Tüm Dönem"],["single","Tek Yıl"],["range","Yıl Aralığı"],["cumulative","Birikimli"],["animation","Animasyon"]] as const;

export default function ControlBar({ metadata }: Props) {
  const mode = useLinkedStore((s) => s.timeMode);
  const product = useLinkedStore((s) => s.selectedProduct);
  const selectedYear = useLinkedStore((s) => s.selectedYear);
  const rangeStart = useLinkedStore((s) => s.rangeStart);
  const rangeEnd = useLinkedStore((s) => s.rangeEnd);
  const isPlaying = useLinkedStore((s) => s.isPlaying);
  const setMode = useLinkedStore((s) => s.setTimeMode);
  const setProduct = useLinkedStore((s) => s.setSelectedProduct);
  const setSelectedYear = useLinkedStore((s) => s.setSelectedYear);
  const setRangeStart = useLinkedStore((s) => s.setRangeStart);
  const setRangeEnd = useLinkedStore((s) => s.setRangeEnd);
  const setIsPlaying = useLinkedStore((s) => s.setIsPlaying);
  const resetAll = useLinkedStore((s) => s.resetAll);
  const years = metadata?.years ?? [];
  const firstYear = years[0] ?? 2017;
  const lastYear = years[years.length - 1] ?? 2026;

  return (
    <div className="controlbar">
      <div className="segmented-control">
        {modes.map(([key,label]) => (
          <button key={key} className={mode === key ? "selected" : ""} onClick={() => {
            if (key === "animation") {
              if (mode !== "animation") { setMode("animation"); setSelectedYear(firstYear); setIsPlaying(true); }
              else setIsPlaying(!isPlaying);
            } else setMode(key);
          }}>{key === "animation" && mode === "animation" ? (isPlaying ? "⏸ Durdur" : "▶ Devam") : label}</button>
        ))}
      </div>

      {mode === "single" && <label className="control-field compact"><span>Yıl</span><select value={selectedYear} onChange={(e)=>setSelectedYear(Number(e.target.value))}>{years.map(y=><option key={y} value={y}>{y}</option>)}</select></label>}
      {mode === "range" && <div className="range-fields">
        <label className="control-field compact"><span>Başlangıç</span><select value={rangeStart} onChange={(e)=>setRangeStart(Number(e.target.value))}>{years.map(y=><option key={y} value={y}>{y}</option>)}</select></label>
        <label className="control-field compact"><span>Bitiş</span><select value={rangeEnd} onChange={(e)=>setRangeEnd(Number(e.target.value))}>{years.map(y=><option key={y} value={y}>{y}</option>)}</select></label>
      </div>}
      {(mode === "cumulative" || mode === "animation") && <div className="active-year-pill">{mode === "cumulative" ? `${firstYear}–${selectedYear}` : selectedYear}</div>}

      <label className="control-field product-control"><span>Ürün</span><select value={product} onChange={(e)=>setProduct(e.target.value)}><option value="">Tüm Ürünler</option>{metadata?.products.map(item=><option key={item} value={item}>{item}</option>)}</select></label>
      <button className="reset-button" onClick={()=>resetAll(firstYear,lastYear)}>Sıfırla</button>
    </div>
  );
}
