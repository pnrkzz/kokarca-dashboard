import { useLinkedStore } from "../store/useLinkedStore";

const copy = {
  stc: ["STC v2", "3B Spaceâ€“Time Cube, slice, hover, animasyon ve baÄŸlÄ± filtreler Faz 6â€™da eklenecek."],
  climate: ["Ä°klim KanÄ±tÄ±", "LLM iklim anlatÄ±larÄ±, ERA5 ve ClimateGPT tutarlÄ±lÄ±k katmanlarÄ± Faz 5â€™te baÄŸlanacak."],
  interventions: ["MÃ¼dahale Linked View", "YÄ±l, kurum ve mÃ¼dahale tÃ¼rÃ¼ koordineli gÃ¶rÃ¼nÃ¼mÃ¼ Faz 4â€™te tamamlanacak."],
  documents: ["Belge Linked View", "BaÅŸlÄ±k, kaynak, yayÄ±m tarihi, kanÄ±t cÃ¼mlesi ve URL Faz 3â€™te baÄŸlanacak."],
  questions: ["AraÅŸtÄ±rma SorularÄ±", "HazÄ±r araÅŸtÄ±rma sorgularÄ± ve gÃ¶rsel yanÄ±tlar Faz 7â€™de eklenecek."],
} as const;

export default function PhasePlaceholder() {
  const activeView = useLinkedStore((s) => s.activeView);
  if (activeView === "map" || activeView === "methodology") return null;
  const [title, text] = copy[activeView];
  return <section className="phase-placeholder"><div className="placeholder-orbit"/><span className="eyebrow">LINKED VIEW MODULE</span><h2>{title}</h2><p>{text}</p><div className="placeholder-note">SeÃ§ili il ve Ã¼rÃ¼n baÄŸlamÄ± bu modÃ¼llere taÅŸÄ±nacak.</div></section>;
}

